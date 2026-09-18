from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_wtf.csrf import CSRFProtect
import math
import mysql.connector
import os
import re
import secrets
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)

is_production = os.environ.get("FLASK_ENV", "").lower() == "production"
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if is_production:
        raise RuntimeError("SECRET_KEY must be configured in production.")
    secret_key = secrets.token_urlsafe(32)

app.config.update(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=is_production or os.environ.get("SESSION_COOKIE_SECURE") == "1",
)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"
csrf = CSRFProtect(app)

DEFAULT_CATEGORIES = ("Food", "Transport", "Entertainment", "Shopping", "Education", "Other")

# =====================================================
# ML RESULTS CACHE
# =====================================================
# In-memory cache for expensive ML calculations
# Key: user_id (int)
# Value: dict with ML results (prediction, mae, anomalies)
# Invalidated on expense mutations (add/update/delete)
_ml_cache = {}

def _invalidate_ml_cache(user_id):
    """Invalidate ML cache for specific user"""
    _ml_cache.pop(user_id, None)
    print(f"[CACHE] Invalidated ML cache for user_id={user_id}", flush=True)


class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_db_connection():

    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE", "defaultdb"),
        ssl_disabled=False
    )


def get_user_by_id(user_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, username, email FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        return User(row["id"], row["username"], row["email"]) if row else None
    finally:
        cursor.close()
        db.close()


@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        identity = request.form.get("identity", "").strip()
        password = request.form.get("password", "")
        db = None
        cursor = None
        row = None
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, username, email, password_hash
                FROM users
                WHERE username = %s OR email = %s
                LIMIT 1
                """,
                (identity, identity),
            )
            row = cursor.fetchone()
        except mysql.connector.Error:
            if db is not None:
                db.rollback()
            flash("Unable to sign in. Please try again.", "error")
            return render_template("login.html")
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row["id"], row["username"], row["email"]))
            return redirect(url_for("home"))
        flash("Invalid username/email or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("password_confirmation", "")

        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,50}", username):
            flash("Username must be 3–50 characters using letters, numbers, dots, hyphens, or underscores.", "error")
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 255:
            flash("Enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirmation:
            flash("Passwords do not match.", "error")
        else:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT id FROM users WHERE username = %s OR email = %s LIMIT 1",
                    (username, email),
                )
                if cursor.fetchone():
                    flash("That username or email is already registered.", "error")
                else:
                    cursor.execute(
                        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                        (username, email, generate_password_hash(password)),
                    )
                    user_id = cursor.lastrowid
                    cursor.executemany(
                        "INSERT INTO categories (user_id, name, monthly_budget) VALUES (%s, %s, %s)",
                        [(user_id, name, 0) for name in DEFAULT_CATEGORIES],
                    )
                    db.commit()
                    login_user(User(user_id, username, email))
                    return redirect(url_for("home"))
            except mysql.connector.Error:
                db.rollback()
                flash("Unable to create the account. Please try again.", "error")
            finally:
                cursor.close()
                db.close()

    return render_template("register.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# =====================================================
# HOME / DASHBOARD
# =====================================================

@app.route("/health")
def health():
    return "OK"

@app.route("/")
@login_required
def home():

    request_start = time.perf_counter()
    category_id = request.args.get("category_id")
    month = request.args.get("month")

    # Check if we should skip ML calculation (after expense mutation)
    skip_ml = session.pop('skip_ml_calculation', False)
    if skip_ml:
        print(f"[CACHE] Skipping ML calculation for user_id={current_user.id} (post-mutation)", flush=True)


    # =================================================
    # CONNECT TO DATABASE
    # =================================================

    db_start = time.perf_counter()

    db = None

    cursor = None

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        print(
            f"[PERF] Database connection: "
            f"{time.perf_counter() - db_start:.3f}s",
            flush=True
        )

        # =================================================
        # GET EXPENSES
        # =================================================

        query = """
            SELECT
                e.id,
                e.amount,
                e.note,
                e.expense_date,
                e.category_id,
                c.name AS category
            FROM expenses e
            JOIN categories c
                ON e.category_id = c.id
                AND c.user_id = e.user_id
            WHERE e.user_id = %s
        """

        params = [current_user.id]

        if category_id:
            query += " AND e.category_id = %s"
            params.append(category_id)

        if month:
            query += " AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s"
            params.append(month)

        query += " ORDER BY e.expense_date DESC"

        cursor.execute(
            query,
            params
        )

        expenses = cursor.fetchall()


        # =================================================
        # CALCULATE TOTAL AND COUNT IN PYTHON
        # =================================================

        total_spending = sum(
            float(expense["amount"])
            for expense in expenses
        )

        expense_count = len(expenses)


        # =================================================
        # CATEGORY-WISE SPENDING
        # =================================================

        category_totals = {}

        for expense in expenses:

            category_name = expense["category"]

            category_totals[category_name] = (
                category_totals.get(category_name, 0)
                + float(expense["amount"])
            )

        category_summary = [
            {
                "category": category_name,
                "total": round(total, 2)
            }
            for category_name, total in category_totals.items()
        ]

        category_summary.sort(
            key=lambda item: item["total"],
            reverse=True
        )


        # =================================================
        # MONTHLY SPENDING
        # =================================================

        # We intentionally do NOT apply the selected
        # month filter here.
        #
        # The ML model needs historical monthly data.
        #
        # Category filter is still respected.

        monthly_query = """
            SELECT
                DATE_FORMAT(e.expense_date, '%Y-%m') AS month,
                SUM(e.amount) AS total
            FROM expenses e
            WHERE e.user_id = %s
        """

        monthly_params = [current_user.id]

        if category_id:
            monthly_query += " AND e.category_id = %s"
            monthly_params.append(category_id)

        monthly_query += """
            GROUP BY
                DATE_FORMAT(e.expense_date, '%Y-%m')
            ORDER BY month
        """

        cursor.execute(
            monthly_query,
            monthly_params
        )

        monthly_summary = cursor.fetchall()

        for item in monthly_summary:
            item["total"] = float(item["total"])


        # =================================================
        # CATEGORIES + BUDGET ANALYSIS
        # =================================================

        budget_start = time.perf_counter()

        budget_month = month

        if not budget_month:
            budget_month = datetime.now().strftime("%Y-%m")

        budget_query = """
            SELECT
                c.id,
                c.name AS category,
                c.monthly_budget AS budget,
                COALESCE(SUM(e.amount), 0) AS actual
            FROM categories c
            LEFT JOIN expenses e
                ON c.id = e.category_id
                AND e.user_id = c.user_id
                AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s
            WHERE c.user_id = %s
            GROUP BY
                c.id,
                c.name,
                c.monthly_budget
            ORDER BY c.id
        """

        cursor.execute(
            budget_query,
            (budget_month, current_user.id)
        )

        budget_summary = cursor.fetchall()

        categories = []

        for item in budget_summary:

            item["budget"] = float(item["budget"])
            item["actual"] = float(item["actual"])

            categories.append({
                "id": item["id"],
                "name": item["category"],
                "monthly_budget": item["budget"]
            })
        categories.sort(key=lambda item: item["name"].lower())
        print(
            f"[PERF] Main database queries: "
            f"{time.perf_counter() - db_start:.3f}s",
            flush=True
        )
    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to load your data. Please try again.", "error")
        # Render the dashboard shell here rather than redirecting to "home":
        # a persistent database outage would otherwise redirect back into
        # this same route and loop. The finally block below still releases
        # any connection that was opened.
        return render_template(
            "index.html",
            expenses=[],
            categories=[],
            category_summary=[],
            monthly_summary=[],
            budget_summary=[],
            total_spending=0,
            expense_count=0,
            selected_category=category_id,
            selected_month=month,
            budget_month=month or datetime.now().strftime("%Y-%m"),
            prediction=None,
            prediction_month=None,
            prediction_message=None,
            mae=None,
            mae_message=None,
            insights=[],
            anomalies=[],
            anomaly_count=0,
            anomaly_message=None,
        )
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()



    # =====================================================
    # LINEAR REGRESSION PREDICTION
    # + MAE MODEL EVALUATION
    # =====================================================

    prediction = None

    prediction_month = None

    prediction_message = None

    mae = None

    mae_message = None


    # =================================================
    # ML CALCULATION WITH CACHING
    # =================================================
    ml_start = time.perf_counter()

    if skip_ml:
        # Skip ML calculation after expense mutation for immediate response
        prediction = None
        prediction_month = None
        prediction_message = "Recalculating prediction..."
        mae = None
        mae_message = "Recalculating model evaluation..."
        print(f"[PERF] Skipped ML calculation: {time.perf_counter() - ml_start:.3f}s", flush=True)
    else:
        # Check if ML results are cached
        cached_ml = _ml_cache.get(current_user.id)

        if cached_ml:
            # Use cached results
            prediction = cached_ml.get("prediction")
            prediction_month = cached_ml.get("prediction_month")
            prediction_message = cached_ml.get("prediction_message")
            mae = cached_ml.get("mae")
            mae_message = cached_ml.get("mae_message")
            print(f"[CACHE] Using cached ML prediction for user_id={current_user.id}", flush=True)
            print(f"[PERF] Linear Regression + MAE (cached): {time.perf_counter() - ml_start:.3f}s", flush=True)
        else:
            # Compute ML results
            # =================================================
            # NEED AT LEAST 2 MONTHS FOR PREDICTION
            # =================================================
            if len(monthly_summary) >= 2:


                # ---------------------------------------------
                # PREPARE DATA
                # ---------------------------------------------

                X = []

                y = []


                for index, item in enumerate(monthly_summary):

                    X.append(
                        [index + 1]
                    )

                    y.append(
                        item["total"]
                    )


                # ---------------------------------------------
                # TRAIN FINAL MODEL
                # ---------------------------------------------

                model = LinearRegression()


                model.fit(
                    X,
                    y
                )


                # ---------------------------------------------
                # PREDICT NEXT MONTH
                # ---------------------------------------------

                next_month_number = (
                    len(monthly_summary) + 1
                )


                predicted_value = model.predict(
                    [[next_month_number]]
                )[0]


                # Prevent negative prediction

                predicted_value = max(
                    0,
                    predicted_value
                )


                prediction = round(
                    float(predicted_value),
                    2
                )


                # ---------------------------------------------
                # FIND NEXT MONTH
                # ---------------------------------------------

                last_month = datetime.strptime(
                    monthly_summary[-1]["month"],
                    "%Y-%m"
                )


                if last_month.month == 12:

                    next_year = (
                        last_month.year + 1
                    )

                    next_month = 1

                else:

                    next_year = last_month.year

                    next_month = (
                        last_month.month + 1
                    )


                prediction_month = (
                    f"{next_year:04d}-{next_month:02d}"
                )


                # =================================================
                # MODEL EVALUATION USING MAE
                # =================================================
                #
                # We keep the latest month as test data.
                #
                # Example:
                #
                # May     -> training
                # June    -> training
                # July    -> training
                # August  -> test
                #
                # Then we predict August and compare it with
                # the actual August spending.
                # =================================================

                if len(monthly_summary) >= 4:


                    # -----------------------------------------
                    # TRAINING DATA
                    # -----------------------------------------

                    train_X = X[:-1]

                    train_y = y[:-1]


                    # -----------------------------------------
                    # TEST DATA
                    # -----------------------------------------

                    test_X = X[-1:]

                    test_y = y[-1:]


                    # -----------------------------------------
                    # CREATE EVALUATION MODEL
                    # -----------------------------------------

                    evaluation_model = LinearRegression()


                    evaluation_model.fit(
                        train_X,
                        train_y
                    )


                    # -----------------------------------------
                    # PREDICT TEST MONTH
                    # -----------------------------------------

                    test_prediction = (
                        evaluation_model.predict(
                            test_X
                        )
                    )


                    # -----------------------------------------
                    # CALCULATE MAE
                    # -----------------------------------------

                    mae_value = mean_absolute_error(
                        test_y,
                        test_prediction
                    )


                    mae = round(
                        float(mae_value),
                        2
                    )


                    mae_message = (
                        "Latest month was used as test data "
                        "for MAE evaluation."
                    )


                else:

                    mae_message = (
                        "Add expenses across at least 4 "
                        "different months to calculate MAE."
                    )


            else:

                prediction_message = (
                    "Add expenses across at least 2 "
                    "different months to generate a prediction."
                )


                mae_message = (
                    "Add expenses across at least 4 "
                    "different months to calculate MAE."
                )

            # Cache the computed ML results
            _ml_cache[current_user.id] = {
                "prediction": prediction,
                "prediction_month": prediction_month,
                "prediction_message": prediction_message,
                "mae": mae,
                "mae_message": mae_message
            }
            print(f"[CACHE] Stored ML prediction for user_id={current_user.id}", flush=True)
            print(
                f"[PERF] Linear Regression + MAE: "
                f"{time.perf_counter() - ml_start:.3f}s",
                flush=True
            )


    # =====================================================
    # STAGE 5 - SMART SPENDING INSIGHTS
    # =====================================================

    insights = []


    # -----------------------------------------------------
    # INSIGHT 1: HIGHEST SPENDING CATEGORY
    # -----------------------------------------------------

    if category_summary:

        highest_category = category_summary[0]

        insights.append(
            f"💡 {highest_category['category']} is your "
            f"highest spending category with "
            f"₹{highest_category['total']:.2f} spent."
        )


    # -----------------------------------------------------
    # INSIGHT 2: BUDGET WARNING
    # -----------------------------------------------------

    for item in budget_summary:

        budget = item["budget"]

        actual = item["actual"]


        if budget > 0:

            percentage = (
                actual / budget
            ) * 100


            # Budget exceeded

            if percentage >= 100:

                insights.append(
                    f"⚠️ You have exceeded your "
                    f"{item['category']} budget by "
                    f"₹{actual - budget:.2f}."
                )


            # 80% or more of budget used

            elif percentage >= 80:

                insights.append(
                    f"⚠️ You have used "
                    f"{percentage:.0f}% of your "
                    f"{item['category']} budget."
                )


    # -----------------------------------------------------
    # INSIGHT 3: MONTHLY SPENDING TREND
    # -----------------------------------------------------

    if len(monthly_summary) >= 2:

        current_month = monthly_summary[-1]["total"]

        previous_month = monthly_summary[-2]["total"]


        if previous_month > 0:

            change = (
                (current_month - previous_month)
                / previous_month
            ) * 100


            # Spending increased

            if change > 0:

                insights.append(
                    f"📈 Your spending increased by "
                    f"{change:.1f}% compared with the "
                    f"previous month."
                )


            # Spending decreased

            elif change < 0:

                insights.append(
                    f"📉 Your spending decreased by "
                    f"{abs(change):.1f}% compared with the "
                    f"previous month."
                )


            # Spending unchanged

            else:

                insights.append(
                    "➡️ Your spending is unchanged "
                    "compared with the previous month."
                )


    # -----------------------------------------------------
    # INSIGHT 4: GENERAL SPENDING STATUS
    # -----------------------------------------------------

    if total_spending == 0:

        insights.append(
            "💰 No spending recorded for the "
            "selected period."
        )


    elif total_spending < 1000:

        insights.append(
            "💰 Your spending is relatively low "
            "for the selected period."
        )


    # -----------------------------------------------------
    # INSIGHT 5: GOOD BUDGET CONTROL
    # -----------------------------------------------------

    budget_exceeded = False


    for item in budget_summary:

        if (
            item["budget"] > 0
            and item["actual"] > item["budget"]
        ):

            budget_exceeded = True

            break


    if not budget_exceeded and total_spending > 0:

        insights.append(
            "✅ You are currently within all "
            "category budgets."
        )


    # =====================================================
    # STAGE 6 - EXPENSE ANOMALY DETECTION
    # =====================================================

    anomaly_start = time.perf_counter()
    anomalies = []
    anomaly_message = None
    anomaly_count = 0

    if skip_ml:
        # Skip anomaly detection after expense mutation for immediate response
        anomalies = []
        anomaly_count = 0
        anomaly_message = "Recalculating anomaly detection..."
        print(f"[PERF] Skipped Isolation Forest: {time.perf_counter() - anomaly_start:.3f}s", flush=True)
    else:
        # Check if anomaly results are cached
        cached_ml = _ml_cache.get(current_user.id)

        if cached_ml and "anomalies" in cached_ml:
            # Use cached anomaly results
            anomalies = cached_ml.get("anomalies", [])
            anomaly_count = cached_ml.get("anomaly_count", 0)
            anomaly_message = cached_ml.get("anomaly_message")
            print(f"[CACHE] Using cached anomaly detection for user_id={current_user.id}", flush=True)
            print(f"[PERF] Isolation Forest (cached): {time.perf_counter() - anomaly_start:.3f}s", flush=True)
        else:
            # Compute anomaly detection
            # Isolation Forest needs multiple expense records
            # to identify unusual spending patterns.
            if len(expenses) >= 5:

                # Use expense amount as the ML feature.
                anomaly_data = [
                    [float(expense["amount"])]
                    for expense in expenses
                ]

                anomaly_model = IsolationForest(
                    contamination=0.10,
                    random_state=42
                )

                anomaly_predictions = anomaly_model.fit_predict(
                    anomaly_data
                )

                anomaly_scores = anomaly_model.decision_function(
                    anomaly_data
                )

                for index, prediction_result in enumerate(
                    anomaly_predictions
                ):

                    if prediction_result == -1:

                        anomaly = dict(expenses[index])

                        anomaly["amount"] = float(
                            anomaly["amount"]
                        )

                        anomaly["anomaly_score"] = round(
                            float(anomaly_scores[index]),
                            4
                        )

                        anomalies.append(anomaly)

                # Show the most unusual/high-value expenses first.
                anomalies.sort(
                    key=lambda item: item["amount"],
                    reverse=True
                )

                anomaly_count = len(anomalies)

                if anomaly_count > 0:
                    anomaly_message = (
                        f"{anomaly_count} potentially unusual "
                        f"expense(s) detected."
                    )
                else:
                    anomaly_message = (
                        "No unusual expenses detected."
                    )

            else:

                anomaly_message = (
                    "Add at least 5 expenses to detect "
                    "unusual spending patterns."
                )

            # Update cache with anomaly results
            cache_entry = _ml_cache.get(current_user.id, {})
            cache_entry.update({
                "anomalies": anomalies,
                "anomaly_count": anomaly_count,
                "anomaly_message": anomaly_message
            })
            _ml_cache[current_user.id] = cache_entry
            print(f"[CACHE] Stored anomaly detection for user_id={current_user.id}", flush=True)
            print(
                f"[PERF] Isolation Forest: "
                f"{time.perf_counter() - anomaly_start:.3f}s",
                flush=True
            )


    # =================================================
    # SEND DATA TO HTML
    # =================================================

    response = render_template(
        "index.html",

        expenses=expenses,

        categories=categories,

        category_summary=category_summary,

        monthly_summary=monthly_summary,

        budget_summary=budget_summary,

        total_spending=total_spending,

        expense_count=expense_count,

        selected_category=category_id,

        selected_month=month,

        budget_month=budget_month,

        prediction=prediction,

        prediction_month=prediction_month,

        prediction_message=prediction_message,

        mae=mae,

        mae_message=mae_message,

        insights=insights,

        anomalies=anomalies,

        anomaly_count=anomaly_count,

        anomaly_message=anomaly_message
    )
    print(
    f"[PERF] TOTAL HOME REQUEST: "
    f"{time.perf_counter() - request_start:.3f}s",
    flush=True
    )

    return response


# =====================================================
# ADD EXPENSE
# =====================================================

@app.route(
    "/add-expense",
    methods=["POST"]
)
@login_required
def add_expense():

    amount = request.form.get("amount", "").strip()

    category_id = request.form.get("category_id", type=int)

    expense_date = request.form.get("expense_date", "").strip()

    note = request.form.get("note", "").strip()


    if category_id is None:
        abort(404)


    # Validate amount: numeric, finite, and strictly positive.

    try:
        amount_value = round(float(amount), 2)
    except (TypeError, ValueError):
        amount_value = None

    if amount_value is None or not math.isfinite(amount_value) or amount_value <= 0:
        flash("Enter a valid, positive amount.", "error")
        return redirect(url_for("home"))


    # Validate expense_date: must be a real date in YYYY-MM-DD form.
    # Future dates are allowed.

    try:
        datetime.strptime(expense_date, "%Y-%m-%d")
    except ValueError:
        flash("Enter a valid expense date (YYYY-MM-DD).", "error")
        return redirect(url_for("home"))


    # Validate note length against the column size.

    if len(note) > 255:
        flash("Note must be 255 characters or fewer.", "error")
        return redirect(url_for("home"))


    db = None

    cursor = None

    try:
        db = get_db_connection()

        cursor = db.cursor()


        query = """
            INSERT INTO expenses
            (
                user_id,
                category_id,
                amount,
                note,
                expense_date
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """


        cursor.execute(
            "SELECT id FROM categories WHERE id = %s AND user_id = %s",
            (category_id, current_user.id),
        )
        if cursor.fetchone() is None:
            abort(404)

        cursor.execute(query, (current_user.id, category_id, amount_value, note, expense_date))


        db.commit()
    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to save the expense. Please try again.", "error")
        return redirect(url_for("home"))
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()

    # Invalidate ML cache and set skip flag for immediate redirect
    _invalidate_ml_cache(current_user.id)
    session['skip_ml_calculation'] = True

    return redirect(url_for("home"))


# =====================================================
# DELETE EXPENSE
# =====================================================

@app.route(
    "/delete-expense/<int:expense_id>",
    methods=["POST"]
)
@login_required
def delete_expense(expense_id):

    db = None

    cursor = None

    try:
        db = get_db_connection()

        cursor = db.cursor()


        query = """
            DELETE FROM expenses
            WHERE id = %s AND user_id = %s
        """


        cursor.execute(
            query,
            (expense_id, current_user.id)
        )

        if cursor.rowcount != 1:
            db.rollback()
            abort(404)

        db.commit()
    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to delete the expense. Please try again.", "error")
        return redirect(url_for("home"))
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()

    # Invalidate ML cache and set skip flag for immediate redirect
    _invalidate_ml_cache(current_user.id)
    session['skip_ml_calculation'] = True

    return redirect(url_for("home"))


# =====================================================
# EDIT EXPENSE
# =====================================================

@app.route(
    "/edit-expense/<int:expense_id>"
)
@login_required
def edit_expense(expense_id):

    db = None

    cursor = None

    try:
        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )


        query = """
            SELECT
                id,
                amount,
                category_id,
                note,
                expense_date
            FROM expenses
            WHERE id = %s AND user_id = %s
        """


        cursor.execute(
            query,
            (expense_id, current_user.id)
        )


        expense = cursor.fetchone()


        if expense is None:
            abort(404)

        cursor.execute(
            "SELECT id, name FROM categories WHERE user_id = %s ORDER BY name",
            (current_user.id,),
        )
        categories = cursor.fetchall()
    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to load the expense. Please try again.", "error")
        return redirect(url_for("home"))
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()

    return render_template(
        "edit_expense.html",
        expense=expense,
        categories=categories
    )


# =====================================================
# UPDATE EXPENSE
# =====================================================

@app.route(
    "/update-expense/<int:expense_id>",
    methods=["POST"]
)
@login_required
def update_expense(expense_id):

    amount = request.form.get("amount", "").strip()

    category_id = request.form.get("category_id", type=int)

    expense_date = request.form.get("expense_date", "").strip()

    note = request.form.get("note", "").strip()


    if category_id is None:
        abort(404)


    # Validate amount: numeric, finite, and strictly positive.

    try:
        amount_value = round(float(amount), 2)
    except (TypeError, ValueError):
        amount_value = None

    if amount_value is None or not math.isfinite(amount_value) or amount_value <= 0:
        flash("Enter a valid, positive amount.", "error")
        return redirect(url_for("home"))


    # Validate expense_date: must be a real date in YYYY-MM-DD form.
    # Future dates are allowed.

    try:
        datetime.strptime(expense_date, "%Y-%m-%d")
    except ValueError:
        flash("Enter a valid expense date (YYYY-MM-DD).", "error")
        return redirect(url_for("home"))


    # Validate note length against the column size.

    if len(note) > 255:
        flash("Note must be 255 characters or fewer.", "error")
        return redirect(url_for("home"))


    db = None

    cursor = None

    try:
        db = get_db_connection()

        cursor = db.cursor()


        query = """
            UPDATE expenses

            SET
                amount = %s,
                category_id = %s,
                expense_date = %s,
                note = %s

            WHERE id = %s AND user_id = %s
        """


        cursor.execute(
            "SELECT id FROM categories WHERE id = %s AND user_id = %s",
            (category_id, current_user.id),
        )
        if cursor.fetchone() is None:
            abort(404)

        cursor.execute(
            query,
            (amount_value, category_id, expense_date, note, expense_id, current_user.id)
        )

        if cursor.rowcount != 1:
            db.rollback()
            abort(404)

        db.commit()
    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to update the expense. Please try again.", "error")
        return redirect(url_for("home"))
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()

    # Invalidate ML cache and set skip flag for immediate redirect
    _invalidate_ml_cache(current_user.id)
    session['skip_ml_calculation'] = True

    return redirect(url_for("home"))


# =====================================================
# SET / UPDATE CATEGORY BUDGET
# =====================================================

@app.route(
    "/set-budget",
    methods=["POST"]
)
@login_required
def set_budget():

    category_id = request.form.get("category_id", type=int)

    monthly_budget = request.form.get("monthly_budget", "").strip()

    selected_month = request.form.get("month", "").strip()

    filter_category = request.form.get("filter_category_id", "").strip()


    if category_id is None:
        abort(404)


    db = get_db_connection()

    cursor = db.cursor()

    try:

        cursor.execute(
            "SELECT id FROM categories WHERE id = %s AND user_id = %s",
            (category_id, current_user.id),
        )

        if cursor.fetchone() is None:
            abort(404)


        try:
            budget_value = round(float(monthly_budget), 2)
        except (TypeError, ValueError):
            budget_value = None

        if budget_value is None or not math.isfinite(budget_value) or budget_value < 0:
            flash("Enter a valid, non-negative budget amount.", "error")
            return redirect(url_for("home", month=selected_month, category_id=filter_category))


        cursor.execute(
            """
            UPDATE categories
            SET monthly_budget = %s
            WHERE id = %s AND user_id = %s
            """,
            (budget_value, category_id, current_user.id),
        )

        if cursor.rowcount != 1:
            db.rollback()
            abort(404)

        db.commit()

    except mysql.connector.Error:
        if db is not None:
            db.rollback()
        flash("Unable to update the budget. Please try again.", "error")
        return redirect(url_for("home", month=selected_month, category_id=filter_category))

    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()


    flash("Budget updated successfully.", "success")

    return redirect(url_for("home", month=selected_month, category_id=filter_category))


# =====================================================
# GLOBAL ERROR HANDLERS
# =====================================================
# Present friendly pages for missing pages (404) and unhandled errors (500).
# Exception details and stack traces are logged server-side only and are
# never rendered to the client.

_ERROR_PAGE_404 = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page Not Found</title>
</head>
<body style="font-family:Arial,sans-serif;text-align:center;padding:4rem 1rem;">
<h1>404 - Page Not Found</h1>
<p>Sorry, the page you are looking for does not exist or may have moved.</p>
<a href="/">Back to home</a>
</body>
</html>"""

_ERROR_PAGE_500 = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Something Went Wrong</title>
</head>
<body style="font-family:Arial,sans-serif;text-align:center;padding:4rem 1rem;">
<h1>Something went wrong</h1>
<p>An unexpected error occurred. The incident has been logged, so please try
again in a moment.</p>
<a href="/">Back to home</a>
</body>
</html>"""


@app.errorhandler(404)
def handle_404(error):
    """Return a friendly page for missing pages or resources."""
    return _ERROR_PAGE_404, 404


@app.errorhandler(500)
def handle_500(error):
    """Log the failure and show a friendly page, hiding all exception detail.

    Rolls back the current transaction when a database connection is available
    through the application's existing connection helper. If the database
    itself is the source of the failure, no connection is available and the
    rollback is skipped silently.
    """
    app.logger.exception("Unhandled server error")

    db = None
    try:
        db = get_db_connection()
        db.rollback()
    except Exception:
        # No connection is available (e.g. the database is down); there is
        # nothing to roll back.
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    return _ERROR_PAGE_500, 500

# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )
