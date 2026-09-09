from flask import Flask, render_template, request, redirect
import mysql.connector
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest

app = Flask(__name__)


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_db_connection():

    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE"),
        ssl_disabled=False
    )

# =====================================================
# HOME / DASHBOARD
# =====================================================

@app.route("/")
def home():

    category_id = request.args.get("category_id")
    month = request.args.get("month")


    # =================================================
    # CONNECT TO DATABASE
    # =================================================

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)


    # =================================================
    # GET CATEGORIES
    # =================================================

    cursor.execute("""
        SELECT
            id,
            name,
            monthly_budget
        FROM categories
        ORDER BY name
    """)

    categories = cursor.fetchall()


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
        WHERE 1=1
    """

    params = []


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
    # TOTAL SPENDING
    # =================================================

    total_query = """
        SELECT
            COALESCE(SUM(e.amount), 0) AS total
        FROM expenses e
        WHERE 1=1
    """

    total_params = []


    if category_id:

        total_query += " AND e.category_id = %s"

        total_params.append(category_id)


    if month:

        total_query += " AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s"

        total_params.append(month)


    cursor.execute(
        total_query,
        total_params
    )


    total_result = cursor.fetchone()

    total_spending = float(
        total_result["total"]
    )


    # =================================================
    # EXPENSE COUNT
    # =================================================

    count_query = """
        SELECT
            COUNT(*) AS total_count
        FROM expenses e
        WHERE 1=1
    """

    count_params = []


    if category_id:

        count_query += " AND e.category_id = %s"

        count_params.append(category_id)


    if month:

        count_query += " AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s"

        count_params.append(month)


    cursor.execute(
        count_query,
        count_params
    )


    count_result = cursor.fetchone()

    expense_count = count_result["total_count"]


    # =================================================
    # CATEGORY-WISE SPENDING
    # =================================================

    category_query = """
        SELECT
            c.name AS category,
            COALESCE(SUM(e.amount), 0) AS total
        FROM expenses e
        JOIN categories c
            ON e.category_id = c.id
        WHERE 1=1
    """

    category_params = []


    if category_id:

        category_query += " AND e.category_id = %s"

        category_params.append(category_id)


    if month:

        category_query += " AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s"

        category_params.append(month)


    category_query += """
        GROUP BY
            c.id,
            c.name
        ORDER BY total DESC
    """


    cursor.execute(
        category_query,
        category_params
    )


    category_summary = cursor.fetchall()


    for item in category_summary:

        item["total"] = float(
            item["total"]
        )


    # =================================================
    # MONTHLY SPENDING
    # =================================================
    #
    # IMPORTANT:
    # We intentionally do NOT apply the selected
    # month filter here.
    #
    # The ML model needs historical monthly data.
    #
    # Category filter is still respected.
    # =================================================

    monthly_query = """
        SELECT
            DATE_FORMAT(e.expense_date, '%Y-%m') AS month,
            SUM(e.amount) AS total
        FROM expenses e
        WHERE 1=1
    """

    monthly_params = []


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

        item["total"] = float(
            item["total"]
        )


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


    # =================================================
    # BUDGET ANALYSIS
    # =================================================

    budget_month = month


    if not budget_month:

        budget_month = datetime.now().strftime(
            "%Y-%m"
        )


    budget_query = """
        SELECT
            c.id,
            c.name AS category,
            c.monthly_budget AS budget,
            COALESCE(SUM(e.amount), 0) AS actual
        FROM categories c
        LEFT JOIN expenses e
            ON c.id = e.category_id
            AND DATE_FORMAT(e.expense_date, '%Y-%m') = %s
        GROUP BY
            c.id,
            c.name,
            c.monthly_budget
        ORDER BY c.id
    """


    cursor.execute(
        budget_query,
        (budget_month,)
    )


    budget_summary = cursor.fetchall()


    for item in budget_summary:

        item["budget"] = float(
            item["budget"]
        )

        item["actual"] = float(
            item["actual"]
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

    anomalies = []
    anomaly_message = None
    anomaly_count = 0

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


    # =================================================
    # CLOSE DATABASE
    # =================================================

    cursor.close()

    db.close()


    # =================================================
    # SEND DATA TO HTML
    # =================================================

    return render_template(
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


# =====================================================
# ADD EXPENSE
# =====================================================

@app.route(
    "/add-expense",
    methods=["POST"]
)
def add_expense():

    amount = request.form["amount"]

    category_id = request.form["category_id"]

    expense_date = request.form["expense_date"]

    note = request.form["note"]


    db = get_db_connection()

    cursor = db.cursor()


    query = """
        INSERT INTO expenses
        (
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
            %s
        )
    """


    cursor.execute(
        query,
        (
            category_id,
            amount,
            note,
            expense_date
        )
    )


    db.commit()


    cursor.close()

    db.close()


    return redirect("/")


# =====================================================
# DELETE EXPENSE
# =====================================================

@app.route(
    "/delete-expense/<int:expense_id>",
    methods=["POST"]
)
def delete_expense(expense_id):

    db = get_db_connection()

    cursor = db.cursor()


    query = """
        DELETE FROM expenses
        WHERE id = %s
    """


    cursor.execute(
        query,
        (expense_id,)
    )


    db.commit()


    cursor.close()

    db.close()


    return redirect("/")


# =====================================================
# EDIT EXPENSE
# =====================================================

@app.route(
    "/edit-expense/<int:expense_id>"
)
def edit_expense(expense_id):

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
        WHERE id = %s
    """


    cursor.execute(
        query,
        (expense_id,)
    )


    expense = cursor.fetchone()


    cursor.close()

    db.close()


    if expense is None:

        return "Expense not found", 404


    return render_template(
        "edit_expense.html",
        expense=expense
    )


# =====================================================
# UPDATE EXPENSE
# =====================================================

@app.route(
    "/update-expense/<int:expense_id>",
    methods=["POST"]
)
def update_expense(expense_id):

    amount = request.form["amount"]

    category_id = request.form["category_id"]

    expense_date = request.form["expense_date"]

    note = request.form["note"]


    db = get_db_connection()

    cursor = db.cursor()


    query = """
        UPDATE expenses

        SET
            amount = %s,
            category_id = %s,
            expense_date = %s,
            note = %s

        WHERE id = %s
    """


    cursor.execute(
        query,
        (
            amount,
            category_id,
            expense_date,
            note,
            expense_id
        )
    )


    db.commit()


    cursor.close()

    db.close()


    return redirect("/")


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )