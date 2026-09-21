from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_wtf.csrf import CSRFProtect
import math
import mysql.connector
import os
import re
import secrets
import time
import threading
from datetime import datetime, timedelta, timezone
import resend
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
import hashlib
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1
)

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
# OTP & PASSWORD RESET CONFIGURATION
# =====================================================
OTP_EXPIRY_MINUTES = int(os.environ.get("OTP_EXPIRY_MINUTES", "10"))
MAX_OTP_ATTEMPTS = int(os.environ.get("MAX_OTP_ATTEMPTS", "5"))
OTP_RESEND_COOLDOWN_SECONDS = int(os.environ.get("OTP_RESEND_COOLDOWN_SECONDS", "60"))
MAX_OTP_HOURLY_EMAIL = int(os.environ.get("MAX_OTP_HOURLY_EMAIL", "5"))
MAX_OTP_HOURLY_IP = int(os.environ.get("MAX_OTP_HOURLY_IP", "10"))
MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "Expense Tracker <noreply@expensetracker.local>")

# Hook for overriding email delivery during testing or mocking
_email_sender_hook = None

# =====================================================
# ML RESULTS CACHE
# =====================================================
# In-memory cache for expensive ML calculations
# Key: user_id (int)
# Value: dict with ML results (prediction, mae, anomalies)
# Invalidated on expense mutations (add/update/delete)
_ml_cache = {}

def _ml_cache_key(user_id, category_id, month):
    """Build the ML cache key from the active dashboard filters.

    Prediction, evaluation and anomaly results are all derived from a
    filter-specific expense set, so the key must carry those filters.
    Without them a result computed for one combination (e.g. no filter)
    could be served for another (e.g. Category=Entertainment).
    """
    return (user_id, category_id, month)


def _invalidate_ml_cache(user_id):
    """Invalidate every ML cache entry for a user.

    Entries are keyed by (user_id, category_id, month), so a single
    expense mutation must drop all of that user filter variants.
    """
    for key in [k for k in _ml_cache if k[0] == user_id]:
        _ml_cache.pop(key, None)
    print(f"[CACHE] Invalidated ML cache for user_id={user_id}", flush=True)

def _walk_forward_evaluate(monthly_totals, min_train=2):
    """Walk-forward (expanding-window) evaluation of the forecast model.

    For each available month N, a Linear Regression is trained on months
    1..N-1 only and then used to predict month N. Every prediction is
    therefore genuinely out-of-sample: no month is ever predicted by a
    model that saw that month (or any later month) during training.

    Returns a dict of aggregate metrics (mae, rmse, r2, mape) plus the
    number of out-of-sample months evaluated. Each metric is None when it
    is not mathematically defined for the collected predictions:
      - MAPE is None if any actual value is zero (division by zero).
      - R2 is None unless there are 2+ predictions with non-zero variance.
    Returns an explicit insufficient-data state (all None, n=0) when too
    few months exist to produce even one out-of-sample prediction.
    """
    values = [float(v) for v in monthly_totals]

    # Need at least min_train months of history before the first
    # out-of-sample month, plus that month itself.
    if len(values) < min_train + 1:
        return {"mae": None, "rmse": None, "r2": None, "mape": None, "n": 0}

    y_true = []
    y_pred = []

    for split in range(min_train, len(values)):
        train_X = [[i + 1] for i in range(split)]
        train_y = values[:split]

        model = LinearRegression()
        model.fit(train_X, train_y)

        # Predict month index `split`, which the model has never seen.
        predicted = model.predict([[split + 1]])[0]
        predicted = max(0.0, float(predicted))

        y_true.append(values[split])
        y_pred.append(predicted)

    return _eval_regression(y_true, y_pred)


def _eval_regression(y_true, y_pred):
    """Compute regression metrics for a set of predictions.

    Pure function: takes parallel actual/predicted values and returns only
    the metrics that are mathematically defined for the available data.

    - MAPE is reported only when every actual value is non-zero, otherwise
      it would divide by zero.
    - R2 is reported only when the actual values have non-zero variance.
      With a single test month (the dashboard hold-out) R2 is undefined, so
      None is returned rather than a misleading perfect score.
    - An empty input means insufficient historical data; all metrics None.
    """
    y_true = [float(v) for v in y_true]
    y_pred = [float(v) for v in y_pred]
    n = len(y_true)

    if n == 0:
        return {"mae": None, "rmse": None, "r2": None, "mape": None, "n": 0}

    errors = [actual - predicted for actual, predicted in zip(y_true, y_pred)]

    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)

    if all(v != 0 for v in y_true):
        mape = sum(abs(e) / abs(t) for e, t in zip(errors, y_true)) / n * 100
    else:
        mape = None

    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    if n >= 2 and ss_tot > 0:
        ss_res = sum(e * e for e in errors)
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = None

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 2) if r2 is not None else None,
        "mape": round(mape, 1) if mape is not None else None,
        "n": n,
    }


def _anomaly_rate(n_evaluated, n_anomalies):
    """Anomaly rate as a percentage, or None when nothing was evaluated.

    Isolation Forest runs unsupervised here: the application has no
    ground-truth anomaly labels, so classification accuracy, precision and
    recall cannot be calculated for it.
    """
    n_evaluated = int(n_evaluated or 0)
    n_anomalies = int(n_anomalies or 0)
    if n_evaluated <= 0:
        return None
    return round(n_anomalies / n_evaluated * 100, 1)



class User(UserMixin):
    def __init__(self, user_id, username, email, password_hash=None):
        self.id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash


def _hash_sig(password_hash):
    """Compute a truncated cryptographic digest of the password hash to validate sessions."""
    if not password_hash:
        return ""
    return hashlib.sha256(password_hash.encode("utf-8")).hexdigest()[:16]


# =====================================================
# DATABASE CONNECTION
# =====================================================

# Shared MySQL connection pool.
# A fresh TLS-encrypted connection was previously opened for every
# request, and again by the login user loader, which dominated response
# time against a remote database. The pool is created once and reused;
# connections return to it when the existing finally blocks call
# db.close(), which for a pooled connection is a return-to-pool.
_db_pool = None
_db_pool_lock = threading.Lock()


def _get_db_pool():
    """Return the shared pool, creating it once on first real use.

    Creation is deferred to the first request so that merely importing
    app.py (as the test suite does) never opens a database connection.
    """
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name="expense_tracker_pool",
                    pool_size=5,
                    host=os.environ.get("MYSQL_HOST"),
                    port=int(os.environ.get("MYSQL_PORT", "3306")),
                    user=os.environ.get("MYSQL_USER"),
                    password=os.environ.get("MYSQL_PASSWORD"),
                    database=os.environ.get("MYSQL_DATABASE", "defaultdb"),
                    ssl_disabled=False,
                )
    return _db_pool


def get_db_connection():
    """Borrow a connection from the shared pool.

    Callers return it with the close() call already present in every
    finally block; for a pooled connection that puts it back in the
    pool instead of destroying it.
    """
    return _get_db_pool().get_connection()


def _month_range(month):
    """Return (month_start, next_month_start) datetimes for a YYYY-MM string.

    Falls back to the current calendar month when month is empty or invalid,
    matching the previous default behaviour. Used in place of
    DATE_FORMAT(expense_date) predicates so an index-friendly range is used.
    """
    try:
        start = datetime.strptime(month, "%Y-%m")
    except (TypeError, ValueError):
        start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if start.month == 12:
        nxt = start.replace(year=start.year + 1, month=1)
    else:
        nxt = start.replace(month=start.month + 1)
    return start, nxt


def get_user_by_id(user_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        return User(row["id"], row["username"], row["email"], row.get("password_hash")) if row else None
    finally:
        cursor.close()
        db.close()


@login_manager.user_loader
def load_user(user_id):
    try:
        user = get_user_by_id(user_id)
    except mysql.connector.Error:
        app.logger.warning("[Auth] Database unavailable during session load; logging user out.")
        return None
    if user is None:
        return None
    sig = session.get("_password_hash_sig")
    if sig is not None and user.password_hash:
        if sig != _hash_sig(user.password_hash):
            return None
    return user


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
        except mysql.connector.Error as exc:
            app.logger.warning(
                "[Auth] Login database error: %s - %s",
                type(exc).__name__,
                str(exc),
            )
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
            login_user(User(row["id"], row["username"], row["email"], row["password_hash"]))
            session["_password_hash_sig"] = _hash_sig(row["password_hash"])
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
                    pwd_hash = generate_password_hash(password)
                    cursor.execute(
                        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                        (username, email, pwd_hash),
                    )
                    user_id = cursor.lastrowid
                    cursor.executemany(
                        "INSERT INTO categories (user_id, name, monthly_budget) VALUES (%s, %s, %s)",
                        [(user_id, name, 0) for name in DEFAULT_CATEGORIES],
                    )
                    db.commit()
                    login_user(User(user_id, username, email, pwd_hash))
                    session["_password_hash_sig"] = _hash_sig(pwd_hash)
                    return redirect(url_for("home"))
            except mysql.connector.Error as exc:
                app.logger.warning(
                    "[Auth] Register database error: %s - %s",
                    type(exc).__name__,
                    str(exc),
                )
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
# PASSWORD RESET HELPERS & ROUTES
# =====================================================

def _generate_otp():
    """Return a cryptographically secure 6-digit numeric string."""
    return f"{secrets.randbelow(1000000):06d}"


def _get_client_ip():
    """Extract client IP safely from request.remote_addr (configured with ProxyFix)."""
    return request.remote_addr or "127.0.0.1"


def _send_otp_email(to_email, otp):
    """Deliver a 6-digit OTP code to the recipient's email address.

    Uses the Resend Python SDK. Never logs passwords, API keys, or plaintext OTPs.
    """
    global _email_sender_hook
    if _email_sender_hook is not None:
        return _email_sender_hook(to_email, otp)

    if app.config.get("TESTING"):
        return True

    resend_api_key = os.environ.get("RESEND_API_KEY")
    if not resend_api_key:
        app.logger.warning("[Email] RESEND_API_KEY not configured; email delivery skipped.")
        return False

    resend.api_key = resend_api_key

    text_content = (
        "Hello,\n\n"
        "We received a request to reset your password for your Expense Tracker account.\n"
        f"Your 6-digit verification code is: {otp}\n\n"
        f"This code will expire in {OTP_EXPIRY_MINUTES} minutes.\n\n"
        "If you did not request this code, please ignore this email. Your password will remain unchanged.\n\n"
        "— Expense Tracker Team\n"
    )
    html_content = f"""<!DOCTYPE html>
<html>
<body style="font-family: system-ui, -apple-system, sans-serif; background-color: #0F1712; padding: 24px; color: #1E2A22;">
  <div style="max-width: 480px; margin: 0 auto; background-color: #F4EFE0; padding: 32px; border-radius: 8px;">
    <h2 style="margin-top: 0; color: #1E2A22;">Password Reset Verification</h2>
    <p style="color: #4A5A4E; line-height: 1.5;">We received a request to reset your password for your Expense Tracker account.</p>
    <p style="color: #4A5A4E; line-height: 1.5;">Use the following 6-digit verification code to complete your password reset:</p>
    <div style="text-align: center; margin: 28px 0;">
      <span style="display: inline-block; font-size: 32px; font-weight: 700; letter-spacing: 6px; padding: 12px 24px; background: #1E2A22; color: #F4EFE0; border-radius: 6px;">
        {otp}
      </span>
    </div>
    <p style="color: #4A5A4E; font-size: 0.9em; line-height: 1.4;">
      This code is valid for <strong>{OTP_EXPIRY_MINUTES} minutes</strong>.
      If you did not request a password reset, you can safely ignore this email; your account remains secure.
    </p>
    <hr style="border: none; border-top: 1px solid #D9D0B7; margin: 24px 0;">
    <p style="color: #6C7A70; font-size: 0.8em; margin-bottom: 0;">— Expense Tracker Team</p>
  </div>
</body>
</html>"""

    try:
        resend.Emails.send({
            "from": MAIL_DEFAULT_SENDER,
            "to": to_email,
            "subject": "Your Password Reset Verification Code - Ledger",
            "text": text_content,
            "html": html_content
        })
        return True
    except Exception as exc:
        app.logger.warning(
            "[Email] Failed to deliver email: %s - %s",
            type(exc).__name__,
            str(exc),
        )
        return False


def _check_otp_rate_limits(cursor, email, ip_address):
    """Enforce rate limits on password reset requests:
    1. Per-email cooldown (e.g. 60 seconds)
    2. Per-email maximum hourly requests (e.g. 5/hour)
    3. Per-IP maximum hourly requests (e.g. 10/hour)
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    one_hour_ago = now - timedelta(hours=1)

    # 1. Cooldown check (most recent request for this email)
    cursor.execute(
        """
        SELECT created_at
        FROM password_resets
        WHERE email = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (email,)
    )
    latest = cursor.fetchone()
    if latest and latest.get("created_at"):
        created_at = latest["created_at"]
        if isinstance(created_at, datetime):
            created_cmp = created_at.replace(tzinfo=None) if created_at.tzinfo else created_at
            diff_seconds = (now - created_cmp).total_seconds()
            if diff_seconds < OTP_RESEND_COOLDOWN_SECONDS:
                remaining = max(1, int(OTP_RESEND_COOLDOWN_SECONDS - diff_seconds))
                return False, f"Please wait {remaining} second(s) before requesting another code."

    # 2. Hourly per-email check
    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM password_resets
        WHERE email = %s AND created_at >= %s
        """,
        (email, one_hour_ago)
    )
    email_count_row = cursor.fetchone()
    if email_count_row and email_count_row["count"] >= MAX_OTP_HOURLY_EMAIL:
        return False, "Too many password reset requests for this email. Please try again later."

    # 3. Hourly per-IP check
    if ip_address:
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM password_resets
            WHERE ip_address = %s AND created_at >= %s
            """,
            (ip_address, one_hour_ago)
        )
        ip_count_row = cursor.fetchone()
        if ip_count_row and ip_count_row["count"] >= MAX_OTP_HOURLY_IP:
            return False, "Too many password reset requests from your network. Please try again later."

    return True, ""


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        ip_address = _get_client_ip()

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 255:
            flash("Enter a valid email address.", "error")
            return render_template("forgot_password.html")

        db = None
        cursor = None
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)

            allowed, rate_msg = _check_otp_rate_limits(cursor, email, ip_address)
            if not allowed:
                flash(rate_msg, "error")
                return render_template("forgot_password.html")

            cursor.execute(
                "SELECT id, username, email FROM users WHERE email = %s LIMIT 1",
                (email,)
            )
            user_row = cursor.fetchone()

            if user_row:
                user_id = user_row["id"]
                otp = _generate_otp()
                otp_hash = generate_password_hash(otp)
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)

                # Invalidate any prior active OTPs for this user
                cursor.execute(
                    "UPDATE password_resets SET is_used = 1 WHERE user_id = %s AND is_used = 0",
                    (user_id,)
                )

                cursor.execute(
                    """
                    INSERT INTO password_resets
                    (user_id, email, otp_hash, created_at, expires_at, attempts, is_used, ip_address)
                    VALUES (%s, %s, %s, %s, %s, 0, 0, %s)
                    """,
                    (user_id, email, otp_hash, now, expires_at, ip_address)
                )
                db.commit()

                _send_otp_email(email, otp)
            else:
                # Anti-enumeration: perform dummy hash computation to balance response timing
                _ = generate_password_hash("dummy_enumeration_mitigation_hash")

        except mysql.connector.Error as exc:
            app.logger.warning(
                "[Auth] Forgot-password database error: %s - %s",
                type(exc).__name__,
                str(exc),
            )
            if db is not None:
                db.rollback()
            flash("Unable to process request. Please try again.", "error")
            return render_template("forgot_password.html")
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

        session["reset_email"] = email
        flash("If an account with that email exists, a 6-digit verification code has been sent.", "info")
        return redirect(url_for("reset_password"))

    return render_template("forgot_password.html")


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        otp = request.form.get("otp", "").strip()
        password = request.form.get("password", "")
        confirmation = request.form.get("password_confirmation", "")

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 255:
            flash("Enter a valid email address.", "error")
            return render_template("reset_password.html", email=email)

        if not re.fullmatch(r"\d{6}", otp):
            flash("Verification code must be exactly 6 digits.", "error")
            return render_template("reset_password.html", email=email)

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("reset_password.html", email=email)

        if password != confirmation:
            flash("Passwords do not match.", "error")
            return render_template("reset_password.html", email=email)

        db = None
        cursor = None
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT id, user_id, email, otp_hash, attempts, is_used, expires_at
                FROM password_resets
                WHERE email = %s AND is_used = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (email,)
            )
            reset_row = cursor.fetchone()

            if not reset_row:
                flash("Invalid or expired verification code.", "error")
                return render_template("reset_password.html", email=email)

            reset_id = reset_row["id"]
            user_id = reset_row["user_id"]
            attempts = reset_row["attempts"]
            expires_at = reset_row["expires_at"]

            # Max attempts check
            if attempts >= MAX_OTP_ATTEMPTS:
                cursor.execute("UPDATE password_resets SET is_used = 1 WHERE id = %s", (reset_id,))
                db.commit()
                flash("Maximum verification attempts exceeded. Please request a new code.", "error")
                return redirect(url_for("forgot_password"))

            # Expiry check
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if isinstance(expires_at, datetime):
                expires_cmp = expires_at.replace(tzinfo=None) if expires_at.tzinfo else expires_at
                is_expired = now > expires_cmp
            else:
                is_expired = False

            if is_expired:
                cursor.execute("UPDATE password_resets SET is_used = 1 WHERE id = %s", (reset_id,))
                db.commit()
                flash("This verification code has expired. Please request a new one.", "error")
                return redirect(url_for("forgot_password"))

            # Verify OTP
            if not check_password_hash(reset_row["otp_hash"], otp):
                cursor.execute(
                    "UPDATE password_resets SET attempts = attempts + 1 WHERE id = %s AND is_used = 0",
                    (reset_id,)
                )
                db.commit()
                cursor.execute(
                    "SELECT attempts FROM password_resets WHERE id = %s",
                    (reset_id,)
                )
                att_row = cursor.fetchone()
                current_attempts = att_row["attempts"] if att_row else (attempts + 1)

                if current_attempts >= MAX_OTP_ATTEMPTS:
                    cursor.execute(
                        "UPDATE password_resets SET is_used = 1 WHERE id = %s",
                        (reset_id,)
                    )
                    db.commit()
                    flash("Maximum verification attempts exceeded. Please request a new code.", "error")
                    return redirect(url_for("forgot_password"))

                remaining = MAX_OTP_ATTEMPTS - current_attempts
                flash(f"Invalid verification code. You have {remaining} attempt(s) remaining.", "error")
                return render_template("reset_password.html", email=email)

            # OTP verified successfully!
            # Atomically consume OTP with conditional update to prevent concurrency race
            cursor.execute(
                "UPDATE password_resets SET is_used = 1 WHERE id = %s AND is_used = 0",
                (reset_id,)
            )
            if cursor.rowcount != 1:
                db.rollback()
                flash("Invalid or expired verification code.", "error")
                return render_template("reset_password.html", email=email)

            new_password_hash = generate_password_hash(password)
            cursor.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_password_hash, user_id)
            )
            cursor.execute(
                "UPDATE password_resets SET is_used = 1 WHERE user_id = %s AND is_used = 0",
                (user_id,)
            )
            db.commit()

            session.pop("reset_email", None)
            flash("Password reset successfully. You can now sign in with your new password.", "success")
            return redirect(url_for("login"))

        except mysql.connector.Error as exc:
            app.logger.warning(
                "[Auth] Reset-password database error: %s - %s",
                type(exc).__name__,
                str(exc),
            )
            if db is not None:
                db.rollback()
            flash("Unable to reset password. Please try again.", "error")
            return render_template("reset_password.html", email=email)
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

    email = request.args.get("email", "").strip().lower() or session.get("reset_email", "")
    return render_template("reset_password.html", email=email)

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

        # Inclusive-start / exclusive-end bounds for the selected month.
        # Replacing the DATE_FORMAT(expense_date) call with a range keeps
        # the idx_expenses_user_date index usable.
        month_start, next_month_start = _month_range(month)

        if category_id:
            query += " AND e.category_id = %s"
            params.append(category_id)

        if month:
            query += " AND e.expense_date >= %s AND e.expense_date < %s"
            params.extend([month_start, next_month_start])

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
                AND e.expense_date >= %s
                AND e.expense_date < %s
            WHERE c.user_id = %s
            GROUP BY
                c.id,
                c.name,
                c.monthly_budget
            ORDER BY c.id
        """

        budget_month_start, budget_next_month_start = _month_range(budget_month)

        cursor.execute(
            budget_query,
            (budget_month_start, budget_next_month_start, current_user.id)
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

    rmse = None

    r2 = None

    mape = None

    eval_months = 0


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
        rmse = None

        r2 = None

        mape = None

        eval_months = 0
        print(f"[PERF] Skipped ML calculation: {time.perf_counter() - ml_start:.3f}s", flush=True)
    else:
        # Check if ML results are cached
        cached_ml = _ml_cache.get(
            _ml_cache_key(current_user.id, category_id, month)
        )

        if cached_ml:
            # Use cached results
            prediction = cached_ml.get("prediction")
            prediction_month = cached_ml.get("prediction_month")
            prediction_message = cached_ml.get("prediction_message")
            mae = cached_ml.get("mae")
            mae_message = cached_ml.get("mae_message")
            rmse = cached_ml.get("rmse")
            r2 = cached_ml.get("r2")
            mape = cached_ml.get("mape")
            eval_months = cached_ml.get("eval_months", 0)
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
                # MODEL EVALUATION - WALK-FORWARD
                # =================================================
                #
                # Each month is predicted by a model trained only on
                # the months before it (expanding window), so every
                # score is out-of-sample. Requires at least 3 months:
                # two to train, one to evaluate.
                #
                # Example with 5 months of history:
                #
                #   train M1..M2  -> predict M3
                #   train M1..M3  -> predict M4
                #   train M1..M4  -> predict M5
                # =================================================

                eval_metrics = _walk_forward_evaluate(y)

                if eval_metrics["n"] > 0:

                    mae = eval_metrics["mae"]

                    rmse = eval_metrics["rmse"]

                    r2 = eval_metrics["r2"]

                    mape = eval_metrics["mape"]

                    eval_months = eval_metrics["n"]

                    mae_message = (
                        f"Walk-forward evaluation across {eval_months} "
                        f"unseen month(s)."
                    )

                else:

                    mae_message = (
                        "Add expenses across at least 3 "
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
            _ml_cache[_ml_cache_key(current_user.id, category_id, month)] = {
                "prediction": prediction,
                "prediction_month": prediction_month,
                "prediction_message": prediction_message,
                "mae": mae,
                "mae_message": mae_message,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "eval_months": eval_months
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

    anomaly_evaluated = 0

    anomaly_rate = None

    score_min = None

    score_max = None

    score_mean = None

    if skip_ml:
        # Skip anomaly detection after expense mutation for immediate response
        anomalies = []
        anomaly_count = 0
        anomaly_message = "Recalculating anomaly detection..."
        anomaly_evaluated = 0
        anomaly_rate = None
        score_min = None
        score_max = None
        score_mean = None
        print(f"[PERF] Skipped Isolation Forest: {time.perf_counter() - anomaly_start:.3f}s", flush=True)
    else:
        # Check if anomaly results are cached
        cached_ml = _ml_cache.get(
            _ml_cache_key(current_user.id, category_id, month)
        )

        if cached_ml and "anomalies" in cached_ml:
            # Use cached anomaly results
            anomalies = cached_ml.get("anomalies", [])
            anomaly_count = cached_ml.get("anomaly_count", 0)
            anomaly_message = cached_ml.get("anomaly_message")
            anomaly_evaluated = cached_ml.get("anomaly_evaluated", 0)
            anomaly_rate = cached_ml.get("anomaly_rate")
            score_min = cached_ml.get("score_min")
            score_max = cached_ml.get("score_max")
            score_mean = cached_ml.get("score_mean")
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

                # Unsupervised evaluation stats. No ground-truth labels
                # exist, so accuracy/precision/recall are not reported.
                anomaly_evaluated = len(expenses)

                anomaly_rate = _anomaly_rate(anomaly_evaluated, anomaly_count)

                scores = [float(s) for s in anomaly_scores]

                score_min = round(min(scores), 4)

                score_max = round(max(scores), 4)

                score_mean = round(sum(scores) / len(scores), 4)

            else:

                anomaly_message = (
                    "Add at least 5 expenses to detect "
                    "unusual spending patterns."
                )

                anomaly_evaluated = len(expenses)

                anomaly_rate = None

            # Update cache with anomaly results
            cache_entry = _ml_cache.get(
                _ml_cache_key(current_user.id, category_id, month), {}
            )
            cache_entry.update({
                "anomalies": anomalies,
                "anomaly_count": anomaly_count,
                "anomaly_message": anomaly_message,
                "anomaly_evaluated": anomaly_evaluated,
                "anomaly_rate": anomaly_rate,
                "score_min": score_min,
                "score_max": score_max,
                "score_mean": score_mean
            })
            _ml_cache[_ml_cache_key(current_user.id, category_id, month)] = cache_entry
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

        rmse=rmse,

        r2=r2,

        mape=mape,

        eval_months=eval_months,

        insights=insights,

        anomalies=anomalies,

        anomaly_count=anomaly_count,

        anomaly_message=anomaly_message,

        anomaly_evaluated=anomaly_evaluated,

        anomaly_rate=anomaly_rate,

        score_min=score_min,

        score_max=score_max,

        score_mean=score_mean
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
