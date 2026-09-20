"""Tests for the secure email OTP-based Forgot Password system."""
from datetime import datetime, timedelta, timezone
import pytest
from werkzeug.security import check_password_hash

import app as app_module
from app import app


def test_forgot_password_get_renders_form(client, db):
    resp = client.get("/forgot-password")
    assert resp.status_code == 200
    assert b"Reset your password" in resp.data
    assert b'name="email"' in resp.data
    assert b"Send verification code" in resp.data


def test_forgot_password_authenticated_redirects_home(auth_client):
    resp = auth_client.get("/forgot-password")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/")


def test_forgot_password_invalid_email_format(client, db):
    resp = client.post("/forgot-password", data={"email": "not-an-email"})
    assert resp.status_code == 200
    assert b"Enter a valid email address." in resp.data


def test_forgot_password_success_flow(client, db, db_store):
    sent_emails = []

    def mock_sender(to_email, otp):
        sent_emails.append({"to": to_email, "otp": otp})
        return True

    app_module._email_sender_hook = mock_sender

    resp = client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/reset-password")

    # Verify email was dispatched with 6-digit OTP
    assert len(sent_emails) == 1
    otp = sent_emails[0]["otp"]
    assert len(otp) == 6
    assert otp.isdigit()

    # Verify database record: OTP stored ONLY as a hash, never plaintext!
    assert len(db_store["password_resets"]) == 1
    record = db_store["password_resets"][0]
    assert record["email"] == "ambuj@example.com"
    assert record["user_id"] == 1
    assert record["otp_hash"] != otp
    assert check_password_hash(record["otp_hash"], otp)
    assert record["is_used"] == 0
    assert record["attempts"] == 0


def test_forgot_password_anti_enumeration(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append((to, otp))

    resp = client.post("/forgot-password", data={"email": "unregistered@example.com"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/reset-password")

    # Follow redirect and verify identical flash message is shown
    with client.session_transaction() as sess:
        flashed = sess.get("_flashes", [])
        assert any("If an account with that email exists" in msg for _, msg in flashed)

    # No email sent and no token stored
    assert len(sent_emails) == 0
    assert len(db_store["password_resets"]) == 0


def test_forgot_password_resend_cooldown(client, db, db_store):
    app_module._email_sender_hook = lambda to, otp: True

    # First request succeeds
    resp1 = client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert resp1.status_code == 302

    # Second request immediately afterwards triggers cooldown
    resp2 = client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert resp2.status_code == 200
    assert b"before requesting another code" in resp2.data


def test_forgot_password_hourly_rate_limit(client, db, db_store):
    app_module._email_sender_hook = lambda to, otp: True
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Populate 5 prior requests in the past hour (older than 60s cooldown)
    for i in range(5):
        db_store["password_resets"].append({
            "id": i + 1,
            "user_id": 1,
            "email": "ambuj@example.com",
            "otp_hash": "hash",
            "created_at": now - timedelta(minutes=5 + i * 2),
            "expires_at": now + timedelta(minutes=10),
            "attempts": 0,
            "is_used": 1,
            "ip_address": "127.0.0.1",
        })

    resp = client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert resp.status_code == 200
    assert b"Too many password reset requests for this email" in resp.data


def test_reset_password_get_renders_form(client, db):
    resp = client.get("/reset-password?email=ambuj@example.com")
    assert resp.status_code == 200
    assert b"Set new password" in resp.data
    assert b'name="otp"' in resp.data
    assert b'name="password"' in resp.data
    assert b'name="password_confirmation"' in resp.data
    assert b"ambuj@example.com" in resp.data


def test_reset_password_validation_failures(client, db):
    # Short password
    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": "123456",
        "password": "short",
        "password_confirmation": "short",
    })
    assert resp.status_code == 200
    assert b"Password must be at least 8 characters." in resp.data

    # Mismatched passwords
    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": "123456",
        "password": "password123",
        "password_confirmation": "password999",
    })
    assert resp.status_code == 200
    assert b"Passwords do not match." in resp.data

    # Invalid OTP format (non-numeric or not 6 digits)
    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": "123",
        "password": "password123",
        "password_confirmation": "password123",
    })
    assert resp.status_code == 200
    assert b"Verification code must be exactly 6 digits." in resp.data


def test_reset_password_success(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    # 1. Request OTP
    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert len(sent_emails) == 1
    otp = sent_emails[0]

    # 2. Reset password
    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/login")

    # 3. Verify OTP is marked is_used = 1
    assert db_store["password_resets"][-1]["is_used"] == 1

    # 4. Verify user password was updated
    user = db_store["users"][1]
    assert check_password_hash(user["password_hash"], "new_secure_password_123")
    assert not check_password_hash(user["password_hash"], "password123")

    # 5. Verify user can now log in with the new password
    login_resp = client.post("/login", data={
        "identity": "ambuj",
        "password": "new_secure_password_123",
    })
    assert login_resp.status_code == 302
    assert login_resp.headers["Location"].endswith("/")


def test_reset_password_wrong_otp_decrements_attempts(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    real_otp = sent_emails[0]
    wrong_otp = "000000" if real_otp != "000000" else "111111"

    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": wrong_otp,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp.status_code == 200
    assert b"Invalid verification code" in resp.data
    assert b"4 attempt(s) remaining" in resp.data
    assert db_store["password_resets"][-1]["attempts"] == 1
    assert db_store["password_resets"][-1]["is_used"] == 0


def test_reset_password_max_attempts_lockout(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    wrong_otp = "000000"

    # Make 5 failed attempts
    for i in range(4):
        resp = client.post("/reset-password", data={
            "email": "ambuj@example.com",
            "otp": wrong_otp,
            "password": "new_secure_password_123",
            "password_confirmation": "new_secure_password_123",
        })
        assert resp.status_code == 200

    # 5th failed attempt locks out the code
    resp5 = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": wrong_otp,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp5.status_code == 302
    assert resp5.headers["Location"].endswith("/forgot-password")
    assert db_store["password_resets"][-1]["is_used"] == 1


def test_reset_password_expired_otp(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    otp = sent_emails[0]

    # Artificially expire the OTP
    db_store["password_resets"][-1]["expires_at"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)

    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/forgot-password")
    assert db_store["password_resets"][-1]["is_used"] == 1


def test_reset_password_replay_attack_rejected(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    otp = sent_emails[0]

    # First reset succeeds
    resp1 = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp1.status_code == 302

    # Replay with same OTP fails
    resp2 = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "another_new_password_123",
        "password_confirmation": "another_new_password_123",
    })
    assert resp2.status_code == 200
    assert b"Invalid or expired verification code." in resp2.data


def test_reset_password_supersedes_prior_otps(client, db, db_store):
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    # First OTP
    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    otp1 = sent_emails[0]

    # Simulate time passing past cooldown
    db_store["password_resets"][-1]["created_at"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=70)

    # Second OTP
    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    otp2 = sent_emails[1]

    # OTP 1 should now be marked is_used = 1
    assert db_store["password_resets"][0]["is_used"] == 1

    # Attempting to use superseded OTP 1 fails
    resp1 = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp1,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert b"Invalid verification code" in resp1.data
    assert not check_password_hash(db_store["users"][1]["password_hash"], "new_secure_password_123")

    # Using OTP 2 succeeds
    resp2 = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp2,
        "password": "new_secure_password_123",
        "password_confirmation": "new_secure_password_123",
    })
    assert resp2.status_code == 302
    assert resp2.headers["Location"].endswith("/login")


def test_reset_password_csrf_protection(monkeypatch, db):
    # Enable CSRF protection explicitly
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", True)
    monkeypatch.setitem(app.config, "TESTING", True)

    with app.test_client() as c:
        # POST without CSRF token must be rejected with 400 Bad Request
        resp = c.post("/reset-password", data={
            "email": "ambuj@example.com",
            "otp": "123456",
            "password": "password123",
            "password_confirmation": "password123",
        })
        assert resp.status_code == 400


def test_backward_compatibility_other_user_login(client, db):
    # Unmodified account can still log in with its original password
    resp = client.post("/login", data={
        "identity": "other",
        "password": "password456",
    })
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/")


def test_reset_password_atomic_concurrency_race(client, db, db_store, monkeypatch):
    """If another concurrent request consumes the OTP first, rowcount != 1 triggers rollback."""
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    client.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert len(sent_emails) == 1
    otp = sent_emails[0]

    # Intercept check_password_hash to simulate a concurrent request setting is_used = 1
    orig_check = app_module.check_password_hash

    def race_check(p_hash, password):
        # Concurrent request completes reset right before our UPDATE
        db_store["password_resets"][-1]["is_used"] = 1
        return orig_check(p_hash, password)

    monkeypatch.setattr(app_module, "check_password_hash", race_check)

    # Attempt to reset with the same valid OTP
    resp = client.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "another_new_password_123",
        "password_confirmation": "another_new_password_123",
    })

    # The reset attempt must be rejected
    assert resp.status_code == 200
    assert b"Invalid or expired verification code." in resp.data

    # Transaction was rolled back and password must NOT have changed
    assert db.rolled_back is True
    assert not check_password_hash(db_store["users"][1]["password_hash"], "another_new_password_123")
    assert check_password_hash(db_store["users"][1]["password_hash"], "password123")


def test_session_invalidated_after_password_reset(db, db_store):
    """Active sessions are invalidated after a password reset without requiring DB schema changes."""
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["TESTING"] = True

    browser1 = app.test_client()
    browser2 = app.test_client()

    # 1. User logs in from Browser 1
    login_resp = browser1.post("/login", data={
        "identity": "ambuj",
        "password": "password123",
    })
    assert login_resp.status_code == 302
    assert login_resp.headers["Location"].endswith("/")

    # Browser 1 session is authenticated and can access home page
    home_resp = browser1.get("/")
    assert home_resp.status_code == 200

    # 2. Reset password from Browser 2 using forgot-password flow
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    fp_resp = browser2.post("/forgot-password", data={"email": "ambuj@example.com"})
    assert fp_resp.status_code == 302
    assert len(sent_emails) == 1
    otp = sent_emails[0]

    reset_resp = browser2.post("/reset-password", data={
        "email": "ambuj@example.com",
        "otp": otp,
        "password": "new_super_secure_pass_123",
        "password_confirmation": "new_super_secure_pass_123",
    })
    assert reset_resp.status_code == 302
    assert reset_resp.headers["Location"].endswith("/login")

    # 3. Old session from Browser 1 attempts to access home page -> must be redirected to login
    post_reset_home = browser1.get("/")
    assert post_reset_home.status_code == 302
    assert "/login" in post_reset_home.headers["Location"]

    # 4. Old password can no longer log in
    old_login = browser1.post("/login", data={
        "identity": "ambuj",
        "password": "password123",
    })
    assert old_login.status_code == 200
    assert b"Invalid username/email or password." in old_login.data

    # 5. New password logs in successfully
    new_login = browser1.post("/login", data={
        "identity": "ambuj",
        "password": "new_super_secure_pass_123",
    })
    assert new_login.status_code == 302
    assert new_login.headers["Location"].endswith("/")

    # New session can access home page
    new_home = browser1.get("/")
    assert new_home.status_code == 200


def test_proxyfix_client_ip_handling(client, db, db_store):
    """ProxyFix unpacks X-Forwarded-For to remote_addr and _get_client_ip returns it."""
    sent_emails = []
    app_module._email_sender_hook = lambda to, otp: sent_emails.append(otp)

    forwarded_ip = "198.51.100.42"
    resp = client.post(
        "/forgot-password",
        data={"email": "ambuj@example.com"},
        headers={"X-Forwarded-For": forwarded_ip}
    )
    assert resp.status_code == 302

    # Verify IP address recorded in password_resets record is the forwarded client IP
    assert len(db_store["password_resets"]) == 1
    assert db_store["password_resets"][0]["ip_address"] == forwarded_ip

