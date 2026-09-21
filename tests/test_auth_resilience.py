"""Tests for load_user resilience and auth diagnostic logging."""
import logging

import mysql.connector

import app as app_module
from app import app


def test_load_user_returns_none_on_db_error(monkeypatch, caplog):
    """When the database is unreachable, load_user returns None instead of crashing."""

    def boom(user_id):
        raise mysql.connector.errors.ProgrammingError(
            msg="Access denied for user ''@'localhost'",
            errno=1045,
        )

    monkeypatch.setattr(app_module, "get_user_by_id", boom)

    app.config["WTF_CSRF_ENABLED"] = False
    app.config["TESTING"] = True

    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["_user_id"] = "1"

        with caplog.at_level(logging.WARNING):
            resp = c.get("/")

        # User is logged out and redirected to login, NOT a 500 error
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]
        assert "Database unavailable during session load" in caplog.text


def test_login_logs_db_error_on_failure(client, monkeypatch, caplog, failing_db):
    """POST /login logs the database error details when the DB is unreachable."""
    with caplog.at_level(logging.WARNING):
        resp = client.post("/login", data={
            "identity": "ambuj",
            "password": "password123",
        })

    assert resp.status_code == 200
    assert b"Unable to sign in" in resp.data
    assert "Login database error" in caplog.text


def test_forgot_password_logs_db_error_on_failure(client, monkeypatch, caplog, failing_db):
    """POST /forgot-password logs the database error details when the DB is unreachable."""
    with caplog.at_level(logging.WARNING):
        resp = client.post("/forgot-password", data={
            "email": "ambuj@example.com",
        })

    assert resp.status_code == 200
    assert b"Unable to process request" in resp.data
    assert "Forgot-password database error" in caplog.text


def test_reset_password_logs_db_error_on_failure(client, monkeypatch, caplog, failing_db):
    """POST /reset-password logs the database error details when the DB is unreachable."""
    with caplog.at_level(logging.WARNING):
        resp = client.post("/reset-password", data={
            "email": "ambuj@example.com",
            "otp": "123456",
            "password": "new_password_123",
            "password_confirmation": "new_password_123",
        })

    assert resp.status_code == 200
    assert b"Unable to reset password" in resp.data
    assert "Reset-password database error" in caplog.text
