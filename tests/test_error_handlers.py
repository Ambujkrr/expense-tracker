"""Tests for the global 404 and 500 error handlers."""
import logging

import app as app_module
from app import app
from conftest import FakeDB


def _disable_propagation(monkeypatch):
    """TESTING=True would otherwise re-raise exceptions past the handlers."""
    monkeypatch.setitem(app.config, "PROPAGATE_EXCEPTIONS", False)


def test_404_handler_returns_friendly_page(client):
    resp = client.get("/this-page-does-not-exist")
    assert resp.status_code == 404
    assert b"Page Not Found" in resp.data


def test_404_handler_used_for_route_abort(auth_client, db):
    # set_budget aborts(404) for a category that does not exist
    resp = auth_client.post(
        "/set-budget",
        data={"category_id": "999", "monthly_budget": "10"},
    )
    assert resp.status_code == 404
    assert b"Page Not Found" in resp.data


def test_500_handler_returns_friendly_page_and_logs(auth_client, monkeypatch, caplog):
    _disable_propagation(monkeypatch)

    def boom():
        raise RuntimeError("kaboom-secret")

    monkeypatch.setattr(app_module, "get_db_connection", boom)

    with caplog.at_level(logging.ERROR):
        resp = auth_client.get("/")

    assert resp.status_code == 500
    body = resp.data.lower()
    assert b"kaboom-secret" not in body      # exception message never leaked
    assert b"traceback" not in body          # no stack trace leaked
    assert b"page not found" not in body     # a genuine 500 page, not the 404
    assert "Unhandled server error" in caplog.text


def test_500_handler_rolls_back_when_db_available(auth_client, monkeypatch, db_store):
    _disable_propagation(monkeypatch)

    class CursorBoomDB(FakeDB):
        # a connection is available, but a query blows up mid-request
        def cursor(self, **kwargs):
            raise RuntimeError("cursor boom")

    fake = CursorBoomDB(db_store)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: fake)

    resp = auth_client.get("/")

    assert resp.status_code == 500
    assert fake.rolled_back        # error handler rolled the transaction back
    assert fake.closed             # and released the connection
    assert not fake.committed
    assert b"cursor boom" not in resp.data


def test_500_handler_survives_when_db_is_down(auth_client, monkeypatch):
    _disable_propagation(monkeypatch)

    def boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(app_module, "get_db_connection", boom)

    resp = auth_client.get("/")

    # the missing connection must not break the friendly 500 response
    assert resp.status_code == 500
    assert b"Something went wrong" in resp.data
