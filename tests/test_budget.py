"""Tests for the /set-budget route."""

import app as app_module

from conftest import FakeError


VALID = {"amount": "125.5", "category_id": "3", "expense_date": "2026-09-18", "note": "lunch"}


def _budget_post(client, category_id, monthly_budget):
    return client.post("/set-budget", data={
        "category_id": str(category_id),
        "monthly_budget": str(monthly_budget),
    })


def test_valid_budget_update(auth_client, db, db_store):
    resp = _budget_post(auth_client, 1, 500.5)
    assert resp.status_code == 302
    assert db_store["categories"][(1, 1)] == 500.5
    assert db.committed


def test_negative_budget_rejected(auth_client, db, db_store):
    before = db_store["categories"][(2, 1)]
    resp = _budget_post(auth_client, 2, -50)
    assert resp.status_code == 302
    assert db_store["categories"][(2, 1)] == before
    assert not db.committed


def test_non_numeric_budget_rejected(auth_client, db, db_store):
    before = db_store["categories"][(2, 1)]
    resp = _budget_post(auth_client, 2, "abc")
    assert resp.status_code == 302
    assert db_store["categories"][(2, 1)] == before
    assert not db.committed


def test_missing_budget_rejected(auth_client, db, db_store):
    resp = auth_client.post("/set-budget", data={"category_id": "1"})
    assert resp.status_code == 302
    assert not db.committed


def test_other_users_category_rejected(auth_client, db, db_store):
    # category 10 belongs to user 2, not the logged-in user 1
    resp = _budget_post(auth_client, 10, 999)
    assert resp.status_code == 404
    assert db_store["categories"][(10, 2)] == 50.0


def test_missing_category_id_rejected(auth_client, db):
    resp = auth_client.post("/set-budget", data={"monthly_budget": "100"})
    assert resp.status_code == 404


def test_anonymous_access_redirects_to_login(client, db):
    resp = _budget_post(client, 1, 100)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

def test_db_error_during_budget_update_is_handled(auth_client, monkeypatch, db_store):
    """A DB failure inside set_budget's guarded block is caught by the
    route's except clause: the transaction is rolled back, the cursor and
    the connection are released, and the user is redirected to home. No
    exception propagates to the client as a 500.
    """
    from conftest import FakeCursor, FakeDB, FakeError

    class ExplodingCursor(FakeCursor):
        def execute(self, query, params=None):
            raise FakeError("boom")

    class ExplodingDB(FakeDB):
        def cursor(self, **kwargs):
            cursor = ExplodingCursor(self.store, self.fail)
            self.cursors.append(cursor)
            return cursor

    fake = ExplodingDB(db_store)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: fake)

    resp = _budget_post(auth_client, 1, 100)

    # no exception propagates: the route redirects to home
    assert resp.status_code == 302
    assert resp.headers["Location"].split("?")[0] == "/"
    # the failed transaction must be rolled back, never committed
    assert fake.rolled_back
    assert not fake.committed
    # the finally clause must still have released the resources
    assert all(c.closed for c in fake.cursors)
    assert fake.closed
