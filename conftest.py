"""Shared pytest fixtures for the ExpenseTracker test suite.

No test in this suite ever touches a real MySQL database: every route
that needs a connection gets a FakeDB through monkeypatch.
"""
import pytest

import mysql.connector

import app as app_module
from app import app, User


# ------------------------------------------------------------------
# Fake database objects
# ------------------------------------------------------------------

class FakeDBError(mysql.connector.Error):
    """Stands in for mysql.connector.Error inside tests."""


class FakeCursor:
    """A scripted cursor. Query dispatch is driven by ``self.rows``,
    a dict mapping a substring of the SQL (upper-cased, whitespace
    normalised) to the value returned by fetchall()."""

    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail
        self.rowcount = 1
        self._row = None
        self._list = None
        self.closed = False
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append(query)
        if self.fail:
            raise FakeDBError("simulated database failure")
        text = " ".join(query.split()).upper()

        # expense mutations
        if "INSERT INTO EXPENSES" in text:
            self.store.setdefault("inserted", []).append(params)
        elif "UPDATE EXPENSES" in text:
            # params end with (expense_id, user_id)
            expense_id = params[-2]
            if expense_id in self.store["owned_expenses"]:
                self.rowcount = 1
                self.store.setdefault("updated", []).append(params)
            else:
                self.rowcount = 0
        elif "DELETE FROM EXPENSES" in text:
            expense_id = params[0]
            if expense_id in self.store["owned_expenses"]:
                self.rowcount = 1
                self.store.setdefault("deleted", []).append(params)
            else:
                self.rowcount = 0
        elif "UPDATE CATEGORIES" in text:
            # params: (budget, category_id, user_id)
            budget, cat_id, user_id = params
            if (cat_id, user_id) in self.store["categories"]:
                self.store["categories"][(cat_id, user_id)] = budget
                self.rowcount = 1
            else:
                self.rowcount = 0
        # category ownership check
        elif text.startswith("SELECT ID FROM CATEGORIES"):
            # (category_id, user_id) present => the category exists and is owned
            cat_id, user_id = params
            self._row = {"id": cat_id} if (cat_id, user_id) in self.store["categories"] else None
        elif "SELECT ID, NAME FROM CATEGORIES" in text:
            self._list = self.store["category_rows"]
        # edit-expense SELECT of a single expense
        elif "SELECT ID, AMOUNT" in text:
            self._row = self.store["expense_rows"].get(params[0]) if params else None
        # dashboard queries (order matters: monthly summary declares AS MONTH)
        elif "AS MONTH" in text:
            self._list = self.store["monthly_rows"]
        elif "C.MONTHLY_BUDGET AS BUDGET" in text:
            self._list = self.store["budget_rows"]
        elif "FROM EXPENSES E" in text:
            self._list = self.store["expense_list"]
        else:
            self._row = {"id": 1, "name": "Food"}

    def fetchall(self):
        if self._list is not None:
            return self._list
        return [self._row] if self._row is not None else []

    def fetchone(self):
        return self._row

    def close(self):
        self.closed = True


class FakeDB:
    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail
        self.rolled_back = False
        self.committed = False
        self.closed = False
        self.cursors = []

    def cursor(self, **kwargs):
        cursor = FakeCursor(self.store, self.fail)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        if self.fail:
            raise FakeDBError("commit failed")
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _make_store():
    """Fresh per-test DB state.

    categories maps (category_id, owner_user_id) -> budget.
    category 1,2 belong to user 1; category 10 belongs to user 2.
    """
    return {
        "categories": {(1, 1): 0.0, (2, 1): 100.0, (10, 2): 50.0},
        "category_rows": [{"id": 1, "name": "Food"}, {"id": 2, "name": "Transport"}],
        "expense_rows": {
            7: {"id": 7, "amount": 10.0, "category_id": 1, "note": "n", "expense_date": "2026-09-01"},
        },
        "expense_list": [{"id": 7, "amount": 10.0, "note": "n", "expense_date": "2026-09-01",
                          "category_id": 1, "category": "Food"}],
        # expense ids the logged-in user actually owns (rowcount simulation)
        "owned_expenses": {7},
        "monthly_rows": [{"month": "2026-09", "total": 10.0}],
        "budget_rows": [{"id": 1, "category": "Food", "budget": 100.0, "actual": 10.0}],
        "inserted": [], "updated": [], "deleted": [],
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Clear module/session-level caches so tests cannot leak into each other."""
    app_module._ml_cache.clear()
    yield
    app_module._ml_cache.clear()


@pytest.fixture
def db_store():
    return _make_store()


@pytest.fixture
def db(db_store, monkeypatch):
    """A controllable FakeDB; available in a test via the ``db`` fixture."""
    fake = FakeDB(db_store)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: fake)
    return fake


@pytest.fixture
def failing_db(db_store, monkeypatch):
    """A FakeDB whose every operation raises a DB error."""
    fake = FakeDB(db_store, fail=True)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _fake_user_loader(monkeypatch):
    """The user_loader would otherwise hit the DB; stub it instead."""
    monkeypatch.setattr(app_module, "get_user_by_id", lambda uid: User(int(uid), "tester", "t@example.com"))


@pytest.fixture
def client():
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(client):
    """A client whose session is logged in as user 1."""
    with client.session_transaction() as sess:
        sess["_user_id"] = "1"
    return client


# Expose the fake error type for tests that simulate failures.
FakeError = FakeDBError