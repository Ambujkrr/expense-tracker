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
            # Honor the same category/month filters the real dashboard
            # query carries, so tests can assert filter-specific
            # expense sets and filter-scoped anomaly results.
            rows = list(self.store["expense_list"])
            rest = list(params[1:]) if params else []
            if "AND E.CATEGORY_ID = %S" in text:
                category_id = rest.pop(0)
                rows = [
                    r for r in rows
                    if str(r["category_id"]) == str(category_id)
                ]
            if "E.EXPENSE_DATE >= %S" in text:
                start = str(rest.pop(0))[:10]
                end = str(rest.pop(0))[:10]
                rows = [
                    r for r in rows
                    if start <= str(r["expense_date"])[:10] < end
                ]
            self._list = rows
        # password reset queries
        elif "COUNT(*) AS COUNT FROM PASSWORD_RESETS WHERE EMAIL" in text:
            email, since = params
            cnt = sum(
                1 for r in self.store.get("password_resets", [])
                if r["email"] == email and r["created_at"] >= since
            )
            self._row = {"count": cnt}
        elif "COUNT(*) AS COUNT FROM PASSWORD_RESETS WHERE IP_ADDRESS" in text:
            ip, since = params
            cnt = sum(
                1 for r in self.store.get("password_resets", [])
                if r.get("ip_address") == ip and r["created_at"] >= since
            )
            self._row = {"count": cnt}
        elif "FROM PASSWORD_RESETS WHERE EMAIL = %S AND IS_USED = 0" in text:
            email = params[0]
            matching = [
                r for r in self.store.get("password_resets", [])
                if r["email"] == email and r["is_used"] == 0
            ]
            self._row = matching[-1] if matching else None
        elif "FROM PASSWORD_RESETS WHERE EMAIL = %S" in text:
            email = params[0]
            matching = [
                r for r in self.store.get("password_resets", [])
                if r["email"] == email
            ]
            self._row = matching[-1] if matching else None
        elif "INSERT INTO PASSWORD_RESETS" in text:
            next_id = len(self.store.setdefault("password_resets", [])) + 1
            reset_record = {
                "id": next_id,
                "user_id": params[0],
                "email": params[1],
                "otp_hash": params[2],
                "created_at": params[3],
                "expires_at": params[4],
                "attempts": 0,
                "is_used": 0,
                "ip_address": params[5],
            }
            self.store["password_resets"].append(reset_record)
            self.rowcount = 1
        elif "UPDATE PASSWORD_RESETS SET IS_USED = 1 WHERE USER_ID = %S" in text:
            uid = params[0]
            matched = 0
            for r in self.store.get("password_resets", []):
                if r["user_id"] == uid:
                    if "AND IS_USED = 0" in text:
                        if r.get("is_used", 0) == 0:
                            r["is_used"] = 1
                            matched += 1
                    else:
                        r["is_used"] = 1
                        matched += 1
            self.rowcount = matched
        elif "UPDATE PASSWORD_RESETS SET IS_USED = 1 WHERE ID = %S AND IS_USED = 0" in text:
            rid = params[0]
            matched = 0
            for r in self.store.get("password_resets", []):
                if r["id"] == rid and r.get("is_used", 0) == 0:
                    r["is_used"] = 1
                    matched += 1
            self.rowcount = matched
        elif "UPDATE PASSWORD_RESETS SET IS_USED = 1 WHERE ID = %S" in text:
            rid = params[0]
            matched = 0
            for r in self.store.get("password_resets", []):
                if r["id"] == rid:
                    r["is_used"] = 1
                    matched += 1
            self.rowcount = matched
        elif "UPDATE PASSWORD_RESETS SET ATTEMPTS = ATTEMPTS + 1 WHERE ID = %S AND IS_USED = 0" in text:
            rid = params[0]
            matched = 0
            for r in self.store.get("password_resets", []):
                if r["id"] == rid and r.get("is_used", 0) == 0:
                    r["attempts"] = r.get("attempts", 0) + 1
                    matched += 1
            self.rowcount = matched
        elif "SELECT ATTEMPTS FROM PASSWORD_RESETS WHERE ID = %S" in text:
            rid = params[0]
            matching = [r for r in self.store.get("password_resets", []) if r["id"] == rid]
            self._row = {"attempts": matching[0]["attempts"]} if matching else None
        elif "UPDATE PASSWORD_RESETS SET ATTEMPTS = %S, IS_USED = 1 WHERE ID = %S" in text:
            att, rid = params
            for r in self.store.get("password_resets", []):
                if r["id"] == rid:
                    r["attempts"] = att
                    r["is_used"] = 1
            self.rowcount = 1
        elif "UPDATE PASSWORD_RESETS SET ATTEMPTS = %S WHERE ID = %S" in text:
            att, rid = params
            for r in self.store.get("password_resets", []):
                if r["id"] == rid:
                    r["attempts"] = att
            self.rowcount = 1
        elif "UPDATE USERS SET PASSWORD_HASH = %S WHERE ID = %S" in text:
            new_hash, uid = params
            if "users" in self.store and uid in self.store["users"]:
                self.store["users"][uid]["password_hash"] = new_hash
            self.store.setdefault("updated_passwords", []).append(params)
            self.rowcount = 1
        elif "FROM USERS WHERE ID = %S" in text:
            uid = params[0]
            try:
                uid = int(uid)
            except (ValueError, TypeError):
                pass
            self._row = self.store.get("users", {}).get(uid)
        elif "FROM USERS WHERE EMAIL = %S" in text:
            email = params[0]
            matching = [u for u in self.store.get("users", {}).values() if u["email"] == email]
            self._row = matching[0] if matching else None
        elif "FROM USERS WHERE USERNAME = %S OR EMAIL = %S" in text:
            ident1, ident2 = params
            matching = [
                u for u in self.store.get("users", {}).values()
                if u["username"] == ident1 or u["email"] == ident2
            ]
            self._row = matching[0] if matching else None
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
    from werkzeug.security import generate_password_hash
    return {
        "users": {
            1: {"id": 1, "username": "ambuj", "email": "ambuj@example.com", "password_hash": generate_password_hash("password123")},
            2: {"id": 2, "username": "other", "email": "other@example.com", "password_hash": generate_password_hash("password456")},
        },
        "password_resets": [],
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
        "inserted": [], "updated": [], "deleted": [], "updated_passwords": [],
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Clear module/session-level caches so tests cannot leak into each other."""
    app_module._ml_cache.clear()
    app_module._email_sender_hook = None
    yield
    app_module._ml_cache.clear()
    app_module._email_sender_hook = None


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
def _fake_user_loader(monkeypatch, db_store):
    """The user_loader would otherwise hit the DB; stub it instead."""
    def _loader(uid):
        try:
            uid_int = int(uid)
        except (ValueError, TypeError):
            return None
        u = db_store.get("users", {}).get(uid_int)
        if u:
            return User(u["id"], u["username"], u["email"], u.get("password_hash"))
        return User(uid_int, "tester", "t@example.com")
    monkeypatch.setattr(app_module, "get_user_by_id", _loader)


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