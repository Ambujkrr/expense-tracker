"""Tests that anomaly detection and ML caching respect the dashboard filters.

Guarantees enforced here:
  - the Isolation Forest only ever sees the filtered expense set, so a
    flagged row can never belong to another category or month
  - ML cache entries are keyed by (user_id, category_id, month), so a
    result computed for one filter combination is never served for another
  - an expense mutation drops every filter variant for that user
  - the dashboard reports the walk-forward evaluation method
"""
from app import _invalidate_ml_cache, _ml_cache, _ml_cache_key


FOOD, TRANSPORT, SHOPPING, ENTERTAINMENT = 1, 2, 3, 4


def _mixed_expenses():
    """Seventeen rows spanning 4 categories and 2 months.

    Several deliberately large outliers (4200 Food, 9000 Shopping, 3500
    Transport, 8000 Entertainment) guarantee the Isolation Forest always
    has something to flag, while each filter still leaves 5+ rows in scope.
    """
    return [
        {"id": 1, "amount": 50.0, "note": "lunch", "expense_date": "2026-08-03", "category_id": FOOD, "category": "Food"},
        {"id": 2, "amount": 60.0, "note": "groceries", "expense_date": "2026-08-05", "category_id": FOOD, "category": "Food"},
        {"id": 3, "amount": 40.0, "note": "snack", "expense_date": "2026-08-07", "category_id": FOOD, "category": "Food"},
        {"id": 4, "amount": 4200.0, "note": "wedding catering", "expense_date": "2026-08-09", "category_id": FOOD, "category": "Food"},
        {"id": 5, "amount": 9000.0, "note": "laptop", "expense_date": "2026-08-13", "category_id": SHOPPING, "category": "Shopping"},
        {"id": 6, "amount": 45.0, "note": "dinner", "expense_date": "2026-09-10", "category_id": FOOD, "category": "Food"},
        {"id": 7, "amount": 30.0, "note": "bus fare", "expense_date": "2026-09-02", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 8, "amount": 35.0, "note": "metro", "expense_date": "2026-09-04", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 9, "amount": 25.0, "note": "parking", "expense_date": "2026-09-08", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 10, "amount": 28.0, "note": "fuel", "expense_date": "2026-09-12", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 11, "amount": 22.0, "note": "taxi", "expense_date": "2026-09-14", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 12, "amount": 3500.0, "note": "car repair", "expense_date": "2026-09-16", "category_id": TRANSPORT, "category": "Transport"},
        {"id": 13, "amount": 300.0, "note": "movie", "expense_date": "2026-09-01", "category_id": ENTERTAINMENT, "category": "Entertainment"},
        {"id": 14, "amount": 250.0, "note": "streaming", "expense_date": "2026-09-03", "category_id": ENTERTAINMENT, "category": "Entertainment"},
        {"id": 15, "amount": 8000.0, "note": "concert tickets", "expense_date": "2026-09-06", "category_id": ENTERTAINMENT, "category": "Entertainment"},
        {"id": 16, "amount": 220.0, "note": "game", "expense_date": "2026-09-09", "category_id": ENTERTAINMENT, "category": "Entertainment"},
        {"id": 17, "amount": 280.0, "note": "theater", "expense_date": "2026-09-11", "category_id": ENTERTAINMENT, "category": "Entertainment"},
    ]


def _cached_anomalies(category_id=None, month=None, user_id=1):
    """Read the anomalies cached for one filter combination."""
    entry = _ml_cache.get(_ml_cache_key(user_id, category_id, month), {})
    return entry.get("anomalies", [])


# ------------------------------------------------------------------ #
# filter containment                                                 #
# ------------------------------------------------------------------ #

def test_unfiltered_anomaly_results(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()
    resp = auth_client.get("/")
    assert resp.status_code == 200
    anomalies = _cached_anomalies()
    # No filter: every category is in scope and the outliers are extreme
    # enough that at least one row is flagged.
    assert len(anomalies) >= 1
    assert {a["category"] for a in anomalies} <= {"Food", "Shopping", "Transport", "Entertainment"}
    assert {a["amount"] for a in anomalies} & {4200.0, 3500.0, 8000.0, 9000.0}


def test_category_filtered_anomalies_stay_within_category(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()
    resp = auth_client.get("/?category_id=%d" % FOOD)
    assert resp.status_code == 200
    anomalies = _cached_anomalies(category_id=str(FOOD))
    # A Food-only view can never surface Shopping/Entertainment rows, even
    # though those outliers are far larger than anything in Food.
    assert {a["category"] for a in anomalies} <= {"Food"}
    amounts = {a["amount"] for a in anomalies}
    assert 9000.0 not in amounts
    assert 8000.0 not in amounts


def test_month_filtered_anomalies_stay_within_month(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()
    resp = auth_client.get("/?month=2026-09")
    assert resp.status_code == 200
    anomalies = _cached_anomalies(month="2026-09")
    for a in anomalies:
        assert str(a["expense_date"]).startswith("2026-09")
    # The August Shopping outlier must never leak into a September view.
    assert 9000.0 not in {a["amount"] for a in anomalies}


def test_category_and_month_filters_applied_together(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()
    resp = auth_client.get("/?category_id=%d&month=2026-09" % TRANSPORT)
    assert resp.status_code == 200
    anomalies = _cached_anomalies(category_id=str(TRANSPORT), month="2026-09")
    # 6 Transport rows in September -> detection runs, so a result here is
    # a real filtered result rather than an empty skip state.
    assert len(anomalies) >= 1
    for a in anomalies:
        assert a["category"] == "Transport"
        assert str(a["expense_date"]).startswith("2026-09")
    amounts = {a["amount"] for a in anomalies}
    # Both the September Entertainment outlier and the August Food outlier
    # sit outside this filter combination.
    assert 8000.0 not in amounts
    assert 4200.0 not in amounts


# ------------------------------------------------------------------ #
# cache isolation + invalidation                                     #
# ------------------------------------------------------------------ #

def test_cache_isolation_between_different_filters(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()

    # 1. Unfiltered request populates the (1, None, None) entry.
    first = auth_client.get("/")
    assert first.status_code == 200
    assert _ml_cache_key(1, None, None) in _ml_cache
    unfiltered = _cached_anomalies()
    assert len(unfiltered) >= 1

    # 2. The Entertainment view must not reuse that unfiltered entry.
    second = auth_client.get("/?category_id=%d" % ENTERTAINMENT)
    assert second.status_code == 200
    filtered = _cached_anomalies(category_id=str(ENTERTAINMENT))
    assert len(filtered) >= 1
    assert all(a["category"] == "Entertainment" for a in filtered)
    # Whatever the unfiltered view flagged, none of its non-Entertainment
    # rows may survive into the filtered result.
    leaked = {a["id"] for a in unfiltered if a["category"] != "Entertainment"}
    assert not ({a["id"] for a in filtered} & leaked)

    # 3. Both entries coexist under distinct keys.
    assert _ml_cache_key(1, None, None) in _ml_cache
    assert _ml_cache_key(1, str(ENTERTAINMENT), None) in _ml_cache


def test_invalidate_ml_cache_drops_all_filter_variants():
    _ml_cache.clear()
    try:
        _ml_cache[_ml_cache_key(1, None, None)] = {"prediction": 1}
        _ml_cache[_ml_cache_key(1, "1", None)] = {"prediction": 2}
        _ml_cache[_ml_cache_key(1, None, "2026-09")] = {"prediction": 3}
        _ml_cache[_ml_cache_key(2, None, None)] = {"prediction": 4}

        _invalidate_ml_cache(1)

        # Every variant for user 1 is gone; user 2 is untouched.
        assert _ml_cache == {_ml_cache_key(2, None, None): {"prediction": 4}}
    finally:
        _ml_cache.clear()


def test_cache_invalidation_after_expense_mutation(auth_client, db, db_store):
    db_store["expense_list"] = _mixed_expenses()

    # Prime several filter variants for the logged-in user.
    auth_client.get("/")
    auth_client.get("/?category_id=%d" % FOOD)
    auth_client.get("/?month=2026-09")
    for key in (
        _ml_cache_key(1, None, None),
        _ml_cache_key(1, str(FOOD), None),
        _ml_cache_key(1, None, "2026-09"),
    ):
        assert key in _ml_cache

    # Adding an expense must drop every variant for this user.
    resp = auth_client.post("/add-expense", data={
        "amount": "75.0",
        "category_id": "1",
        "expense_date": "2026-09-20",
        "note": "dinner",
    })
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == 1
    for key in (
        _ml_cache_key(1, None, None),
        _ml_cache_key(1, str(FOOD), None),
        _ml_cache_key(1, None, "2026-09"),
    ):
        assert key not in _ml_cache


# ------------------------------------------------------------------ #
# evaluation method                                                  #
# ------------------------------------------------------------------ #

def test_dashboard_reports_walk_forward_not_hold_out(auth_client, db, db_store):
    # Four months of history -> walk-forward produces 2 out-of-sample
    # predictions, so MAE is defined and the method label is rendered.
    db_store["monthly_rows"] = [
        {"month": "2026-06", "total": 100.0},
        {"month": "2026-07", "total": 120.0},
        {"month": "2026-08", "total": 90.0},
        {"month": "2026-09", "total": 110.0},
    ]
    resp = auth_client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Walk-forward" in body
    assert "Hold-out" not in body
