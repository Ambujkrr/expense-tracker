"""Ownership tests: one user must never touch another user's expenses."""


VALID = {"amount": "125.5", "category_id": "1", "expense_date": "2026-09-18", "note": "lunch"}


def _foreign_expense(store):
    """Mark expense 7 as NOT owned by the logged-in user (user 1)."""
    store["owned_expenses"].discard(7)


def test_other_users_expense_cannot_be_updated(auth_client, db, db_store):
    _foreign_expense(db_store)
    resp = auth_client.post("/update-expense/7", data=VALID)
    # the UPDATE is scoped by user_id, so rowcount == 0 => abort(404)
    assert resp.status_code == 404
    assert len(db_store["updated"]) == 0


def test_other_users_expense_cannot_be_deleted(auth_client, db, db_store):
    _foreign_expense(db_store)
    resp = auth_client.post("/delete-expense/7")
    assert resp.status_code == 404
    assert len(db_store["deleted"]) == 0


def test_other_users_expense_cannot_be_edited(auth_client, db, db_store):
    _foreign_expense(db_store)
    # the SELECT in edit_expense is scoped by user_id; a foreign owner means
    # the row is not returned
    db_store["expense_rows"].pop(7, None)
    resp = auth_client.get("/edit-expense/7")
    assert resp.status_code == 404


def test_own_expense_can_be_edited(auth_client, db, db_store):
    resp = auth_client.get("/edit-expense/7")
    assert resp.status_code == 200


def test_own_expense_can_be_deleted(auth_client, db, db_store):
    resp = auth_client.post("/delete-expense/7")
    assert resp.status_code == 302
    assert len(db_store["deleted"]) == 1