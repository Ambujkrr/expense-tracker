"""Validation tests for add_expense and update_expense."""
import pytest


VALID = {"amount": "125.5", "category_id": "1", "expense_date": "2026-09-18", "note": "lunch"}


def _post(client, url, data):
    return client.post(url, data=data)


# ---- amount validation ------------------------------------------------

@pytest.mark.parametrize("bad_amount", ["-50", "0", "abc", "", "nan", "inf", "-inf", "NaN", "Infinity"])
def test_add_invalid_amount_rejected(auth_client, db, db_store, bad_amount):
    before = len(db_store["inserted"])
    resp = _post(auth_client, "/add-expense", {**VALID, "amount": bad_amount})
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == before


@pytest.mark.parametrize("bad_amount", ["-50", "0", "abc", "", "nan", "inf"])
def test_update_invalid_amount_rejected(auth_client, db, db_store, bad_amount):
    before = len(db_store["updated"])
    resp = _post(auth_client, "/update-expense/7", {**VALID, "amount": bad_amount})
    assert resp.status_code == 302
    assert len(db_store["updated"]) == before


def test_valid_amount_rounded_to_two_decimals(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "amount": "125.555"})
    assert resp.status_code == 302
    row = db_store["inserted"][-1]
    # columns: user_id, category_id, amount, note, expense_date
    assert row[2] == 125.56


def test_large_valid_amount_accepted(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "amount": "999999.99"})
    assert resp.status_code == 302
    assert db_store["inserted"][-1][2] == 999999.99


# ---- date validation --------------------------------------------------

@pytest.mark.parametrize("bad_date", ["18/09/2026", "not-a-date", "2026-13-45", "", "2026-02-31", "20260918"])
def test_add_invalid_date_rejected(auth_client, db, db_store, bad_date):
    before = len(db_store["inserted"])
    resp = _post(auth_client, "/add-expense", {**VALID, "expense_date": bad_date})
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == before


@pytest.mark.parametrize("bad_date", ["18/09/2026", "not-a-date", "2026-13-45", ""])
def test_update_invalid_date_rejected(auth_client, db, db_store, bad_date):
    before = len(db_store["updated"])
    resp = _post(auth_client, "/update-expense/7", {**VALID, "expense_date": bad_date})
    assert resp.status_code == 302
    assert len(db_store["updated"]) == before


def test_future_date_accepted(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "expense_date": "2099-12-31"})
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == 1


def test_valid_date_written_verbatim(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "expense_date": "2026-09-18"})
    assert resp.status_code == 302
    assert db_store["inserted"][-1][4] == "2026-09-18"


# ---- note validation --------------------------------------------------

def test_oversized_note_rejected(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "note": "x" * 256})
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == 0


def test_note_of_exactly_255_accepted(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", {**VALID, "note": "x" * 255})
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == 1


def test_oversized_note_rejected_on_update(auth_client, db, db_store):
    before = len(db_store["updated"])
    resp = _post(auth_client, "/update-expense/7", {**VALID, "note": "y" * 256})
    assert resp.status_code == 302
    assert len(db_store["updated"]) == before


def test_note_of_exactly_255_accepted_on_update(auth_client, db, db_store):
    resp = _post(auth_client, "/update-expense/7", {**VALID, "note": "y" * 255})
    assert resp.status_code == 302
    assert len(db_store["updated"]) == 1


def test_missing_note_treated_as_empty(auth_client, db, db_store):
    data = {**VALID}
    data.pop("note")
    resp = _post(auth_client, "/add-expense", data)
    assert resp.status_code == 302
    assert db_store["inserted"][-1][3] == ""


# ---- missing category -------------------------------------------------

def test_missing_category_id_rejected(auth_client, db, db_store):
    data = {**VALID}
    data.pop("category_id")
    resp = _post(auth_client, "/add-expense", data)
    assert resp.status_code == 404
    assert len(db_store["inserted"]) == 0


# ---- combined happy path ----------------------------------------------

def test_add_then_update_happy_path(auth_client, db, db_store):
    resp = _post(auth_client, "/add-expense", VALID)
    assert resp.status_code == 302
    assert len(db_store["inserted"]) == 1

    resp = _post(auth_client, "/update-expense/7", {**VALID, "amount": "200"})
    assert resp.status_code == 302
    assert len(db_store["updated"]) == 1
    assert db_store["updated"][-1][0] == 200.0