"""Tests for the ML evaluation metrics: regression metrics and anomaly rate.

These exercise the pure calculation helpers directly, plus the dashboard
rendering path that surfaces them. They deliberately assert that metrics
are *suppressed* (not fabricated) when the data cannot support them.
"""
import pytest

from app import _eval_regression, _anomaly_rate


# ------------------------------------------------------------------ #
# regression metrics                                                 #
# ------------------------------------------------------------------ #

def test_regression_normal_data():
    result = _eval_regression([100.0, 200.0, 300.0], [110.0, 190.0, 320.0])
    assert result["n"] == 3
    assert result["mae"] == round((10 + 10 + 20) / 3, 2)
    assert result["rmse"] == round(((100 + 100 + 400) / 3) ** 0.5, 2)
    assert result["r2"] is not None
    assert result["mape"] is not None


def test_regression_perfect_prediction():
    result = _eval_regression([50.0, 150.0], [50.0, 150.0])
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["mape"] == 0.0


def test_regression_zero_actual_blocks_mape():
    # MAPE would divide by zero; it must be None, other metrics still valid.
    result = _eval_regression([0.0, 200.0], [10.0, 190.0])
    assert result["mape"] is None
    assert result["mae"] == round((10 + 10) / 2, 2)
    assert result["rmse"] == round(((100 + 100) / 2) ** 0.5, 2)
    assert result["r2"] is not None


def test_regression_single_point_r2_undefined():
    # The dashboard hold-out is a single month: R2 is undefined there.
    result = _eval_regression([500.0], [450.0])
    assert result["n"] == 1
    assert result["mae"] == 50.0
    assert result["rmse"] == 50.0
    assert result["r2"] is None, "R2 must not be fabricated for one test point"


def test_regression_constant_actual_r2_undefined():
    # Zero variance in the actuals also makes R2 undefined.
    result = _eval_regression([100.0, 100.0], [80.0, 120.0])
    assert result["r2"] is None


def test_regression_insufficient_data():
    result = _eval_regression([], [])
    assert result["n"] == 0
    assert result["mae"] is None
    assert result["rmse"] is None
    assert result["r2"] is None
    assert result["mape"] is None


# ------------------------------------------------------------------ #
# anomaly rate                                                       #
# ------------------------------------------------------------------ #

def test_anomaly_rate_basic():
    assert _anomaly_rate(10, 1) == 10.0


def test_anomaly_rate_none_flagged():
    assert _anomaly_rate(20, 0) == 0.0


def test_anomaly_rate_no_transactions():
    assert _anomaly_rate(0, 0) is None


def test_anomaly_rate_handles_none():
    assert _anomaly_rate(None, None) is None


# ------------------------------------------------------------------ #
# dashboard integration                                               #
# ------------------------------------------------------------------ #

def test_dashboard_renders_regression_metrics(auth_client, db, db_store):
    db_store["monthly_rows"] = [
        {"month": "2026-06", "total": 100.0},
        {"month": "2026-07", "total": 120.0},
        {"month": "2026-08", "total": 90.0},
        {"month": "2026-09", "total": 110.0},
    ]
    resp = auth_client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Mean Absolute Error (MAE)" in body
    assert "RMSE" in body


def test_dashboard_renders_anomaly_stats(auth_client, db, db_store):
    db_store["expense_list"] = [
        {"id": i, "amount": float(i) * 10, "note": "n", "expense_date": "2026-09-01",
         "category_id": 1, "category": "Food"}
        for i in range(1, 9)
    ]
    resp = auth_client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "transactions evaluated" in body
    assert "anomaly rate" in body
    assert "accuracy cannot be calculated" in body


def test_dashboard_hides_forecast_with_insufficient_data(auth_client, db, db_store):
    # Only one month of history: no forecast is produced at all.
    resp = auth_client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "at least 2 different months" in body
    assert "Next month forecast" not in body


def test_dashboard_shows_mae_unavailable_with_two_months(auth_client, db, db_store):
    # Two months: a forecast exists, but walk-forward needs three.
    db_store["monthly_rows"] = [
        {"month": "2026-08", "total": 120.0},
        {"month": "2026-09", "total": 90.0},
    ]
    resp = auth_client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "MAE unavailable" in body
    assert "Requirement" in body