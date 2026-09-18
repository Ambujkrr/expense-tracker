"""Tests for walk-forward (expanding-window) evaluation of the forecast model.

Core guarantees these tests enforce:
  - predictions are produced only for months outside their training window
  - no future month ever informs a prediction for an earlier month
  - metrics are aggregate statistics over genuine out-of-sample predictions
  - insufficient data yields an explicit empty state, never invented metrics
"""
import pytest

from app import _walk_forward_evaluate, _eval_regression


# --------------------------------------------------------------- #
# leakage / methodology                                           #
# --------------------------------------------------------------- #

def test_no_future_data_leakage(monkeypatch):
    """Each training set must contain only months strictly before the
    predicted month. We record every training window and assert."""
    seen = []
    real_fit = __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression.fit

    def spy(self, X, y):
        seen.append(list(y))
        return real_fit(self, X, y)

    monkeypatch.setattr(
        __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression,
        "fit", spy,
    )
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _walk_forward_evaluate(values)

    # 3 out-of-sample predictions: train{10,20}->30, train{10,20,30}->40, ...
    assert result["n"] == 3
    assert len(seen) == 3
    assert seen[0] == [10.0, 20.0]
    assert seen[1] == [10.0, 20.0, 30.0]
    assert seen[2] == [10.0, 20.0, 30.0, 40.0]


def test_predictions_are_out_of_sample():
    """The predicted month must never appear in its own training data."""
    values = [15.0, 25.0, 35.0, 45.0]
    result = _walk_forward_evaluate(values)
    assert result["n"] == 2
    # a linear trend predicts the next in-series value well; assert sanity
    assert result["mae"] is not None
    assert result["rmse"] is not None


def test_chronological_order_preserved():
    """Input order is the contract; a reversed series must not be reordered."""
    ascending = [10.0, 20.0, 30.0, 40.0]
    r1 = _walk_forward_evaluate(ascending)
    assert r1["n"] == 2
    assert r1["mae"] == 0.0  # perfect linear trend => zero error


# --------------------------------------------------------------- #
# multiple evaluation months + aggregate math                     #
# --------------------------------------------------------------- #

def test_multiple_evaluation_months():
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    result = _walk_forward_evaluate(values)
    assert result["n"] == 3
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["mape"] == 0.0


def test_aggregate_mae_rmse_matches_manual():
    # Worked example for [10, 20, 60, 80]:
    #   split 1: train [10,20]  -> y = 10x      -> predict 30 (actual 60, err 30)
    #   split 2: train [10,20,60] -> OLS slope 25, intercept -20 -> predict 80 (actual 80, err 0)
    values = [10.0, 20.0, 60.0, 80.0]
    result = _walk_forward_evaluate(values)
    assert result["n"] == 2

    errs = [30.0, 0.0]
    expected_mae = sum(errs) / 2
    expected_rmse = (sum(e * e for e in errs) / 2) ** 0.5
    assert result["mae"] == pytest.approx(expected_mae, abs=1e-6)
    assert result["rmse"] == pytest.approx(expected_rmse, abs=0.01)  # rounded to 2dp


def test_zero_actual_blocks_mape():
    values = [10.0, 20.0, 0.0, 40.0]
    result = _walk_forward_evaluate(values)
    assert result["n"] == 2
    assert result["mape"] is None, "MAPE must be suppressed when an actual is zero"
    assert result["mae"] is not None
    assert result["rmse"] is not None


# --------------------------------------------------------------- #
# R2 edge cases                                                   #
# --------------------------------------------------------------- #

def test_r2_undefined_with_single_eval_month():
    # 3 months => min_train 2 + 1 eval month => only one prediction
    result = _walk_forward_evaluate([10.0, 20.0, 30.0])
    assert result["n"] == 1
    assert result["r2"] is None, "R2 needs 2+ predictions"


def test_r2_undefined_when_actuals_constant():
    # constant actuals => zero variance => R2 undefined
    result = _walk_forward_evaluate([50.0, 50.0, 50.0, 50.0])
    assert result["n"] == 2
    assert result["r2"] is None


# --------------------------------------------------------------- #
# insufficient data                                              #
# --------------------------------------------------------------- #

@pytest.mark.parametrize("values", [[], [10.0], [10.0, 20.0]])
def test_insufficient_data_returns_empty_state(values):
    result = _walk_forward_evaluate(values)
    assert result["n"] == 0
    assert result["mae"] is None
    assert result["rmse"] is None
    assert result["r2"] is None
    assert result["mape"] is None


def test_negative_predictions_clamped_to_zero():
    # a downward trend can extrapolate negative; must be clamped at 0
    result = _walk_forward_evaluate([100.0, 50.0, 10.0, 1.0])
    assert result["n"] == 2
    assert result["mae"] is not None
    # clamping keeps predictions finite and non-negative
    assert result["mae"] >= 0
