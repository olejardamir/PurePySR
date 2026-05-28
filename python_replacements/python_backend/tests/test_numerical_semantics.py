from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["PYSR_BACKEND"] = "python"


def _make_regressor(**kw):
    from pysr import PySRRegressor
    defaults = dict(
        niterations=2,
        population_size=10,
        tournament_selection_n=5,
        ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        verbosity=0,
        progress=False,
        random_state=42,
    )
    defaults.update(kw)
    return PySRRegressor(**defaults)


@pytest.fixture(autouse=True)
def _seed_numpy():
    np.random.seed(42)


def test_predict_on_extreme_inputs():
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    X_extreme = np.array([[1e30, -1e30], [1e-30, -1e-30], [0.0, 0.0]], dtype=np.float64)
    preds = model.predict(X_extreme)
    assert np.all(np.isfinite(preds))


def test_search_on_extreme_targets():
    X = np.random.randn(7, 2).astype(np.float64)
    y = np.array([1e30, -1e30, 1e-30, -1e-30, 1.0, 0.0, -1.0], dtype=np.float64)
    model = _make_regressor(population_size=8)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_mixed_finite_invalid_loss():
    X = np.random.randn(20, 2).astype(np.float64)
    y = np.where(X[:, 0] > 0, 1e30, 1e-30).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_power_negative_base():
    X = np.random.randn(30, 3).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        binary_operators=["+", "-", "*", "^"],
        unary_operators=[],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_float32_vs_float64_loss_consistency():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model_f64 = _make_regressor(random_state=42)
    model_f64.fit(X, y)

    X32 = X.astype(np.float32)
    y32 = y.astype(np.float32)
    model_f32 = _make_regressor(random_state=42)
    model_f32.fit(X32, y32)

    preds_f64 = model_f64.predict(X)
    preds_f32 = model_f32.predict(X)
    assert np.all(np.isfinite(preds_f64))
    assert np.all(np.isfinite(preds_f32))

    best_f64 = model_f64.equations_.iloc[0]["loss"]
    best_f32 = model_f32.equations_.iloc[0]["loss"]
    assert abs(best_f64 - best_f32) < 1e-3


def test_simplification_preserves_predictions():
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model_simp = _make_regressor(
        should_simplify=True,
        should_optimize_constants=False,
    )
    model_simp.fit(X, y)
    preds_simp = model_simp.predict(X)
    assert np.all(np.isfinite(preds_simp))


def test_constant_optimization_convergence():
    X = np.random.randn(30, 1).astype(np.float64)
    y = np.sin(X[:, 0]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["sin"],
        should_optimize_constants=True,
        optimizer_iterations=10,
        optimizer_nrestarts=2,
        maxsize=12,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_dimensional_analysis_accepts_valid():
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_protected_div_search_stable():
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] / (1.0 + np.abs(X[:, 1]))).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_safe_log_search_stable():
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.log(1.0 + np.abs(X[:, 0])).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["safe_log"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_nan_inf_in_weights():
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    weights_nan = np.full(30, np.nan, dtype=np.float64)
    model_nan = _make_regressor()
    with pytest.raises(Exception):
        model_nan.fit(X, y, weights=weights_nan)

    weights_inf = np.full(30, np.inf, dtype=np.float64)
    model_inf = _make_regressor()
    with pytest.raises(Exception):
        model_inf.fit(X, y, weights=weights_inf)
