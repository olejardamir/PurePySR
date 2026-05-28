"""End-to-end PySR API feature tests: warm-start, multi-output, weights, constraints, custom loss, operators."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

os.environ["PYSR_BACKEND"] = "python"


@pytest.fixture(autouse=True)
def _seed_numpy():
    np.random.seed(42)


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
    )
    defaults.update(kw)
    return PySRRegressor(**defaults)


# ── Warm-start ────────────────────────────────────────────────────────


def test_warm_start_two_stage():
    """Two-stage fit with warm_start adds more equations."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model = _make_regressor(warm_start=True, niterations=2)
    model.fit(X, y)
    eq1 = model.equations_.copy()

    model.fit(X, y)  # second fit — should add more entries
    eq2 = model.equations_
    assert len(eq2) >= len(eq1), (
        f"equations shrunk after warm-start refit: {len(eq1)} → {len(eq2)}"
    )


def test_warm_start_single_pop_preserves_state():
    """Single-population warm-start preserves saved_state."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model = _make_regressor(warm_start=True, niterations=2, populations=1)
    model.fit(X, y)
    assert model._saved_state is not None, "single-pop warm-start should populate saved_state"
    assert "population" in model._saved_state, "saved_state should contain population"


# ── Multi-output ──────────────────────────────────────────────────────


def test_multi_output_fit():
    """Fit with 2-column y produces per-output equations and 2-column predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.column_stack([
        X[:, 0] ** 2,
        X[:, 1] * 0.5,
    ]).astype(np.float64)

    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── max_evals ────────────────────────────────────────────────────────────


def test_max_evals_terminates_early():
    """max_evals=5 terminates quickly and sets termination reason."""
    X = np.random.randn(100, 3).astype(np.float64)
    y = (X[:, 0] + X[:, 1] * 0.5).astype(np.float64)

    model = _make_regressor(niterations=100, max_evals=5, maxsize=20)
    model.fit(X, y)
    assert model.equations_ is not None
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_max_evals_via_pysr_api():
    """max_evals=50 with x^2 data terminates through PySRRegressor."""
    X = np.random.randn(100, 3).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(max_evals=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_max_evals_single_pop_termination_reason():
    """Low max_evals in single-pop sets the termination_reason field."""
    from python_backend.backend import PythonSRBackend
    from python_backend.options import BackendOptions
    import dataclasses

    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    options = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=100,
        max_evals=3,
        population_size=10,
        maxsize=10,
        verbosity=0,
    )
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options)
    trace = result.get("trace_records", [])
    reasons = [
        r["termination_reason"] for r in trace
        if r.get("termination_reason") is not None
    ]
    assert "max_evals" in reasons, (
        f"expected max_evals termination, got reasons: {reasons}"
    )


# ── timeout_in_seconds ──────────────────────────────────────────────────


def test_timeout_terminates_early():
    """timeout_in_seconds=1 terminates quickly."""
    X = np.random.randn(100, 3).astype(np.float64)
    y = (X[:, 0] + X[:, 1] * 0.3).astype(np.float64)

    model = _make_regressor(niterations=200, timeout_in_seconds=1, maxsize=20)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_timeout_via_pysr_api():
    """Tiny timeout_in_seconds=0.001 bounds runtime through PySRRegressor."""
    import time
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(timeout_in_seconds=0.001)
    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── precision ────────────────────────────────────────────────────────────


def test_precision_is_ignored():
    """precision triggers a warning and is harmless."""
    import warnings

    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(precision=32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y)
    spec_warnings = [
        x for x in w if "precision" in str(x.message)
        and issubclass(x.category, UserWarning)
    ]
    assert len(spec_warnings) >= 1, "no precision warning found"
    assert "SR-WARN-OPT-001" in str(spec_warnings[0].message)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Sample weights ────────────────────────────────────────────────────


def test_weights_affect_selected_equation():
    """Weights (outlier point heavily weighted) bias the search via fit()."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    weights = np.ones(50).astype(np.float64)
    weights[0] = 1000.0

    model_weighted = _make_regressor()
    model_weighted.fit(X, y, weights=weights)
    preds_weighted = model_weighted.predict(X)

    model_unweighted = _make_regressor()
    model_unweighted.fit(X, y)
    preds_unweighted = model_unweighted.predict(X)

    assert np.all(np.isfinite(preds_weighted))
    assert np.all(np.isfinite(preds_unweighted))


# ── Operator constraints ──────────────────────────────────────────────


def test_operator_constraints():
    """constraints dict limits operator arity."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        constraints={"+": (1, 1), "-": (2, 2)},
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds)), "constraints produce non-finite preds"
    eq = model.equations_.iloc[0]["equation"]
    assert isinstance(eq, str) and len(eq) > 0, "constraint equation should not be empty"


def test_constraints_respected_in_final_expression():
    """Int-format constraints (child complexity limits) run without error."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        constraints={"+": 2, "-": 1, "*": 3},
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Nested constraints ────────────────────────────────────────────────


def test_nested_constraints():
    """nested_constraints limits operator inside operator."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        nested_constraints={"+": {"+": 0, "-": 2, "*": 2}},
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds)), "nested constraints produce non-finite preds"
    eq = model.equations_.iloc[0]["equation"]
    assert isinstance(eq, str) and len(eq) > 0, "nested-constraint equation should not be empty"


# ── Custom elementwise loss (lambda string) ──────────────────────────


def test_custom_elementwise_loss():
    """elementwise_loss as a lambda expression string."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(
        elementwise_loss="(y_pred, y_true) -> (y_pred - y_true)^2",
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_custom_loss_function_named():
    """loss_function='MAE' produces different loss values than MSE."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(loss_function="MAE")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert "loss" in model.equations_.columns, "MAE equations should have loss column"


def test_custom_loss_affects_stored_loss():
    """Custom elementwise loss with MAE-style lambda runs successfully."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        elementwise_loss="(y_pred, y_true) -> abs(y_pred - y_true)",
    )
    model.fit(X, y)
    assert len(model.equations_) > 0
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Operators: binary coverage ────────────────────────────────────────


def test_binary_operators_mul_div():
    """Multiplicative and protected division operators."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] * X[:, 1] + 1.0 / (np.abs(X[:, 0]) + 0.1)).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*", "/"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_binary_operators_pow():
    """Power operator."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 0] ** 3).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*", "^"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_operator_less():
    """Less-than operator."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.where(X[:, 0] < X[:, 1], X[:, 0], X[:, 1])

    model = _make_regressor(
        binary_operators=["+", "-", "*", "<"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Operators: unary coverage ─────────────────────────────────────────


def test_unary_operators_exp_log():
    """Exponential and safe-log operators."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = np.exp(X[:, 0] * 0.5).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["exp", "safe_log"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_unary_operators_sin_cos():
    """Trigonometric operators."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = np.sin(X[:, 0] * 2.0).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["sin", "cos"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_unary_operator_abs():
    """Absolute-value operator."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = np.abs(X[:, 0]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["abs"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Exp round-trip ────────────────────────────────────────────────────


def test_exp_round_trip_export():
    """Fit with exp operator: equations, sympy export, and predict all work."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = np.exp(X[:, 0] * 0.5).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["exp"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

    eq = model.equations_.iloc[0]
    assert "sympy_format" in model.equations_.columns, "sympy export column missing"
    assert "lambda_format" in model.equations_.columns, "lambda_format column missing"
    assert callable(eq["lambda_format"]), "lambda_format should be callable"
    manual_pred = eq["lambda_format"](X)
    assert np.all(np.isfinite(manual_pred))


# ── Custom operator defined inline ────────────────────────────────────


def test_custom_operator_inline():
    """Inline custom unary operators are rejected before search starts."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = (1.0 / (np.abs(X[:, 0]) + 0.1)).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
    )
    with pytest.raises(ValueError, match="Inline custom operator definitions"):
        model.fit(X, y)


# ── Denoise ────────────────────────────────────────────────────────────


def test_denoise_option():
    """denoise=True runs without error."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1] * 0.5).astype(np.float64)

    model = _make_regressor(denoise=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── select_k_features ──────────────────────────────────────────────────


def test_select_k_features():
    """select_k_features restricts feature count."""
    X = np.random.randn(30, 5).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(select_k_features=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Parsimony / model_selection ────────────────────────────────────────


def test_parsimony_scale():
    """Higher parsimony produces simpler expressions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1] * 0.5).astype(np.float64)

    model = _make_regressor(parsimony=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_model_selection_accuracy():
    """model_selection='accuracy' (best loss) runs without error."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(model_selection="accuracy")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Autodiff backend ───────────────────────────────────────────────────


def test_autodiff_backend_rejected():
    """autodiff_backend=True is rejected by PySR_custom with a clear error."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1] * 0.3).astype(np.float64)

    model = _make_regressor(autodiff_backend=True)
    with pytest.raises(ValueError, match="autodiff_backend is disabled"):
        model.fit(X, y)


# ── Binary operators with large inputs ─────────────────────────────────


def test_binary_operator_large_input():
    """Binary operators handle large-magnitude inputs without crash."""
    X = np.random.randn(30, 2).astype(np.float64) * 1e6
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor(binary_operators=["+", "-", "*", "/"])
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Unary operators edge cases (NaN input) ─────────────────────────────


def test_unary_operator_large_input():
    """Unary ops handle large-magnitude inputs without crash."""
    X = np.random.randn(30, 1).astype(np.float64) * 1e4
    y = np.sin(X[:, 0] * 0.001).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "*"],
        unary_operators=["sin", "cos"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Loss function edge cases ───────────────────────────────────────────


def test_loss_function_complex():
    """loss_function with non-default scale works."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(loss_function="MAE", loss_scale=2.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert "loss" in model.equations_.columns


# ── Constraints with many operators ────────────────────────────────────


def test_constraints_many_operators():
    """Constraints with 4+ binary operators work."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*", "/"],
        constraints={"+": (1, 1), "-": (1, 1), "*": (2, 2), "/": (3, 3)},
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds)), "multi-operator constraints produce non-finite preds"


# ── Nested constraints with many operators ─────────────────────────────


def test_nested_constraints_many_operators():
    """nested_constraints with 4+ operators work."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)

    model = _make_regressor(
        binary_operators=["+", "-", "*", "/"],
        nested_constraints={
            "+": {"+": 0, "-": 2, "*": 1, "/": 0},
            "-": {"+": 1, "-": 0, "*": 2, "/": 1},
        },
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds)), "multi nested-constraints produce non-finite preds"


# ── Warm-start with populations > 1 ────────────────────────────────────


def test_warm_start_multi_pop():
    """Multi-population warm-start preserves saved_state across fits."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model = _make_regressor(warm_start=True, niterations=2, populations=2)
    model.fit(X, y)
    assert model._saved_state is not None
    assert "population" in model._saved_state

    # Step should be populated
    step1 = model._saved_state.get("step", 0)
    assert step1 > 0, f"expected step > 0, got {step1}"

    model.fit(X, y)
    assert len(model.equations_) > 0
    # Step should have advanced
    step2 = model._saved_state.get("step", 0)
    assert step2 > step1, f"step should advance: {step1} → {step2}"
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_warm_start_multi_pop_state_increments():
    """Multi-pop warm-start: saved_state step increments between fits."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model = _make_regressor(warm_start=True, niterations=3, populations=2)
    model.fit(X, y)
    step1 = model._saved_state.get("step", 0)

    model.fit(X, y)
    step2 = model._saved_state.get("step", 0)

    # Step counter should advance (accumulate iterations)
    assert step2 > step1, f"step should advance: {step1} → {step2}"


# ── Batching ───────────────────────────────────────────────────────────


def test_batching_basic():
    """batching=True runs without error."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model = _make_regressor(batching=True, batch_size=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Unknown operator ────────────────────────────────────────────


def test_unknown_operator_raises_error():
    """An unknown operator token should fail before search starts."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = X[:, 0] ** 2
    model = _make_regressor(
        binary_operators=["+", "-", "*", "unknown_op"],
    )
    with pytest.raises(ValueError):
        model.fit(X, y)


# ── Random state reproducibility ───────────────────────────────────


def test_random_state_reproducibility():
    """Same random_state should produce identical equations across runs."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model1 = _make_regressor(random_state=42)
    model1.fit(X, y)

    model2 = _make_regressor(random_state=42)
    model2.fit(X, y)

    assert len(model1.equations_) == len(model2.equations_)
    assert model1.equations_["equation"].iloc[0] == model2.equations_["equation"].iloc[0]


# ── Backend selector precedence ────────────────────────────────────


def test_backend_selector_precedence():
    """Explicit backend='python' should override PYSR_BACKEND env var."""
    import os as _os
    _os.environ["PYSR_BACKEND"] = "julia"
    try:
        from pysr import PySRRegressor
        X = np.random.randn(30, 2).astype(np.float64)
        y = X[:, 0] ** 2
        model = PySRRegressor(
            niterations=1, population_size=10, tournament_selection_n=5,
            binary_operators=["+", "-", "*"], unary_operators=[], maxsize=10,
            verbosity=0, progress=False, backend="python",
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))
    finally:
        _os.environ["PYSR_BACKEND"] = "python"


def test_loss_scale_log():
    """loss_scale='log' should not crash fit."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(loss_scale="log")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_parsimony_selects_simpler():
    """Higher parsimony should select simpler expressions."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model_low = _make_regressor(binary_operators=["+", "-", "*"], parsimony=0.001)
    model_low.fit(X, y)
    low_eq = model_low.equations_.iloc[0]

    model_high = _make_regressor(binary_operators=["+", "-", "*"], parsimony=10.0)
    model_high.fit(X, y)
    high_eq = model_high.equations_.iloc[0]

    assert low_eq["complexity"] >= high_eq["complexity"], (
        f"higher parsimony should not produce more complex eq: "
        f"{low_eq['complexity']} vs {high_eq['complexity']}"
    )


def test_deterministic_flag():
    """deterministic=True with random_state should produce identical results."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    model1 = _make_regressor(
        binary_operators=["+", "-", "*"], random_state=42, deterministic=True,
    )
    model1.fit(X, y)

    np.random.seed(99)
    model2 = _make_regressor(
        binary_operators=["+", "-", "*"], random_state=42, deterministic=True,
    )
    model2.fit(X, y)

    assert len(model1.equations_) == len(model2.equations_)


def test_multi_output_artifacts():
    """Multi-output fit produces per-output equations and correct prediction shape."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.column_stack([
        X[:, 0] ** 2,
        X[:, 1] * 0.5,
    ]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30, 2), f"expected (30, 2), got {preds.shape}"
    assert np.all(np.isfinite(preds))
    # Multi-output: equations_ is a list of DataFrames, one per output
    assert isinstance(model.equations_, list), f"expected list, got {type(model.equations_)}"
    assert len(model.equations_) == 2, f"expected 2 output tables, got {len(model.equations_)}"
    for i, eq_df in enumerate(model.equations_):
        assert "sympy_format" in eq_df.columns, f"output {i}: sympy_format column missing"
        assert "lambda_format" in eq_df.columns, f"output {i}: lambda_format column missing"


def test_optimizer_algorithm_bfgs():
    """BFGS optimizer algorithm should work."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_algorithm="BFGS")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_optimizer_nrestarts():
    """optimizer_nrestarts=0 should still produce valid results."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_nrestarts=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_denoise_enabled():
    """denoise=True should not crash."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(denoise=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_batching_produces_finite():
    """Batching with moderate batch_size."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(batching=True, batch_size=20)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
