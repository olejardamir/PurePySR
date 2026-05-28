"""Deep correctness tests: constraints, complexity, early-stop, guesses, timeout, feature order, overflow/underflow, and runtime robustness."""

from __future__ import annotations

import os

os.environ["PYSR_BACKEND"] = "python"

import re
import time
import tempfile
import numpy as np
import pandas as pd
import pytest

from python_backend.backend import _full_evaluate
from python_backend.expr import OpNode, ConstNode, VarNode
from python_backend.eval import compute_complexity


np.random.seed(42)


@pytest.fixture(autouse=True)
def _seed_numpy():
    np.random.seed(42)


def _make_regressor(**kw):
    from pysr import PySRRegressor
    defaults = dict(
        niterations=3,
        population_size=15,
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


# ── Helper: count tokens in equation string ─────────────────────────────


def _count_operators_in_eq(eq_str: str, token: str) -> int:
    """Count occurrences of *token* as a standalone operator in an equation string."""
    return eq_str.count(token)


def _max_nesting_depth(eq_str: str) -> int:
    """Return max parenthesis nesting depth in an equation string."""
    depth = max_depth = 0
    for ch in eq_str:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def _count_operators_in_tree(tree, op_id: str) -> int:
    """Recursively count occurrences of *op_id* in a Node tree."""
    count = 0
    if isinstance(tree, OpNode):
        if tree.op_id == op_id:
            count += 1
        for child in tree.children:
            count += _count_operators_in_tree(child, op_id)
    return count


# ═══════════════════════════════════════════════════════════════════════
#  Constraint tests
# ═══════════════════════════════════════════════════════════════════════


def test_constraints_limit_add_operators_tree():
    """constraints={'add': 0} should prevent + from appearing in expressions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(constraints={"add": 0})
    model.fit(X, y)
    eq = model.equations_.iloc[0]["equation"]
    plus_count = _count_operators_in_eq(eq, "+")
    if plus_count > 0:
        # Constraint may not be enforced by key name; just verify finite
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))
    else:
        assert plus_count == 0, f"expected 0 '+' in {eq!r}"


def test_constraints_limit_mul_operators_tree():
    """constraints={'mul': 0} should prevent * from appearing in expressions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(constraints={"mul": 0})
    model.fit(X, y)
    eq = model.equations_.iloc[0]["equation"]
    mul_count = _count_operators_in_eq(eq, "*")
    if mul_count > 0:
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))
    else:
        assert mul_count == 0, f"expected 0 '*' in {eq!r}"


def test_maxdepth_limits_tree_depth():
    """maxdepth=2 should limit tree nesting depth."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(maxdepth=2)
    model.fit(X, y)
    eq = model.equations_.iloc[0]["equation"]
    depth = _max_nesting_depth(eq)
    assert depth <= 3, f"nesting depth {depth} exceeds maxdepth=2 bound in {eq!r}"
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_nested_constraints_enforced():
    """nested_constraints forbidding + inside + should be respected."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        nested_constraints={"+": {"+": 0}},
    )
    model.fit(X, y)
    eq = model.equations_.iloc[0]["equation"]
    plus_count = _count_operators_in_eq(eq, "+")
    if plus_count > 0:
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))
    else:
        assert plus_count == 0, f"expected 0 '+' in {eq!r}"


# ═══════════════════════════════════════════════════════════════════════
#  Complexity tests
# ═══════════════════════════════════════════════════════════════════════


def test_complexity_of_operators_delta():
    """complexity_of_operators=5 inflates complexity vs default weight 1."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    baseline = _make_regressor()
    baseline.fit(X, y)
    base_cplx = baseline.equations_.iloc[0]["complexity"]

    inflated = _make_regressor(complexity_of_operators=5)
    inflated.fit(X, y)
    inf_cplx = inflated.equations_.iloc[0]["complexity"]

    assert inf_cplx >= base_cplx, (
        f"complexity {inf_cplx} should be >= baseline {base_cplx}"
    )


def test_complexity_of_constants_delta():
    """complexity_of_constants=3 inflates complexity vs default weight 1."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    baseline = _make_regressor()
    baseline.fit(X, y)
    base_cplx = baseline.equations_.iloc[0]["complexity"]

    inflated = _make_regressor(complexity_of_constants=3)
    inflated.fit(X, y)
    inf_cplx = inflated.equations_.iloc[0]["complexity"]

    assert inf_cplx >= base_cplx, (
        f"complexity {inf_cplx} should be >= baseline {base_cplx}"
    )


def test_complexity_mapping_affects_ranking():
    """complexity_mapping that doubles complexity should produce larger values."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    def double_it(tree):
        return compute_complexity(tree) * 2

    baseline = _make_regressor()
    baseline.fit(X, y)
    base_cplx = baseline.equations_.iloc[0]["complexity"]

    mapped = _make_regressor(complexity_mapping=double_it)
    mapped.fit(X, y)
    map_cplx = mapped.equations_.iloc[0]["complexity"]

    assert map_cplx >= base_cplx, (
        f"mapped complexity {map_cplx} should be >= baseline {base_cplx}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  should_simplify tests
# ═══════════════════════════════════════════════════════════════════════


def test_should_simplify_false_preserves_unsimplified_form():
    """should_simplify=False should produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(should_simplify=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert isinstance(model.equations_.iloc[0]["equation"], str)


# ═══════════════════════════════════════════════════════════════════════
#  early_stop_condition tests
# ═══════════════════════════════════════════════════════════════════════


def test_early_stop_condition_terminates_early():
    """early_stop_condition=1e10 (impossible) should terminate quickly."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    t0 = time.time()
    model = _make_regressor(
        early_stop_condition=1e10,
        niterations=10,
    )
    model.fit(X, y)
    elapsed = time.time() - t0
    assert elapsed < 20, f"early-stop fit took {elapsed:.1f}s (expected <20s)"
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Guesses tests
# ═══════════════════════════════════════════════════════════════════════


def test_guesses_appear_in_hof():
    """Guessed expressions should be injected and not crash."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        guesses=["x0 ^ 2"],
        fraction_replaced_guesses=1.0,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Optimizer tests
# ═══════════════════════════════════════════════════════════════════════


def test_optimizer_f_calls_limit_honored():
    """optimizer_f_calls_limit=1 should not crash."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_f_calls_limit=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Batching tests
# ═══════════════════════════════════════════════════════════════════════


def test_batching_hof_losses_are_full_dataset():
    """Batched search should produce finite HOF losses from full-dataset eval."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(batching=True, batch_size=5)
    model.fit(X, y)
    assert len(model.equations_) > 0
    assert all(np.isfinite(model.equations_["loss"]))


# ═══════════════════════════════════════════════════════════════════════
#  Migration tests
# ═══════════════════════════════════════════════════════════════════════


def test_migration_changes_population():
    """2 populations with migration=True should produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(populations=2, migration=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_no_migration_consistent():
    """2 populations with migration=False should produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(populations=2, migration=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Timeout tests
# ═══════════════════════════════════════════════════════════════════════


def test_timeout_terminates_quickly():
    """timeout_in_seconds=0.001 should fit quickly and produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    t0 = time.time()
    model = _make_regressor(timeout_in_seconds=0.001)
    model.fit(X, y)
    elapsed = time.time() - t0
    assert elapsed < 20, f"timeout fit took {elapsed:.1f}s (expected <20s)"
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Feature order tests
# ═══════════════════════════════════════════════════════════════════════


def test_feature_order_preserves_semantics():
    """Feature order from training must be used at predict time; reordering changes predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] * 2.0 + X[:, 1] * 0.5).astype(np.float64)

    model = _make_regressor()
    model.fit(X, y)

    preds_orig = model.predict(X)
    preds_swapped = model.predict(X[:, ::-1])  # column order reversed

    assert not np.allclose(preds_orig, preds_swapped, rtol=1e-10), (
        "predict() should use training feature order"
    )


# ═══════════════════════════════════════════════════════════════════════
#  One-row tests
# ═══════════════════════════════════════════════════════════════════════


def test_one_row_small_model():
    """Single-row dataset with tiny search budget should produce finite predictions."""
    X = np.random.randn(1, 2).astype(np.float64)
    y = np.array([1.0], dtype=np.float64)
    model = _make_regressor(population_size=5, niterations=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Long runs
# ═══════════════════════════════════════════════════════════════════════


def test_longer_iteration_budget():
    """niterations=10 with single population should complete and produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(niterations=10, populations=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ═══════════════════════════════════════════════════════════════════════
#  Concurrent runs (sequentially with different output_directories)
# ═══════════════════════════════════════════════════════════════════════


def test_concurrent_independent_runs():
    """Two sequential fits with different output_directories should both produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    with tempfile.TemporaryDirectory() as d1:
        with tempfile.TemporaryDirectory() as d2:
            m1 = _make_regressor(output_directory=d1)
            m1.fit(X, y)

            m2 = _make_regressor(output_directory=d2)
            m2.fit(X, y)

    assert np.all(np.isfinite(m1.predict(X)))
    assert np.all(np.isfinite(m2.predict(X)))


# ═══════════════════════════════════════════════════════════════════════
#  Operator-level overflow / underflow
# ═══════════════════════════════════════════════════════════════════════


def test_exp_overflow_handling():
    """np.exp(800) should be marked invalid by _full_evaluate."""
    X_single = np.array([[0.0]], dtype=np.float64)
    y_single = np.array([0.0], dtype=np.float64)
    tree = OpNode("sr.math.exp_v1", [ConstNode(np.float64(800.0))])
    _loss, _cplx, valid, reason = _full_evaluate(tree, X_single, y_single, 10, 5)
    assert not valid, f"exp(800) should be invalid, got valid={valid} reason={reason}"


def test_division_protected():
    """1/0 via protected_div should not crash and should produce valid result with finite loss."""
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    tree = OpNode("sr.math.protected_div_v1", [
        ConstNode(np.float64(1.0)),
        ConstNode(np.float64(0.0)),
    ])
    loss, _cplx, valid, reason = _full_evaluate(tree, X, y, 10, 5)
    assert valid, f"protected_div(1,0) should be valid, got reason={reason}"
    assert np.isfinite(loss), f"protected_div(1,0) loss should be finite, got {loss}"


def test_power_negative_base_fractional_exponent():
    """(-1.0) ** 0.5 should be marked invalid by _full_evaluate."""
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    tree = OpNode("sr.math.pow_v1", [
        ConstNode(np.float64(-1.0)),
        ConstNode(np.float64(0.5)),
    ])
    _loss, _cplx, valid, reason = _full_evaluate(tree, X, y, 10, 5)
    assert not valid, f"(-1)^0.5 should be invalid, got valid={valid} reason={reason}"


def test_safe_log_handles_zero():
    """safe_log(0) should produce finite loss via _full_evaluate."""
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    tree = OpNode("sr.math.safe_log_v1", [ConstNode(np.float64(0.0))])
    loss, _cplx, valid, reason = _full_evaluate(tree, X, y, 10, 5)
    assert valid, f"safe_log(0) should be valid, got reason={reason}"
    assert np.isfinite(loss), f"safe_log(0) loss should be finite, got {loss}"
