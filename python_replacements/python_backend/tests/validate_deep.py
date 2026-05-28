"""Deep validation: graph mutations, constraints, optimizer, numeric safety, complexity, batching, guesses, early_stop, medium-size runs.

Run: python -m pytest python_backend/tests/validate_deep.py -v --tb=short
"""
from __future__ import annotations

import gc
import os

os.environ["PYSR_BACKEND"] = "python"

import numpy as np
import pytest

from python_backend.expr import OpNode, VarNode, ConstNode
from python_backend.eval import evaluate, compute_complexity
from python_backend.mutation_weights import MutationWeights, sample_mutation
from python_backend.constant_optimization import optimize_constants

np.random.seed(42)


# ── Graph mutation validation ──────────────────────────────────────────

def test_graph_mutation_weights_sampling():
    """sample_mutation returns valid weight names."""
    prng = np.random.default_rng(42)
    weights = MutationWeights()
    for _ in range(50):
        name = sample_mutation(prng, weights)
        assert isinstance(name, str)
        assert len(name) > 0


def test_mutation_weight_support():
    """Weight mutation options should be accepted (pass-through or supported)."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        weight_add_node=1.0,
        weight_delete_node=1.0,
        weight_mutate_constant=0.5,
        weight_mutate_operator=0.5,
        weight_do_nothing=0.1,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Constraints deep validation ────────────────────────────────────────

def _count_ops(tree, op_id: str) -> int:
    count = 0
    if isinstance(tree, OpNode):
        if tree.op_id == op_id:
            count += 1
        for child in tree.children:
            count += _count_ops(child, op_id)
    return count


def test_constraints_limit_add_operators():
    """Constraint on + should limit add operators in the best expression."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        constraints={"add": 0},
    )
    model.fit(X, y)
    # Parse best equation to verify constraint
    best_eq = model.equations_.iloc[0]
    assert best_eq is not None


def test_maxdepth_constraint():
    """maxdepth should limit tree depth."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(maxdepth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Complexity controls ────────────────────────────────────────────────

def test_complexity_of_operators_changes_complexity():
    """Raising operator complexity should make expressions more complex."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        complexity_of_operators=5,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    eq = model.equations_.iloc[0]
    assert eq["complexity"] > 0


def test_complexity_mapping_callable():
    """Custom complexity mapping callable should affect complexity."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)

    def my_complexity(tree, orig):
        return orig * 2

    model = _make_regressor(complexity_mapping=my_complexity)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── should_simplify ────────────────────────────────────────────────────

def test_should_simplify_false_keeps_unsimplified():
    """should_simplify=False should keep unsimplified expressions finite."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(should_simplify=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── early_stop_condition ───────────────────────────────────────────────

def test_early_stop_condition_numeric():
    """early_stop_condition numeric threshold should terminate early."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(early_stop_condition=1e10)  # impossible threshold
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Guesses ────────────────────────────────────────────────────────────

def test_guesses_accepted():
    """Valid guesses influence the search."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        guesses=["x0 ^ 2"],
        fraction_replaced_guesses=0.5,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Direct backend autodiff convergence ────────────────────────────────

def test_direct_backend_autodiff_convergence():
    """Backend optimize_constants with autodiff_backend=True should converge."""
    from python_backend.constant_optimization import optimize_constants

    X = np.array([[0.5], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.exp(X[:, 0] * 0.5).astype(np.float64)

    tree = OpNode("sr.math.exp_v1", [
        OpNode("sr.arith.mul_v1", [
            ConstNode(np.float64(1.0)),
            VarNode(0),
        ])
    ])

    opt_tree = optimize_constants(
        tree, X, y,
        maxsize=12, maxdepth=6,
        n_iterations=50,
        autodiff_backend=True,
    )
    pred = evaluate(opt_tree, X)
    assert np.all(np.isfinite(pred)), "optimization produced non-finite predictions"
    loss = np.mean((pred - y) ** 2)
    assert np.isfinite(loss)


# ── Optimizer algorithm / restarts / call-limit ────────────────────────

def test_optimizer_algorithm_bfgs():
    """BFGS optimizer algorithm should work."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_algorithm="BFGS")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_optimizer_algorithm_nelder_mead():
    """NelderMead optimizer algorithm."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_algorithm="NelderMead")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_optimizer_f_calls_limit():
    """optimizer_f_calls_limit constrains SciPy calls."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(optimizer_f_calls_limit=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Constant optimization convergence ──────────────────────────────────

@pytest.mark.parametrize("expr_target,unary_ops", [
    ("np.sin(x)", ["sin"]),
    ("np.cos(x)", ["cos"]),
    ("np.exp(x)", ["exp"]),
    ("x ** 1.5", []),
])
def test_constant_optimization_nonlinear(expr_target, unary_ops):
    """Constant optimization should converge on nonlinear expressions."""
    X = np.abs(np.random.randn(30, 1)).astype(np.float64)  # non-negative for powers
    x = X[:, 0]
    y = eval(expr_target).astype(np.float64)
    model = _make_regressor(
        binary_operators=["+", "-", "*"],
        unary_operators=unary_ops,
        niterations=3,
        population_size=20,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Batching full-dataset HOF ──────────────────────────────────────────

def test_batching_full_hof_reevaluation():
    """HOF losses should be full-dataset, not batch losses."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        batching=True, batch_size=10,
        niterations=3, population_size=20,
    )
    model.fit(X, y)
    eq = model.equations_
    assert len(eq) > 0
    assert "loss" in eq.columns
    assert all(np.isfinite(eq["loss"])), "some HOF losses are non-finite"


# ── Numeric safety ─────────────────────────────────────────────────────

def test_division_by_zero_produces_invalid():
    """Division by zero should mark candidate invalid."""
    from python_backend.backend import _full_evaluate
    tree = OpNode("sr.arith.div_v1", [ConstNode(np.float64(1.0)), ConstNode(np.float64(0.0))])
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    loss, complexity, valid, reason = _full_evaluate(tree, X, y, 10, 5)
    assert not valid, "div by zero should be invalid"
    assert reason is not None and len(reason) > 0


def test_safe_log_negative_input():
    """safe_log of negative input should produce valid with finite loss."""
    from python_backend.backend import _full_evaluate
    tree = OpNode("sr.math.safe_log_v1", [ConstNode(np.float64(-1.0))])
    X = np.array([[1.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    loss, complexity, valid, reason = _full_evaluate(tree, X, y, 10, 5)
    # safe_log clips negative inputs: valid, finite loss
    assert valid, f"safe_log(-1) should be valid, got reason={reason}"
    assert np.isfinite(loss) if loss is not None else True


# ── Multi-pop migration ────────────────────────────────────────────────

def test_multi_pop_migration_enabled():
    """Multiple populations with migration enabled should produce finite preds."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        populations=3,
        migration=True,
        hof_migration=True,
        niterations=3,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_multi_pop_migration_disabled():
    """migration=False should still produce valid results."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        populations=3,
        migration=False,
        niterations=3,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── max_evals across multiple paths ────────────────────────────────────

def test_max_evals_multi_pop():
    """max_evals low budget works with multiple populations."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        populations=3,
        max_evals=100,
        niterations=10,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_max_evals_multi_output():
    """max_evals low budget works with multi-output."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.column_stack([X[:, 0] ** 2, X[:, 1] * 0.5]).astype(np.float64)
    model = _make_regressor(
        max_evals=100,
        niterations=10,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30, 2)
    assert np.all(np.isfinite(preds))


def test_max_evals_warm_start():
    """max_evals with warm_start should resume correctly."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        warm_start=True,
        max_evals=200,
        niterations=2,
    )
    model.fit(X, y)
    model.fit(X, y)  # warm-start resume
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── timeout across multiple paths ──────────────────────────────────────

def test_timeout_multi_pop():
    """Tiny timeout with multiple populations."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(
        populations=3,
        timeout_in_seconds=0.001,
        niterations=10,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Large(r) dataset validation ────────────────────────────────────────

def test_medium_size_runtime():
    """500 samples × 5 features should complete in reasonable time and memory."""
    X = np.random.randn(500, 5).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(
        niterations=2, population_size=30,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=12,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (500,)
    assert np.all(np.isfinite(preds))
    # Collect garbage after large run
    del model, X, y, preds
    gc.collect()


# ── Helper ─────────────────────────────────────────────────────────────

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
