from __future__ import annotations

import numpy as np

from python_backend.constant_optimization import optimize_constants
from python_backend.eval import evaluate, compute_loss
from python_backend.expr import ConstNode, OpNode, VarNode


def test_optimize_constants_reduces_loss():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    optimized = optimize_constants(expr, X, y, maxsize=10, maxdepth=5, n_iterations=5)

    y_pred_before = evaluate(expr, X)
    y_pred_after = evaluate(optimized, X)
    loss_before, valid_before, _ = compute_loss(y, y_pred_before)
    loss_after, valid_after, _ = compute_loss(y, y_pred_after)

    assert valid_before
    assert valid_after
    assert loss_after < loss_before


def test_optimize_constants_no_constants_unchanged():
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = X[:, 0]
    expr = VarNode(0)

    result = optimize_constants(expr, X, y, maxsize=10, maxdepth=5, n_iterations=5)

    assert result is expr


def test_optimize_constants_exact_constants_unchanged():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
        ConstNode(3.0),
    ])

    optimized = optimize_constants(expr, X, y, maxsize=10, maxdepth=5, n_iterations=5)

    y_pred_before = evaluate(expr, X)
    y_pred_after = evaluate(optimized, X)
    loss_before, _, _ = compute_loss(y, y_pred_before)
    loss_after, _, _ = compute_loss(y, y_pred_after)

    assert loss_after <= loss_before


def test_optimize_constants_scipy_lbfgsb():
    """Test scipy L-BFGS-B optimizer."""
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    optimized = optimize_constants(
        expr, X, y, maxsize=10, maxdepth=5,
        n_iterations=5, algorithm="L-BFGS-B",
    )

    y_pred_before = evaluate(expr, X)
    y_pred_after = evaluate(optimized, X)
    loss_before, valid_before, _ = compute_loss(y, y_pred_before)
    loss_after, valid_after, _ = compute_loss(y, y_pred_after)

    assert valid_before
    assert valid_after
    assert loss_after < loss_before


def test_optimize_constants_exp_expr():
    """Exp expressions can have constants optimized."""
    rng = np.random.default_rng(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.exp(0.5 * X[:, 0])

    expr = OpNode("sr.math.exp_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
    ])

    optimized = optimize_constants(expr, X, y, maxsize=10, maxdepth=5, n_iterations=5)
    y_pred = evaluate(optimized, X)
    loss, valid, _ = compute_loss(y, y_pred)
    assert valid, "exp expression should evaluate finitely"
    assert np.isfinite(loss)


def test_optimize_constants_scipy_f_calls_limit():
    """Scipy optimizer respects f_calls_limit."""
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    from python_backend.constant_optimization import _scipy_optimize

    # Very tight limit should still converge somewhat
    result, loss = _scipy_optimize(
        expr, X, y, f_calls_limit=5, method="L-BFGS-B",
    )
    assert np.isfinite(loss)
    assert loss > 0  # should not be zero with very few calls but at least finite
