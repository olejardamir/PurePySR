from __future__ import annotations

import numpy as np

from python_backend.constant_optimization import optimize_constants
from python_backend.eval import clear_tree_cache, evaluate, compute_loss
from python_backend.expr import ConstNode, OpNode, VarNode
from python_backend.gradients import _collect_consts, _evaluate_with_gradient, loss_gradient


def _finite_diff_gradient(
    params: np.ndarray,
    expr,
    X: np.ndarray,
    y: np.ndarray,
    consts,
    loss_fn=None,
    weights=None,
    eps: float = 1e-8,
) -> np.ndarray:
    grad = np.empty_like(params)
    base_loss = None
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        for c, p in zip(consts, params):
            c.value = float(p)
        clear_tree_cache(expr)
        y_pred = evaluate(expr, X)
        loss_up, valid, _ = compute_loss(y, y_pred, loss_fn=loss_fn, weights=weights)

        params[i] = orig - eps
        for c, p in zip(consts, params):
            c.value = float(p)
        clear_tree_cache(expr)
        y_pred = evaluate(expr, X)
        loss_down, _, _ = compute_loss(y, y_pred, loss_fn=loss_fn, weights=weights)

        grad[i] = (loss_up - loss_down) / (2 * eps)
        params[i] = orig

    for c, p in zip(consts, params):
        c.value = float(p)
    return grad


def test_gradient_linear():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_complex_tree():
    rng = np.random.default_rng(42)
    X = np.linspace(0.1, 2.0, 50).reshape(-1, 1)
    y = np.sin(0.5 * X[:, 0]) + 0.3 * X[:, 0] ** 2

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.math.sin_v1", [
            OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(0.7)])
        ]),
        OpNode("sr.arith.mul_v1", [
            OpNode("sr.math.pow_v1", [VarNode(0), ConstNode(2.0)]),
            ConstNode(0.3),
        ]),
    ])

    consts = _collect_consts(expr)
    assert len(consts) == 3
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-4), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_protected_div():
    rng = np.random.default_rng(42)
    X = np.linspace(0.5, 2.0, 50).reshape(-1, 1)
    y = X[:, 0] / 3.0

    expr = OpNode("sr.math.protected_div_v1", [
        VarNode(0),
        ConstNode(1.0),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 3.0
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_safe_log():
    rng = np.random.default_rng(42)
    X = np.linspace(0.1, 2.0, 50).reshape(-1, 1)
    y = np.log(np.abs(X[:, 0]) + 1e-8)

    expr = OpNode("sr.math.safe_log_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(1.0)]),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 1.0
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_abs():
    rng = np.random.default_rng(42)
    X = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
    y = np.abs(1.5 * X[:, 0])

    expr = OpNode("sr.math.abs_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(1.0)]),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 1.5
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_cos():
    rng = np.random.default_rng(42)
    X = np.linspace(0.0, np.pi, 50).reshape(-1, 1)
    y = np.cos(2.0 * X[:, 0])

    expr = OpNode("sr.math.cos_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(1.0)]),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 2.0
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_sub():
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X[:, 0] - 2.0

    expr = OpNode("sr.arith.sub_v1", [
        VarNode(0),
        ConstNode(0.0),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 2.0
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_gradient_multi_const():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )
    assert len(analytic) == 2  # two constants
    assert analytic.shape == (2,)


def test_gradient_with_weights():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0
    weights = np.ones(50)
    weights[:10] = 2.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts, weights=weights)
    numeric = _finite_diff_gradient(params.copy(), expr, X, y, consts, weights=weights)

    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5), (
        f"analytic={analytic}, numeric={numeric}"
    )


def test_autodiff_optimization_lbfgsb():
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
        autodiff_backend=True,
    )

    y_pred_before = evaluate(expr, X)
    y_pred_after = evaluate(optimized, X)
    loss_before, valid_before, _ = compute_loss(y, y_pred_before)
    loss_after, valid_after, _ = compute_loss(y, y_pred_after)

    assert valid_before
    assert valid_after
    assert loss_after < loss_before


def test_autodiff_optimization_bfgs():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(5.0)]),
        ConstNode(10.0),
    ])

    optimized = optimize_constants(
        expr, X, y, maxsize=10, maxdepth=5,
        n_iterations=5, algorithm="BFGS",
        autodiff_backend=True,
    )

    y_pred_after = evaluate(optimized, X)
    loss_after, valid_after, _ = compute_loss(y, y_pred_after)
    assert valid_after
    assert loss_after < 1.0


def test_autodiff_vs_fd_convergence():
    rng = np.random.default_rng(42)
    X = np.linspace(0.1, 2.0, 100).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.5

    expr = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(10.0)]),
        ConstNode(10.0),
    ])

    result_ad = optimize_constants(
        expr, X, y, maxsize=20, maxdepth=10,
        n_iterations=10, algorithm="L-BFGS-B",
        autodiff_backend=True,
        nrestarts=2,
    )
    result_fd = optimize_constants(
        expr, X, y, maxsize=20, maxdepth=10,
        n_iterations=10, algorithm="L-BFGS-B",
        autodiff_backend=False,
        nrestarts=2,
    )

    loss_ad, valid_ad, _ = compute_loss(y, evaluate(result_ad, X))
    loss_fd, valid_fd, _ = compute_loss(y, evaluate(result_fd, X))

    assert valid_ad
    assert valid_fd
    assert loss_ad <= loss_fd + 1e-6


def test_gradient_no_consts():
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    expr = VarNode(0)

    consts = _collect_consts(expr)
    assert len(consts) == 0

    params = np.array([], dtype=np.float64)
    grad = loss_gradient(params, expr, X, X[:, 0], consts)
    assert len(grad) == 0


def test_gradient_less():
    rng = np.random.default_rng(42)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = np.where(X[:, 0] < 0.5, 1.0, 0.0)

    expr = OpNode("sr.arith.less_v1", [
        VarNode(0),
        ConstNode(0.0),
    ])

    consts = _collect_consts(expr)
    consts[0].value = 0.5
    params = np.array([c.value for c in consts], dtype=np.float64)

    analytic = loss_gradient(params, expr, X, y, consts)
    assert np.all(analytic == 0.0)


def test_gradient_nan_prediction_returns_zero():
    """When y_pred contains NaN, gradient should be zero, not NaN."""
    rng = np.random.default_rng(42)
    X = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    expr = OpNode("sr.math.pow_v1", [
        VarNode(0),
        ConstNode(0.5),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    grad = loss_gradient(params, expr, X, y, consts)
    assert np.isfinite(grad[0]), f"gradient should be finite but got {grad[0]}"
    assert np.all(np.isfinite(grad))


def test_gradient_zero_base_pow():
    """∂(a^b)/∂a at a=0 should give finite gradient."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 4.0])

    expr = OpNode("sr.math.pow_v1", [
        VarNode(0),
        ConstNode(2.0),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    grad = loss_gradient(params, expr, X, y, consts)
    assert np.isfinite(grad[0]), f"gradient should be finite but got {grad[0]}"


def test_gradient_near_singular_div():
    """Gradient of protected_div near zero denominator should be finite."""
    X = np.array([[1.0, 1.0, 1.0]]).T
    y = np.array([10.0, 10.0, 10.0])

    expr = OpNode("sr.math.protected_div_v1", [
        VarNode(0),
        ConstNode(1e-10),
    ])

    consts = _collect_consts(expr)
    params = np.array([c.value for c in consts], dtype=np.float64)

    grad = loss_gradient(params, expr, X, y, consts)
    assert np.isfinite(grad[0]), f"gradient should be finite but got {grad[0]}"
