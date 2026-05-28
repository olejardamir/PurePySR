from __future__ import annotations

import numpy as np
import sympy

from python_backend.expr import OpNode, VarNode, ConstNode, node_to_sympy, parse_canonical
from python_backend.gradients import _evaluate_with_gradient
from python_backend.backend import _full_evaluate
from python_backend.eval import evaluate, compute_complexity


def test_exp_gradient_against_finite_differences():
    """AD gradient for exp(a*x) should match central finite differences."""
    a = 2.5
    x = np.array([[-1.0], [0.0], [0.5], [1.0], [3.0]], dtype=np.float64)

    c_node = ConstNode(np.float64(a))
    tree = OpNode("sr.math.exp_v1", [
        OpNode("sr.arith.mul_v1", [
            c_node,
            VarNode(0),
        ])
    ])

    val, grad = _evaluate_with_gradient(tree, x, [c_node])
    assert np.all(np.isfinite(val)), "exp evaluation should be finite"
    assert grad is not None, "gradient should be computed"
    assert grad.shape == (1, 5), f"expected (1, 5), got {grad.shape}"

    # FD w.r.t. the constant a (not x)
    h = 1e-6
    a_plus = a + h
    tree_plus = OpNode("sr.math.exp_v1", [
        OpNode("sr.arith.mul_v1", [
            ConstNode(np.float64(a_plus)),
            VarNode(0),
        ])
    ])
    val_plus = evaluate(tree_plus, x)

    a_minus = a - h
    tree_minus = OpNode("sr.math.exp_v1", [
        OpNode("sr.arith.mul_v1", [
            ConstNode(np.float64(a_minus)),
            VarNode(0),
        ])
    ])
    val_minus = evaluate(tree_minus, x)

    fd_grad = (val_plus - val_minus) / (2 * h)  # shape (5,)

    grad_1d = grad[0, :]  # gradient w.r.t. the single constant
    assert np.allclose(grad_1d, fd_grad, rtol=1e-4, atol=1e-6), (
        f"AD grad {grad_1d} differs from FD grad {fd_grad}"
    )


def test_exp_overflow_makes_candidate_invalid():
    """_full_evaluate on exp of large input marks candidate invalid."""
    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    x = np.array([[800.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)

    loss, complexity, valid, reason = _full_evaluate(tree, x, y, 10, 5)
    assert not valid, "exp(800) should be invalid"
    assert reason is not None and len(reason) > 0, "invalid reason should be non-empty"


def test_exp_moderate_input_ok():
    """exp on moderate input should be valid."""
    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    x = np.array([[0.5]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)

    loss, complexity, valid, reason = _full_evaluate(tree, x, y, 10, 5)
    assert valid, "exp(0.5) should be valid"
    assert loss is not None and np.isfinite(loss)


def test_exp_tree_evaluates_directly():
    """Build a tree with exp, evaluate it, verify finite result."""
    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    pred = evaluate(tree, x)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
    assert np.allclose(pred, np.exp(x[:, 0]))


def test_exp_sympy_round_trip():
    """exp node converts to sympy and back via canonical."""
    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    sp = node_to_sympy(tree)
    expected = sympy.exp(sympy.Symbol("x0"))
    assert sp == expected, f"expected {expected}, got {sp}"

    canonical = tree.canonical()
    restored = parse_canonical(canonical)
    assert restored.canonical() == canonical, "canonical round-trip should match"


def test_exp_with_units_violates():
    """exp of a dimensionful input should violate dimensional constraints."""
    from python_backend.dimensional import check_dimensions

    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    # x0 has length dimension — exp requires dimensionless
    violates = check_dimensions(
        tree, x_units=["m"], y_units="m",
        allow_wildcards=False,
    )
    assert violates, "exp of length should violate dimensions"


def test_exp_of_dimensionless_ok():
    """exp of a dimensionless input with dimensionless target should pass."""
    from python_backend.dimensional import check_dimensions

    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    ok = not check_dimensions(
        tree, x_units=["kg/kg"], y_units="",
        allow_wildcards=False,
    )
    assert ok, "exp of dimensionless with dimensionless target should not violate"


def test_exp_dimensionless_but_output_mismatch():
    """exp of dimensionless input fails when y_units are dimensionful."""
    from python_backend.dimensional import check_dimensions

    tree = OpNode("sr.math.exp_v1", [VarNode(0)])
    violates = check_dimensions(
        tree, x_units=["kg/kg"], y_units="m",
        allow_wildcards=False,
    )
    assert violates, "exp output (dimensionless) should not match y_units=m"
