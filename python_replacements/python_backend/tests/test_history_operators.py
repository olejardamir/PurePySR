from __future__ import annotations

import numpy as np

from python_backend.eval import evaluate
from python_backend.expr import ConstNode, OpNode, VarNode, node_to_sympy, parse_expression
from python_backend.ops import OP_ID_TO_ARITY, resolve_operator_tokens


def test_delay_operator_uses_prior_rows():
    X = np.arange(1.0, 6.0).reshape(-1, 1)
    tree = OpNode(resolve_operator_tokens(["delay"])[0], [VarNode(0), ConstNode(2.0)])
    np.testing.assert_allclose(evaluate(tree, X), [1.0, 1.0, 1.0, 2.0, 3.0])


def test_simple_moving_average_operator():
    X = np.arange(1.0, 6.0).reshape(-1, 1)
    tree = OpNode(resolve_operator_tokens(["sma"])[0], [VarNode(0), ConstNode(3.0)])
    np.testing.assert_allclose(evaluate(tree, X), [1.0, 1.5, 2.0, 3.0, 4.0])


def test_weighted_moving_average_operator():
    X = np.arange(1.0, 5.0).reshape(-1, 1)
    tree = OpNode(resolve_operator_tokens(["wma"])[0], [VarNode(0), ConstNode(3.0)])
    expected = [
        1.0,
        (1.0 * 1.0 + 2.0 * 2.0) / 3.0,
        (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 6.0,
        (2.0 * 1.0 + 3.0 * 2.0 + 4.0 * 3.0) / 6.0,
    ]
    np.testing.assert_allclose(evaluate(tree, X), expected)


def test_modified_moving_average_operator():
    X = np.array([1.0, 4.0, 7.0, 10.0]).reshape(-1, 1)
    tree = OpNode(resolve_operator_tokens(["mma"])[0], [VarNode(0), ConstNode(3.0)])
    expected = [1.0]
    for value in X[1:, 0]:
        expected.append((expected[-1] * 2.0 + value) / 3.0)
    np.testing.assert_allclose(evaluate(tree, X), expected)


def test_moving_median_operator():
    X = np.array([3.0, 1.0, 7.0, 2.0, 5.0]).reshape(-1, 1)
    tree = OpNode(resolve_operator_tokens(["median"])[0], [VarNode(0), ConstNode(3.0)])
    np.testing.assert_allclose(evaluate(tree, X), [3.0, 2.0, 3.0, 2.0, 5.0])


def test_history_operator_parser_and_sympy_export():
    tree = parse_expression("sma(x0, 3)")
    X = np.arange(1.0, 5.0).reshape(-1, 1)
    np.testing.assert_allclose(evaluate(tree, X), [1.0, 1.5, 2.0, 3.0])

    op_id = resolve_operator_tokens(["sma"])[0]
    assert OP_ID_TO_ARITY[op_id] == 2
    exported = node_to_sympy(
        OpNode(op_id, [VarNode(0), ConstNode(3.0)]),
        variable_names=["x0"],
    )
    assert str(exported) == "sma(x0, 3.0)"
