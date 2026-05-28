from __future__ import annotations

import numpy as np
import scipy.special as sp_special

from python_backend.eval import evaluate
from python_backend.expr import OpNode, VarNode, node_to_sympy, parse_expression
from python_backend.ops import OP_ID_TO_ARITY, OP_ID_TO_FN, resolve_operator_tokens


UNARY_EXPECTED = {
    "tan": np.tan,
    "sqrt": lambda x: np.sqrt(np.where(x >= 0.0, x, np.nan)),
    "logistic": sp_special.expit,
    "step": lambda x: np.where(x > 0.0, 1.0, 0.0),
    "sgn": np.sign,
    "gauss": lambda x: np.exp(-(x * x)),
    "tanh": np.tanh,
    "erf": sp_special.erf,
    "erfc": sp_special.erfc,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.rint,
    "asin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
    "acos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    "atan": np.arctan,
    "neg": np.negative,
}


BINARY_EXPECTED = {
    "equal": lambda x, y: np.where(np.isclose(x, y), 1.0, 0.0),
    "less": lambda x, y: np.where(x < y, 1.0, 0.0),
    "less_or_equal": lambda x, y: np.where(x <= y, 1.0, 0.0),
    "greater": lambda x, y: np.where(x > y, 1.0, 0.0),
    "greater_or_equal": lambda x, y: np.where(x >= y, 1.0, 0.0),
    "and": lambda x, y: np.where((x > 0.0) & (y > 0.0), 1.0, 0.0),
    "or": lambda x, y: np.where((x > 0.0) | (y > 0.0), 1.0, 0.0),
    "xor": lambda x, y: np.where((x > 0.0) ^ (y > 0.0), 1.0, 0.0),
    "min": np.minimum,
    "max": np.maximum,
    "mod": lambda x, y: np.where(np.abs(y) < 1e-8, 0.0, np.mod(x, y)),
    "atan2": np.arctan2,
}


def test_eureqa_unary_operator_evaluation():
    x = np.array([-1.5, -0.25, 0.0, 0.5, 2.0], dtype=np.float64)
    X = x.reshape(-1, 1)

    for token, expected_fn in UNARY_EXPECTED.items():
        op_id = resolve_operator_tokens([token])[0]
        assert OP_ID_TO_ARITY[op_id] == 1
        result = evaluate(OpNode(op_id, [VarNode(0)]), X)
        expected = expected_fn(x)
        np.testing.assert_allclose(result, expected, equal_nan=True)


def test_eureqa_binary_operator_evaluation():
    x = np.array([-1.5, -0.25, 0.0, 0.5, 2.0], dtype=np.float64)
    y = np.array([1.0, -0.25, 0.0, -0.5, 0.0], dtype=np.float64)
    X = np.column_stack([x, y])

    for token, expected_fn in BINARY_EXPECTED.items():
        op_id = resolve_operator_tokens([token])[0]
        assert OP_ID_TO_ARITY[op_id] == 2
        result = evaluate(OpNode(op_id, [VarNode(0), VarNode(1)]), X)
        expected = expected_fn(x, y)
        np.testing.assert_allclose(result, expected, equal_nan=True)


def test_eureqa_function_parser_supports_multi_arg_calls():
    tree = parse_expression("max(gauss(x0), min(x1, 2.0))")
    X = np.array([[0.0, 1.5], [2.0, -1.0]], dtype=np.float64)
    result = evaluate(tree, X)
    expected = np.maximum(np.exp(-(X[:, 0] ** 2)), np.minimum(X[:, 1], 2.0))
    np.testing.assert_allclose(result, expected)


def test_eureqa_sympy_exports():
    for token in [
        "tan",
        "sqrt",
        "logistic",
        "gauss",
        "erf",
        "erfc",
        "min",
        "max",
        "atan2",
    ]:
        op_id = resolve_operator_tokens([token])[0]
        arity = OP_ID_TO_ARITY[op_id]
        children = [VarNode(i) for i in range(arity)]
        expr = node_to_sympy(OpNode(op_id, children), variable_names=["x0", "x1"])
        assert expr is not None


def test_factorial_uses_gamma_extension():
    x = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    fn = OP_ID_TO_FN[resolve_operator_tokens(["factorial"])[0]]
    np.testing.assert_allclose(fn(x), sp_special.gamma(x + 1.0))
