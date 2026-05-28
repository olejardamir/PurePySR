from __future__ import annotations

import numbers
from typing import Any, Callable

import numpy as np

from python_backend.expr import Node, VarNode, ConstNode, OpNode
from python_backend.ops import OP_ID_TO_FN, OP_ID_TO_FN_OUT

LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
ScalarLossFn = Callable[[np.ndarray, np.ndarray], float]


def evaluate(node: Node, X: np.ndarray) -> Any:
    if _is_purely_numeric(node):
        return _evaluate_numeric_fast(node, X)
    return _evaluate_generic(node, X)


def _is_purely_numeric(node: Node) -> bool:
    if isinstance(node, VarNode):
        return True
    if isinstance(node, ConstNode):
        return isinstance(node.value, numbers.Number)
    if isinstance(node, OpNode):
        if node.op_id not in OP_ID_TO_FN_OUT:
            return False
        return all(_is_purely_numeric(c) for c in node.children)
    return False


def _evaluate_numeric_fast(node: Node, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    max_depth = _compute_depth(node) + 1
    scratch = [np.empty(n, dtype=np.float64) for _ in range(max_depth)]

    stack: list[tuple] = [(node, 0, 0)]

    while stack:
        cur_node, state, depth = stack.pop()

        if isinstance(cur_node, VarNode):
            if cur_node._eval_cache is not None:
                scratch[depth][:] = cur_node._eval_cache
            else:
                scratch[depth][:] = X[:, cur_node.index]
                cur_node._eval_cache = scratch[depth].copy()
        elif isinstance(cur_node, ConstNode):
            if cur_node._eval_cache is not None:
                scratch[depth][:] = cur_node._eval_cache
            else:
                scratch[depth][:] = cur_node.value
                cur_node._eval_cache = scratch[depth].copy()
        elif isinstance(cur_node, OpNode):
            if state == 0 and cur_node._eval_cache is not None:
                scratch[depth][:] = cur_node._eval_cache
                continue
            children = cur_node.children
            nch = len(children)

            if nch == 1:
                if state == 0:
                    stack.append((cur_node, 1, depth))
                    stack.append((children[0], 0, depth + 1))
                else:
                    fn_out = OP_ID_TO_FN_OUT.get(cur_node.op_id)
                    if fn_out is not None:
                        fn_out(scratch[depth + 1], out=scratch[depth])
                    else:
                        OP_ID_TO_FN[cur_node.op_id](scratch[depth + 1], out=scratch[depth])
                    cur_node._eval_cache = scratch[depth].copy()
            elif nch == 2:
                if state == 0:
                    stack.append((cur_node, 1, depth))
                    stack.append((children[0], 0, depth + 1))
                elif state == 1:
                    scratch[depth][:] = scratch[depth + 1]
                    stack.append((cur_node, 2, depth))
                    stack.append((children[1], 0, depth + 1))
                else:
                    fn_out = OP_ID_TO_FN_OUT.get(cur_node.op_id)
                    if fn_out is not None:
                        fn_out(scratch[depth], scratch[depth + 1], out=scratch[depth])
                    else:
                        scratch[depth][:] = OP_ID_TO_FN[cur_node.op_id](scratch[depth], scratch[depth + 1])
                    cur_node._eval_cache = scratch[depth].copy()
            else:
                raise ValueError(f"unsupported arity: {nch}")
        else:
            raise ValueError(f"unknown node type: {type(cur_node)}")

    return scratch[0]


def _evaluate_generic(node: Node, X: np.ndarray) -> Any:
    values: list[Any] = []
    control: list[tuple[Node, int]] = [(node, 0)]

    while control:
        cur_node, child_idx = control.pop()

        if isinstance(cur_node, VarNode):
            if cur_node._eval_cache is not None:
                values.append(cur_node._eval_cache)
            else:
                result = X[:, cur_node.index]
                cur_node._eval_cache = result
                values.append(result)
        elif isinstance(cur_node, ConstNode):
            if cur_node._eval_cache is not None:
                values.append(cur_node._eval_cache)
            else:
                values.append(cur_node.value)
                cur_node._eval_cache = cur_node.value
        elif isinstance(cur_node, OpNode):
            if child_idx == 0 and cur_node._eval_cache is not None:
                values.append(cur_node._eval_cache)
                continue
            children = cur_node.children
            if child_idx < len(children):
                control.append((cur_node, child_idx + 1))
                control.append((children[child_idx], 0))
            else:
                n = len(children)
                args = values[-n:]
                del values[-n:]
                result = OP_ID_TO_FN[cur_node.op_id](*args)
                cur_node._eval_cache = result
                values.append(result)
        else:
            raise ValueError(f"unknown node type: {type(cur_node)}")

    return values[0]


def clear_tree_cache(node: Node) -> None:
    """Recursively clear cached evaluation results."""
    node._eval_cache = None
    node._complexity_cache = None
    node._depth_cache = None
    if isinstance(node, OpNode):
        for c in node.children:
            clear_tree_cache(c)


def compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_fn: LossFn | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, bool, str]:
    if not np.all(np.isfinite(y_pred)):
        return (float("inf"), False, "SR-INV-NONFINITE-001")
    if loss_fn is not None:
        elementwise = loss_fn(y_pred, y_true)
    else:
        elementwise = (y_pred - y_true) ** 2
    if not np.all(np.isfinite(elementwise)):
        return (float("inf"), False, "SR-INV-NONFINITE-001")
    if weights is not None:
        loss = float(np.average(elementwise, weights=weights))
    else:
        loss = float(np.mean(elementwise))
    if not np.isfinite(loss):
        return (float("inf"), False, "SR-INV-OBJ-001")
    return (loss, True, "")


def compute_complexity(
    node: Node,
    const_weight: int = 1,
    op_weight: int = 1,
    var_weight: int = 1,
    mapping: Callable | None = None,
) -> int:
    if mapping is not None:
        return mapping(node)
    cached = getattr(node, '_complexity_cache', None)
    if cached is not None:
        return cached
    if isinstance(node, ConstNode):
        complexity = const_weight
    elif isinstance(node, VarNode):
        complexity = var_weight
    else:
        complexity = op_weight + sum(
            compute_complexity(c, const_weight, op_weight, var_weight, mapping)
            for c in node.children
        )
    node._complexity_cache = complexity
    return complexity


def check_constraints(
    node: Node,
    maxsize: int,
    maxdepth: int,
    constraints: dict[str, int | tuple[int, ...]] | None = None,
    nested_constraints: dict[str, dict[str, int]] | None = None,
) -> tuple[bool, str]:
    if compute_complexity(node) > maxsize:
        return (False, "SR-INV-CONSTR-001")
    depth = _compute_depth(node)
    if depth > maxdepth:
        return (False, "SR-INV-CONSTR-001")
    if constraints:
        ok, reason = _check_operator_constraints(node, constraints)
        if not ok:
            return (False, reason)
    if nested_constraints:
        from python_backend.expr import check_nested_constraints as _check_nested
        ok, reason = _check_nested(node, nested_constraints)
        if not ok:
            return (False, reason)
    return (True, "")


_ConstraintValue = int | tuple[int, ...] | list[int]


def _check_operator_constraints(
    node: Node,
    constraints: dict[str, _ConstraintValue],
) -> tuple[bool, str]:
    if not isinstance(node, OpNode):
        return (True, "")
    limit = constraints.get(node.op_id)
    if limit is not None:
        if isinstance(limit, int):
            if compute_complexity(node.children[0]) > limit:
                return (False, "SR-INV-CONSTR-002")
        elif isinstance(limit, (tuple, list)):
            for i, c in enumerate(node.children):
                if i < len(limit):
                    pos_limit = limit[i]
                    if pos_limit >= 0 and compute_complexity(c) > pos_limit:
                        return (False, "SR-INV-CONSTR-002")
    for child in node.children:
        if isinstance(child, OpNode):
            ok, reason = _check_operator_constraints(child, constraints)
            if not ok:
                return (False, reason)
    return (True, "")


def _compute_depth(node: Node) -> int:
    cached = getattr(node, '_depth_cache', None)
    if cached is not None:
        return cached
    if isinstance(node, (VarNode, ConstNode)):
        depth = 0
    elif not node.children:
        depth = 0
    else:
        depth = 1 + max(_compute_depth(c) for c in node.children)
    node._depth_cache = depth
    return depth
