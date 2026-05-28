from __future__ import annotations

from typing import Any

import numpy as np

from python_backend.expr import ConstNode, Node, OpNode, VarNode
from python_backend.ops import OP_ID_TO_ARITY, OP_ID_TO_FN, OP_ID_TO_FN_OUT

NODE_DTYPE = np.dtype([
    ("type", "u1"),
    ("index", "i4"),
    ("value", "f8"),
    ("left", "i4"),
    ("right", "i4"),
], align=True)

_VAR = 0
_CONST = 1
_OP = 2

_OP_IDS: list[str] = []
_OP_ID_TO_INDEX: dict[str, int] = {}


def _ensure_op_registry() -> None:
    if not _OP_IDS:
        for oid in sorted(OP_ID_TO_ARITY):
            _OP_ID_TO_INDEX[oid] = len(_OP_IDS)
            _OP_IDS.append(oid)


def _op_to_index(op_id: str) -> int:
    _ensure_op_registry()
    return _OP_ID_TO_INDEX[op_id]


def _index_to_op(idx: int) -> str:
    _ensure_op_registry()
    return _OP_IDS[idx]


class FlatTree:
    __slots__ = ("arr", "root")

    def __init__(self, arr: np.ndarray, root: int | None = None) -> None:
        self.arr = arr
        self.root = len(arr) - 1 if root is None else root

    def __len__(self) -> int:
        return len(self.arr)

    @classmethod
    def from_node(cls, node: Node) -> FlatTree:
        arr = _tree_to_flat(node)
        return cls(arr)

    def to_node(self) -> Node:
        return _flat_to_tree(self.arr, self.root)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return _evaluate_flat(self.arr, self.root, X)

    def copy(self) -> FlatTree:
        return FlatTree(self.arr.copy(), self.root)

    def replace_subtree(self, target: int, replacement: FlatTree) -> FlatTree:
        new_arr, new_root = _replace_subtree(self.arr, self.root, target, replacement.arr, replacement.root)
        return FlatTree(new_arr, new_root)


def _tree_to_flat(node: Node) -> np.ndarray:
    nodes: list[np.void] = []
    _flatten(node, nodes)
    arr = np.empty(len(nodes), dtype=NODE_DTYPE)
    for i, n in enumerate(nodes):
        arr[i] = n
    return arr


def _flatten(node: Node, out: list[np.void]) -> int:
    idx = len(out)
    if isinstance(node, VarNode):
        out.append(np.void((_VAR, node.index, 0.0, -1, -1), dtype=NODE_DTYPE))
    elif isinstance(node, ConstNode):
        out.append(np.void((_CONST, 0, node.value, -1, -1), dtype=NODE_DTYPE))
    elif isinstance(node, OpNode):
        children = node.children
        child_indices = [_flatten(c, out) for c in children]
        idx = len(out)
        left = child_indices[0] if len(child_indices) > 0 else -1
        right = child_indices[1] if len(child_indices) > 1 else -1
        out.append(np.void((_OP, _op_to_index(node.op_id), 0.0, left, right), dtype=NODE_DTYPE))
    else:
        raise ValueError(f"unknown node type: {type(node)}")
    return idx


def _flat_to_tree(arr: np.ndarray, idx: int) -> Node:
    row = arr[idx]
    t = int(row["type"])
    if t == _VAR:
        return VarNode(int(row["index"]))
    if t == _CONST:
        return ConstNode(float(row["value"]))
    if t == _OP:
        op_id = _index_to_op(int(row["index"]))
        left = int(row["left"])
        right = int(row["right"])
        children: list[Node] = []
        if left >= 0:
            children.append(_flat_to_tree(arr, left))
        if right >= 0:
            children.append(_flat_to_tree(arr, right))
        return OpNode(op_id, children)
    raise ValueError(f"unknown node type: {t}")


def _evaluate_flat(arr: np.ndarray, root: int, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    m = len(arr)
    scratch = [np.empty(n, dtype=np.float64) for _ in range(m)]
    stack: list[tuple[int, int]] = [(root, 0)]

    while stack:
        i, state = stack.pop()
        row = arr[i]
        t = int(row["type"])

        if t == _VAR:
            scratch[i][:] = X[:, int(row["index"])]
        elif t == _CONST:
            scratch[i][:] = float(row["value"])
        elif t == _OP:
            left = int(row["left"])
            right = int(row["right"])
            if state == 0:
                stack.append((i, 1))
                if right >= 0:
                    stack.append((right, 0))
                if left >= 0:
                    stack.append((left, 0))
            else:
                op_id = _index_to_op(int(row["index"]))
                fn_out = OP_ID_TO_FN_OUT.get(op_id)
                if fn_out is not None:
                    if right < 0:
                        fn_out(scratch[left], out=scratch[i])
                    else:
                        fn_out(scratch[left], scratch[right], out=scratch[i])
                else:
                    fn = OP_ID_TO_FN[op_id]
                    if right < 0:
                        scratch[i][:] = fn(scratch[left])
                    else:
                        scratch[i][:] = fn(scratch[left], scratch[right])
        else:
            raise ValueError(f"unknown node type: {t}")

    return scratch[root]


def _subtree_indices(arr: np.ndarray, root: int) -> list[int]:
    result: list[int] = []
    stack = [root]
    while stack:
        i = stack.pop()
        result.append(i)
        row = arr[i]
        l = int(row["left"])
        r = int(row["right"])
        if r >= 0:
            stack.append(r)
        if l >= 0:
            stack.append(l)
    return sorted(result)


def _replace_subtree(
    arr: np.ndarray, old_root: int, target: int,
    new_arr: np.ndarray, new_root: int,
) -> tuple[np.ndarray, int]:
    target_indices = set(_subtree_indices(arr, target))
    survivor_count = len(arr) - len(target_indices)
    out = np.empty(survivor_count + len(new_arr), dtype=NODE_DTYPE)
    remap: dict[int, int] = {}
    dst = 0
    for i in range(len(arr)):
        if i not in target_indices:
            remap[i] = dst
            out[dst] = arr[i]
            dst += 1
    replacement_root_idx = dst + new_root
    new_base = dst
    for i in range(len(new_arr)):
        out[dst] = new_arr[i]
        if int(out[dst]["type"]) == _OP:
            l = int(out[dst]["left"])
            r = int(out[dst]["right"])
            if l >= 0:
                out[dst]["left"] = l + new_base
            if r >= 0:
                out[dst]["right"] = r + new_base
        dst += 1
    for i in range(survivor_count):
        if int(out[i]["type"]) == _OP:
            l = int(out[i]["left"])
            r = int(out[i]["right"])
            if l in target_indices:
                out[i]["left"] = replacement_root_idx
            elif l >= 0 and l in remap:
                out[i]["left"] = remap[l]
            if r in target_indices:
                out[i]["right"] = replacement_root_idx
            elif r >= 0 and r in remap:
                out[i]["right"] = remap[r]

    if old_root in target_indices:
        out_root = replacement_root_idx
    else:
        out_root = remap[old_root]
    return out, out_root


def append_subtree(arr: np.ndarray, root: int, subtree_arr: np.ndarray, subtree_root: int) -> tuple[np.ndarray, int]:
    base = len(arr)
    out = np.empty(len(arr) + len(subtree_arr), dtype=NODE_DTYPE)
    out[:len(arr)] = arr
    for i in range(len(subtree_arr)):
        out[base + i] = subtree_arr[i]
        if int(out[base + i]["type"]) == _OP:
            l = int(out[base + i]["left"])
            r = int(out[base + i]["right"])
            if l >= 0:
                out[base + i]["left"] = l + base
            if r >= 0:
                out[base + i]["right"] = r + base
    return out, root


def flat_random_tree(
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    size: int,
) -> FlatTree:
    from python_backend.search import _random_tree_fixed_size
    node = _random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, size)
    return FlatTree.from_node(node)
