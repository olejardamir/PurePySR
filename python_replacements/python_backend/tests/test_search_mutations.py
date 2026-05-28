from __future__ import annotations

import numpy as np

from python_backend.eval import check_constraints
from python_backend.ops import resolve_operator_tokens
from python_backend.search import (
    OpNode,
    VarNode,
    ConstNode,
    _break_connection,
    _collect_nodes,
    _form_connection,
    generate_expression,
)


def test_form_connection_is_deterministic():
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(np.random.default_rng(99), binary_ids, unary_ids, 2)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    r1 = _form_connection(tree, rng1, 2, binary_ids, unary_ids)
    r2 = _form_connection(tree, rng2, 2, binary_ids, unary_ids)
    if r1 is None:
        assert r2 is None
    else:
        assert r1.structural_hash() == r2.structural_hash()


def test_form_connection_returns_none_when_no_non_root_opnode():
    rng = np.random.default_rng(0)
    leaf = VarNode(0)
    result = _form_connection(leaf, rng, 2, [], [])
    assert result is None


def test_form_connection_produces_valid_tree():
    rng = np.random.default_rng(1)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(rng, binary_ids, unary_ids, 2)

    result = _form_connection(tree, rng, 2, binary_ids, unary_ids)
    if result is not None:
        ok, _ = check_constraints(result, 20, 10)
        assert ok


def test_form_connection_adds_new_node():
    rng = np.random.default_rng(2)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(rng, binary_ids, unary_ids, 2)

    result = _form_connection(tree, rng, 2, binary_ids, unary_ids)
    if result is not None:
        assert len(_collect_nodes(result)) > len(_collect_nodes(tree))


def test_break_connection_is_deterministic():
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(np.random.default_rng(99), binary_ids, unary_ids, 2)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    r1 = _break_connection(tree, rng1, 2)
    r2 = _break_connection(tree, rng2, 2)
    if r1 is None:
        assert r2 is None
    else:
        assert r1.structural_hash() == r2.structural_hash()


def test_break_connection_returns_none_when_no_opnode():
    rng = np.random.default_rng(0)
    leaf = VarNode(0)
    result = _break_connection(leaf, rng, 2)
    assert result is None
    const = ConstNode(3.14)
    result = _break_connection(const, rng, 2)
    assert result is None


def test_break_connection_produces_valid_tree():
    rng = np.random.default_rng(3)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(rng, binary_ids, unary_ids, 2)

    result = _break_connection(tree, rng, 2)
    if result is not None:
        ok, _ = check_constraints(result, 20, 10)
        assert ok


def test_break_connection_replaces_subtree_with_terminal():
    rng = np.random.default_rng(4)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    tree = generate_expression(rng, binary_ids, unary_ids, 2)

    result = _break_connection(tree, rng, 2)
    if result is not None:
        nodes = _collect_nodes(result)
        assert any(isinstance(n, (VarNode, ConstNode)) for n in nodes)
        assert len(nodes) <= len(_collect_nodes(tree))
