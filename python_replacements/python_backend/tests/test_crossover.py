from __future__ import annotations

import numpy as np

from python_backend.eval import check_constraints
from python_backend.ops import resolve_operator_tokens
from python_backend.search import crossover_trees, generate_expression


def test_crossover_is_deterministic():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []

    p1 = generate_expression(rng1, binary_ids, unary_ids, 2)
    p2 = generate_expression(rng1, binary_ids, unary_ids, 2)

    result1 = crossover_trees(p1, p2, rng1, 20, 10)
    result2 = crossover_trees(p1, p2, rng2, 20, 10)

    if result1 is None:
        assert result2 is None
    else:
        c1a, c1b = result1
        c2a, c2b = result2
        assert c1a.structural_hash() == c2a.structural_hash()
        assert c1b.structural_hash() == c2b.structural_hash()


def test_crossover_respects_maxsize_maxdepth():
    rng = np.random.default_rng(99)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    p1 = generate_expression(rng, binary_ids, unary_ids, 2)
    p2 = generate_expression(rng, binary_ids, unary_ids, 2)

    result = crossover_trees(p1, p2, rng, 20, 10)
    if result is not None:
        c1, c2 = result
        ok1, _ = check_constraints(c1, 20, 10)
        ok2, _ = check_constraints(c2, 20, 10)
        assert ok1 and ok2


def test_crossover_returns_none_for_degenerate():
    rng = np.random.default_rng(0)
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []
    p1 = generate_expression(rng, binary_ids, unary_ids, 2)
    p2 = generate_expression(rng, binary_ids, unary_ids, 2)

    result = crossover_trees(p1, p2, rng, 0, 10)
    assert result is None
