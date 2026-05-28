from __future__ import annotations

import numpy as np

from python_backend.eval import compute_loss, evaluate, compute_complexity
from python_backend.ops import resolve_operator_tokens
from python_backend.regularized_evolution import (
    RegularizedEvolutionConfig,
    run_regularized_evolution,
)


def _make_eval(X: np.ndarray, y: np.ndarray, *, maxsize: int, maxdepth: int):
    def _eval(expr):
        complexity = compute_complexity(expr)
        if complexity > maxsize:
            return (float("inf"), complexity, False, "SR-INV-COMPLEXITY-001")
        try:
            y_pred = evaluate(expr, X)
        except Exception:
            return (float("inf"), complexity, False, "SR-INV-EVAL-001")
        loss, valid, reason = compute_loss(y, y_pred)
        if not valid:
            return (float("inf"), complexity, False, reason)
        if maxdepth < 1:
            return (float("inf"), complexity, False, "SR-INV-DEPTH-001")
        return (loss, complexity, True, "")

    return _eval


def test_re_is_deterministic_for_seed():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X[:, 0]

    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []

    cfg = RegularizedEvolutionConfig(
        population_size=10,
        tournament_size=3,
        cycles=50,
        maxsize=10,
        maxdepth=5,
    )
    eval_fn = _make_eval(X, y, maxsize=cfg.maxsize, maxdepth=cfg.maxdepth)
    r1 = run_regularized_evolution(
        rng=rng1,
        binary_ids=binary_ids,
        unary_ids=unary_ids,
        n_features=1,
        config=cfg,
        evaluate=eval_fn,
    )
    r2 = run_regularized_evolution(
        rng=rng2,
        binary_ids=binary_ids,
        unary_ids=unary_ids,
        n_features=1,
        config=cfg,
        evaluate=eval_fn,
    )
    assert r1["best"] == r2["best"]


def test_re_best_is_stable_and_valid():
    rng = np.random.default_rng(1)
    X = np.linspace(-1, 1, 60).reshape(-1, 1)
    y = X[:, 0]
    binary_ids = resolve_operator_tokens(["+", "*", "-"])
    unary_ids: list[str] = []

    cfg = RegularizedEvolutionConfig(
        population_size=8,
        tournament_size=3,
        cycles=30,
        maxsize=10,
        maxdepth=5,
    )
    eval_fn = _make_eval(X, y, maxsize=cfg.maxsize, maxdepth=cfg.maxdepth)
    result = run_regularized_evolution(
        rng=rng,
        binary_ids=binary_ids,
        unary_ids=unary_ids,
        n_features=1,
        config=cfg,
        evaluate=eval_fn,
    )
    best = result["best"]
    assert best["hash"]
    assert best["loss"] is None or best["loss"] >= 0.0

