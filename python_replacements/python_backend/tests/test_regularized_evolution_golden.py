from __future__ import annotations

import numpy as np

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.golden import load_problem, generate_dataset


def _run_re_golden(problem_id: str, seed: int = 42) -> tuple[dict, dict]:
    problem = load_problem(problem_id)
    X, y = generate_dataset(problem, rng=np.random.default_rng(seed))
    options = BackendOptions(
        binary_operators=problem["operators"].get("binary", []),
        unary_operators=problem["operators"].get("unary", []),
        maxsize=problem["acceptance"]["max_complexity"],
        search_algorithm="regularized_evolution",
        population_size=150,
        niterations=30,
        ncycles_per_iteration=400,
        tournament_selection_n=15,
        maxdepth=6,
        deterministic=True,
        topn=10,
    )
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options, seed=seed)
    return result, problem


def _check_golden(result: dict, problem: dict) -> None:
    best = result["best"]
    assert best is not None, f"{problem['problem_id']}: no best expression found"
    assert best["loss"] <= problem["acceptance"]["max_loss"], (
        f"{problem['problem_id']}: loss {best['loss']} > "
        f"{problem['acceptance']['max_loss']}"
    )
    assert best["complexity"] <= problem["acceptance"]["max_complexity"], (
        f"{problem['problem_id']}: complexity {best['complexity']} > "
        f"{problem['acceptance']['max_complexity']}"
    )


def test_re_golden_lin_002():
    result, problem = _run_re_golden("GOLDEN-LIN-002", seed=1)
    _check_golden(result, problem)


def test_re_golden_mul_001():
    result, problem = _run_re_golden("GOLDEN-MUL-001", seed=2)
    _check_golden(result, problem)


def test_re_golden_quad_001_with_constant_optimization():
    problem = load_problem("GOLDEN-QUAD-001")
    X, y = generate_dataset(problem, rng=np.random.default_rng(42))
    options = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=9,
        search_algorithm="regularized_evolution",
        optimize_constants=True,
        population_size=150,
        niterations=30,
        ncycles_per_iteration=400,
        tournament_selection_n=15,
        maxdepth=6,
        deterministic=True,
        topn=10,
    )
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options, seed=42)
    best = result["best"]
    assert best is not None
    assert best["loss"] <= 1e-4
    assert best["complexity"] <= 9
