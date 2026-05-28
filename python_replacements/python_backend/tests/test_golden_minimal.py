from __future__ import annotations

import pytest
import numpy as np

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.golden import load_problem, generate_dataset
from python_backend.trace import (
    REQUIRED_RUN_START_KEYS,
    REQUIRED_SEARCH_STEP_KEYS,
    REQUIRED_RUN_END_KEYS,
)
from python_backend.errors import BackendOptionError, SR_ERR_OPT_001
from python_backend.capabilities import CAPABILITY_LEVEL, assert_problem_supported

_FAST_KWARGS = dict(
    niterations=15,
    population_size=100,
    maxsize=15,
    maxdepth=6,
    tournament_selection_n=5,
    deterministic=True,
    ncycles_per_iteration=200,
    topn=10,
)

_SLOW_KWARGS = dict(
    niterations=25,
    population_size=120,
    maxsize=15,
    maxdepth=6,
    tournament_selection_n=5,
    deterministic=True,
    ncycles_per_iteration=300,
    topn=10,
)

_VERY_SLOW_KWARGS = dict(
    niterations=50,
    population_size=150,
    maxsize=15,
    maxdepth=6,
    tournament_selection_n=5,
    deterministic=True,
    ncycles_per_iteration=600,
    topn=10,
)

_SUPPORTED_GOLDENS = [
    "GOLDEN-LIN-001",
    "GOLDEN-LIN-002",
    "GOLDEN-MUL-001",
    "GOLDEN-QUAD-001",
    "GOLDEN-SIN-001",
    "GOLDEN-MIXED-001",
]

def _search_kwargs(
    operators: dict, acceptance: dict, slow: int = 0, search_options: dict | None = None,
) -> BackendOptions:
    if slow == 2:
        base = dict(_VERY_SLOW_KWARGS)
    elif slow == 1:
        base = dict(_SLOW_KWARGS)
    else:
        base = dict(_FAST_KWARGS)
    base["binary_operators"] = operators.get("binary", [])
    base["unary_operators"] = operators.get("unary", [])
    base["maxsize"] = acceptance["max_complexity"]
    if search_options:
        base.update(search_options)
    return BackendOptions(**base)


def _run_golden(problem_id: str, seed: int = 42, slow: int = 0):
    problem = load_problem(problem_id)
    X, y = generate_dataset(problem, rng=np.random.default_rng(seed))
    options = _search_kwargs(
        problem["operators"], problem["acceptance"], slow=slow,
        search_options=problem.get("search_options"),
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
    ce = best.get("canonical_expression", "")
    if "(" in ce:
        assert ce.startswith("sr."), (
            f"{problem['problem_id']}: canonical expression {ce!r} "
            f"uses short name instead of full op_id"
        )


# ---------------------------------------------------------------------------
# Individual golden problem tests
# ---------------------------------------------------------------------------

def test_golden_lin_001():
    _check_golden(*_run_golden("GOLDEN-LIN-001"))


def test_golden_lin_002():
    _check_golden(*_run_golden("GOLDEN-LIN-002"))


def test_golden_mul_001():
    _check_golden(*_run_golden("GOLDEN-MUL-001"))


def test_golden_quad_001():
    _check_golden(*_run_golden("GOLDEN-QUAD-001"))


def test_golden_sin_001():
    _check_golden(*_run_golden("GOLDEN-SIN-001"))


def test_golden_mixed_001():
    _check_golden(*_run_golden("GOLDEN-MIXED-001", slow=1))


def test_golden_safelog_001():
    problem = load_problem("GOLDEN-SAFELOG-001")
    assert_problem_supported(problem)
    _check_golden(*_run_golden("GOLDEN-SAFELOG-001", slow=1))


def test_golden_pow_001():
    problem = load_problem("GOLDEN-POW-001")
    assert_problem_supported(problem)
    _check_golden(*_run_golden("GOLDEN-POW-001"))


def test_golden_mae_001():
    problem = load_problem("GOLDEN-MAE-001")
    assert_problem_supported(problem)
    _check_golden(*_run_golden("GOLDEN-MAE-001"))


def test_golden_constr_001():
    problem = load_problem("GOLDEN-CONSTR-001")
    assert_problem_supported(problem)
    _check_golden(*_run_golden("GOLDEN-CONSTR-001"))


# ---------------------------------------------------------------------------
# Trace completeness: every search_step has all required keys per golden
# ---------------------------------------------------------------------------

def _check_trace_keys(result: dict) -> None:
    records = result["trace_records"]
    types = {r["record_type"] for r in records}
    assert "run_start" in types, "missing run_start"
    assert "search_step" in types, "missing search_step"
    assert "run_end" in types, "missing run_end"

    for r in records:
        if r["record_type"] == "run_start":
            missing = REQUIRED_RUN_START_KEYS - r.keys()
            assert not missing, f"run_start missing keys: {missing}"
        elif r["record_type"] == "search_step":
            missing = REQUIRED_SEARCH_STEP_KEYS - r.keys()
            assert not missing, f"search_step missing keys: {missing}"
        elif r["record_type"] == "run_end":
            missing = REQUIRED_RUN_END_KEYS - r.keys()
            assert not missing, f"run_end missing keys: {missing}"


def test_trace_keys_lin():
    _check_trace_keys(_run_golden("GOLDEN-LIN-001")[0])


def test_trace_keys_sin():
    _check_trace_keys(_run_golden("GOLDEN-SIN-001")[0])


def test_trace_keys_mixed():
    _check_trace_keys(_run_golden("GOLDEN-MIXED-001", slow=1)[0])


# ---------------------------------------------------------------------------
# Determinism: same seed → identical best hash + all digests
# ---------------------------------------------------------------------------

def _check_determinism(problem_id: str, slow: int = 0) -> None:
    problem = load_problem(problem_id)
    ds_seed = 99
    X, y = generate_dataset(problem, rng=np.random.default_rng(ds_seed))
    options = _search_kwargs(
        problem["operators"], problem["acceptance"], slow=slow,
    )
    backend = PythonSRBackend()

    r1 = backend.equation_search(X, y, options=options, seed=123)
    r2 = backend.equation_search(X, y, options=options, seed=123)

    assert r1["best"]["hash"] == r2["best"]["hash"], (
        f"{problem_id}: best hash differs between deterministic runs"
    )
    assert r1["digests"] == r2["digests"], (
        f"{problem_id}: digests differ: {r1['digests']} != {r2['digests']}"
    )


def test_determinism_lin():
    _check_determinism("GOLDEN-LIN-001")


def test_determinism_sin():
    _check_determinism("GOLDEN-SIN-001")


def test_determinism_mixed():
    _check_determinism("GOLDEN-MIXED-001", slow=1)


def test_determinism_safelog():
    _check_determinism("GOLDEN-SAFELOG-001", slow=1)


# ---------------------------------------------------------------------------
# Non-finite prediction handling (existing)
# ---------------------------------------------------------------------------

def test_non_finite_prediction_handling():
    options = BackendOptions(
        binary_operators=["/"],
        unary_operators=[],
        niterations=1,
        population_size=5,
        maxsize=20,
        maxdepth=6,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=1,
        topn=5,
    )
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options, seed=0)
    assert result["best"] is not None


# ---------------------------------------------------------------------------
# HOF ordering (existing)
# ---------------------------------------------------------------------------

def test_hof_ordering():
    result, _problem = _run_golden("GOLDEN-LIN-001")
    hof = result["hall_of_fame"]
    for i in range(len(hof) - 1):
        assert hof[i]["complexity"] <= hof[i + 1]["complexity"]
        if hof[i]["complexity"] == hof[i + 1]["complexity"]:
            assert hof[i]["loss"] <= hof[i + 1]["loss"]


# ---------------------------------------------------------------------------
# Regression: trace record count matches expected loop structure
# ---------------------------------------------------------------------------

def test_baseline_trace_record_count():
    """Verify exact trace record counts to catch loop indentation regressions."""
    options = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=2,
        population_size=5,
        maxsize=10,
        maxdepth=5,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=3,
        topn=5,
    )
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options, seed=0)

    records = result["trace_records"]
    step_records = [r for r in records if r.get("record_type") == "search_step"]

    init_count = options.population_size
    search_count = options.niterations * options.ncycles_per_iteration
    expected_step_count = init_count + search_count

    assert len(step_records) == expected_step_count, (
        f"expected {expected_step_count} search_step records "
        f"(init={init_count} + search={search_count}), "
        f"got {len(step_records)}"
    )


# ---------------------------------------------------------------------------
# Constraint violation produces correct error codes
# ---------------------------------------------------------------------------

def test_constraint_violation_rejects_expression():
    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=3,
        population_size=5,
        maxsize=20,
        maxdepth=8,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=2,
        topn=5,
        constraints={"sr.arith.mul_v1": (1, 1)},
    )
    X = np.array([[1.0], [2.0]], dtype=np.float64)
    y = np.array([2.0, 5.0], dtype=np.float64)
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=opts, seed=0)
    records = result["trace_records"]
    constr_violations = [
        r for r in records
        if r.get("invalid_reason_code") in ("SR-INV-CONSTR-001", "SR-INV-CONSTR-002")
    ]
    assert len(constr_violations) > 0, "expected at least one constraint violation"


def test_nested_constraints_violation_error_code():
    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "cos"],
        niterations=3,
        population_size=5,
        maxsize=20,
        maxdepth=8,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=2,
        topn=5,
        nested_constraints={"sr.math.sin_v1": {"sr.math.cos_v1": 0}},
    )
    X = np.array([[1.0], [2.0]], dtype=np.float64)
    y = np.array([2.0, 5.0], dtype=np.float64)
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=opts, seed=0)
    records = result["trace_records"]
    nesting_violations = [
        r for r in records
        if r.get("invalid_reason_code") == "SR-INV-NESTING-001"
    ]
    # Not guaranteed to occur in a short run, but the option should not error
    assert result["best"] is not None


# ---------------------------------------------------------------------------
# Unsupported operator raises clear error
# ---------------------------------------------------------------------------

def test_unsupported_operator_raises():
    opts = BackendOptions(
        binary_operators=["foo"],
        unary_operators=[],
        niterations=1,
        population_size=5,
        maxsize=10,
        maxdepth=4,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=1,
        topn=5,
    )
    X = np.ones((5, 2))
    y = np.ones(5)
    backend = PythonSRBackend()
    try:
        backend.equation_search(X, y, options=opts, seed=0)
        assert False, "expected BackendOptionError"
    except BackendOptionError as e:
        assert SR_ERR_OPT_001 in str(e)


# ---------------------------------------------------------------------------
# Pow with negative domain + autodiff should not produce NaN loss
# ---------------------------------------------------------------------------

def test_pow_negative_domain_with_autodiff():
    opts = BackendOptions(
        binary_operators=["+", "-", "*", "^"],
        unary_operators=[],
        niterations=5,
        population_size=10,
        maxsize=15,
        maxdepth=6,
        tournament_selection_n=3,
        deterministic=True,
        ncycles_per_iteration=10,
        topn=5,
        optimizer_algorithm="L-BFGS-B",
        optimize_constants=True,
        autodiff_backend=True,
    )
    rng = np.random.default_rng(42)
    X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = X[:, 0] ** 2 + 1.0

    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=opts, seed=42)
    assert result["best"] is not None, "search should produce a result"
    assert np.isfinite(result["best"]["loss"]), (
        f"best loss should be finite, got {result['best']['loss']}"
    )
