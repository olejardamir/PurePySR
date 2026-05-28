from __future__ import annotations

import numpy as np

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.trace import REQUIRED_SEARCH_STEP_KEYS


def _run_lin001() -> dict:
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X[:, 0]
    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=3,
        population_size=10,
        maxsize=10,
        maxdepth=4,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=10,
        topn=5,
    )
    backend = PythonSRBackend()
    return backend.equation_search(X, y, options=opts, seed=0)


def _steps(result: dict) -> list[dict]:
    return [r for r in result["trace_records"] if r["record_type"] == "search_step"]


def test_no_required_key_missing():
    result = _run_lin001()
    steps = _steps(result)
    assert len(steps) > 0
    for i, rec in enumerate(steps):
        missing = REQUIRED_SEARCH_STEP_KEYS - rec.keys()
        assert not missing, f"search_step #{i} missing keys: {missing}"


def test_missing_fields_use_none_not_empty_string():
    result = _run_lin001()
    steps = _steps(result)
    for i, rec in enumerate(steps):
        if rec["validity_status"] == "invalid":
            assert rec["loss"] is None, (
                f"step #{i} invalid but loss={rec['loss']!r}"
            )
            assert isinstance(rec["invalid_reason_code"], str), (
                f"step #{i} invalid but invalid_reason_code={rec['invalid_reason_code']!r}"
            )


def test_mutation_steps_have_candidate_hash_before():
    result = _run_lin001()
    steps = _steps(result)
    mutation_steps = [
        s for s in steps if s["mutation_or_crossover_type"] == "mutation"
    ]
    assert len(mutation_steps) > 0, "no mutation steps found"
    for i, rec in enumerate(mutation_steps):
        assert rec["candidate_hash_before"] is not None, (
            f"mutation step #{i} has None candidate_hash_before"
        )


def test_initial_steps_have_none_candidate_hash_before():
    result = _run_lin001()
    steps = _steps(result)
    initial_steps = [
        s for s in steps if s["mutation_or_crossover_type"] == "initial"
    ]
    assert len(initial_steps) > 0, "no initial steps found"
    for i, rec in enumerate(initial_steps):
        assert rec["candidate_hash_before"] is None, (
            f"initial step #{i} has non-None candidate_hash_before"
        )


def test_loss_is_either_none_or_numeric_string():
    result = _run_lin001()
    steps = _steps(result)
    for i, rec in enumerate(steps):
        val = rec["loss"]
        if val is not None:
            assert isinstance(val, str), (
                f"step #{i} loss is {type(val).__name__}, not str"
            )
            float(val)


def test_best_loss_after_step_is_none_or_numeric_string():
    result = _run_lin001()
    steps = _steps(result)
    for i, rec in enumerate(steps):
        val = rec["best_loss_after_step"]
        if val is not None:
            assert isinstance(val, str), (
                f"step #{i} best_loss_after_step is {type(val).__name__}, not str"
            )
            float(val)


def test_best_complexity_after_step_is_none_or_int():
    result = _run_lin001()
    steps = _steps(result)
    for i, rec in enumerate(steps):
        val = rec["best_complexity_after_step"]
        if val is not None:
            assert isinstance(val, int) and val >= 1, (
                f"step #{i} best_complexity_after_step={val!r} not int>=1"
            )


def test_step_trace_digest_identical_for_same_seed():
    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=2,
        population_size=10,
        maxsize=5,
        maxdepth=4,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=5,
        topn=5,
    )
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X[:, 0]
    backend = PythonSRBackend()
    r1 = backend.equation_search(X, y, options=opts, seed=42)
    r2 = backend.equation_search(X, y, options=opts, seed=42)
    assert r1["digests"]["step_trace_digest"] == r2["digests"]["step_trace_digest"]
