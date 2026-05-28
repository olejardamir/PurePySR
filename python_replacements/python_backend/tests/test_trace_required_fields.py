from __future__ import annotations

import numpy as np

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.trace import (
    REQUIRED_RUN_START_KEYS,
    REQUIRED_SEARCH_STEP_KEYS,
    REQUIRED_RUN_END_KEYS,
)


def _minimal_run() -> dict:
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X[:, 0] ** 2
    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=2,
        population_size=10,
        maxsize=10,
        maxdepth=4,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=5,
        topn=5,
    )
    backend = PythonSRBackend()
    return backend.equation_search(X, y, options=opts, seed=0)


def test_run_start_has_all_required_keys():
    result = _minimal_run()
    records = result["trace_records"]
    starts = [r for r in records if r["record_type"] == "run_start"]
    assert len(starts) == 1, "expected exactly one run_start"
    missing = REQUIRED_RUN_START_KEYS - starts[0].keys()
    assert not missing, f"run_start missing keys: {missing}"


def test_search_step_records_have_all_required_keys():
    result = _minimal_run()
    records = result["trace_records"]
    steps = [r for r in records if r["record_type"] == "search_step"]
    assert len(steps) > 0, "expected at least one search_step"
    for i, rec in enumerate(steps):
        missing = REQUIRED_SEARCH_STEP_KEYS - rec.keys()
        assert not missing, f"search_step #{i} missing keys: {missing}"


def test_run_end_has_all_required_keys():
    result = _minimal_run()
    records = result["trace_records"]
    ends = [r for r in records if r["record_type"] == "run_end"]
    assert len(ends) == 1, "expected exactly one run_end"
    missing = REQUIRED_RUN_END_KEYS - ends[0].keys()
    assert not missing, f"run_end missing keys: {missing}"


def test_all_three_record_types_present():
    result = _minimal_run()
    records = result["trace_records"]
    types = {r["record_type"] for r in records}
    assert "run_start" in types
    assert "search_step" in types
    assert "run_end" in types
