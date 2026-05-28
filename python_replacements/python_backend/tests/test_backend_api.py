from __future__ import annotations

import numpy as np
import pytest

from python_backend.backend_api import BackendAPI, get_backend
from python_backend.options import BackendOptions


def test_get_backend_python():
    backend = get_backend("python")
    assert isinstance(backend, BackendAPI)
    assert hasattr(backend, "equation_search")


def test_get_backend_python_runs():
    backend = get_backend("python")
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = X[:, 0]
    opts = BackendOptions(
        binary_operators=["+"],
        unary_operators=[],
        niterations=1,
        population_size=3,
        maxsize=5,
        maxdepth=3,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=2,
        topn=3,
    )
    result = backend.equation_search(X, y, options=opts, seed=0)
    assert "run_id" in result
    assert "trace_records" in result
    assert "digests" in result


def test_get_backend_julia_raises():
    with pytest.raises(NotImplementedError, match="Julia backend"):
        get_backend("julia")


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("rust")
