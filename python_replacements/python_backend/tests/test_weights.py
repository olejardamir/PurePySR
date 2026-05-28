from __future__ import annotations

import numpy as np
import pytest

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.eval import compute_loss


def test_weights_affect_loss():
    y_true = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 10.0, 1.0, 1.0, 5.0])

    unweighted_loss, _, _ = compute_loss(y_true, y_pred)
    weighted_loss, _, _ = compute_loss(
        y_true, y_pred,
        weights=np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert weighted_loss != unweighted_loss, "weights should change the loss"
    expected = np.average(
        (y_pred - y_true) ** 2,
        weights=np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert abs(weighted_loss - expected) < 1e-12, (
        f"weighted loss {weighted_loss} != expected {expected}"
    )


def test_zero_weight_samples_dont_affect_search():
    """With half the samples weighted 0, search should still find a solution."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (100, 2))
    y = X[:, 0] + X[:, 1]
    weights = np.ones(100)
    weights[50:] = 0.0  # ignore second half

    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=3,
        population_size=20,
        maxsize=15,
        maxdepth=6,
        tournament_selection_n=3,
        deterministic=True,
        ncycles_per_iteration=10,
        topn=5,
    )
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=opts, seed=0, weights=weights)
    assert result["best"] is not None, "search should produce a result with weights"


def test_weights_in_extra_options():
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (50, 1))
    y = X[:, 0] ** 2 + 1.0
    weights = np.ones(50)

    opts = BackendOptions(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=2,
        population_size=10,
        maxsize=10,
        maxdepth=5,
        tournament_selection_n=2,
        deterministic=True,
        ncycles_per_iteration=5,
        topn=5,
    )
    backend = PythonSRBackend()
    result = backend.equation_search(
        X, y, options=opts, seed=0, extra_options={"weights": weights},
    )
    assert result["best"] is not None


def test_unweighed_loss_matches_mse():
    """Without weights, compute_loss should match np.mean((y - y_pred)^2)."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(-1, 1, 20)
    y_pred = rng.uniform(-1, 1, 20)
    loss, valid, _ = compute_loss(y_true, y_pred)
    expected = float(np.mean((y_true - y_pred) ** 2))
    assert abs(loss - expected) < 1e-12
    assert valid
