"""Tests for seed stability, best/median/worst outcomes, and regression tracking."""

from __future__ import annotations

import json
import os
import pathlib
import time

os.environ["PYSR_BACKEND"] = "python"

import pytest
import numpy as np


def test_seed_stability_10_seeds():
    from pysr import PySRRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] + np.sin(X[:, 2])

    losses = []
    for seed in range(10):
        model = PySRRegressor(
            niterations=2,
            population_size=10,
            tournament_selection_n=5,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            maxsize=10,
            verbosity=0,
            progress=False,
            random_state=seed,
        )
        model.fit(X, y)
        best_loss = model.equations_.loss.min()
        losses.append(best_loss)

    losses_arr = np.array(losses)
    median_loss = float(np.median(losses_arr))
    best_loss = float(np.min(losses_arr))
    worst_loss = float(np.max(losses_arr))
    p95_loss = float(np.percentile(losses_arr, 95))
    std_loss = float(np.std(losses_arr))

    print(
        f"\n[seed-stability] seeds=10 median={median_loss:.4f} best={best_loss:.4f} "
        f"worst={worst_loss:.4f} p95={p95_loss:.4f} std={std_loss:.4f}"
    )

    assert all(np.isfinite(losses_arr)), f"Non-finite losses found: {losses}"
    assert median_loss < 2.0, f"Median loss {median_loss:.4f} >= 2.0"


def test_deterministic_across_seeds():
    from pysr import PySRRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] + np.sin(X[:, 2])

    losses = []
    for _ in range(3):
        model = PySRRegressor(
            niterations=2,
            population_size=10,
            tournament_selection_n=5,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            maxsize=10,
            verbosity=0,
            progress=False,
            random_state=42,
        )
        model.fit(X, y)
        losses.append(model.equations_.loss.min())

    for i in range(1, len(losses)):
        assert abs(losses[i] - losses[0]) < 1e-10, (
            f"Loss mismatch: {losses[0]} vs {losses[i]}"
        )


def test_best_median_worst_outcomes():
    from pysr import PySRRegressor

    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] + np.sin(X[:, 2])

    losses = []
    for seed in range(10):
        model = PySRRegressor(
            niterations=2,
            population_size=10,
            tournament_selection_n=5,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            maxsize=10,
            verbosity=0,
            progress=False,
            random_state=seed,
        )
        model.fit(X, y)
        losses.append(model.equations_.loss.min())

    losses_arr = np.array(losses)
    result = {
        "min_loss": float(np.min(losses_arr)),
        "max_loss": float(np.max(losses_arr)),
        "median_loss": float(np.median(losses_arr)),
        "mean_loss": float(np.mean(losses_arr)),
        "std_loss": float(np.std(losses_arr)),
    }
    print(f"\n[benchmark] {result}")
    assert all(np.isfinite(losses_arr)), f"Non-finite losses found: {losses}"


def test_baseline_regression_against_saved():
    """Compare current benchmark against saved baseline; fail if regressed >20%."""
    from pysr import PySRRegressor

    baseline_path = pathlib.Path(__file__).parent / "baseline_benchmark.json"
    if not baseline_path.exists():
        pytest.skip("baseline_benchmark.json not found")
    with open(baseline_path) as f:
        baseline = json.load(f)

    bench = baseline["quick_bench"]
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] + np.sin(X[:, 2])

    t0 = time.time()
    model = PySRRegressor(
        niterations=bench["niterations"],
        population_size=bench["population_size"],
        tournament_selection_n=5, ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"], unary_operators=[],
        maxsize=10, verbosity=0, progress=False, random_state=42,
    )
    model.fit(X, y)
    elapsed = time.time() - t0

    cur_loss = float(model.equations_.iloc[0]["loss"])
    prev_loss = bench["best_loss"]
    rtol = 0.2
    max_allowed = prev_loss * (1 + rtol)
    assert cur_loss <= max_allowed, (
        f"Loss regressed: current={cur_loss:.4f} vs baseline={prev_loss:.4f} "
        f"(max allowed: {max_allowed:.4f})"
    )

    max_runtime = bench["runtime_seconds"] * (1 + rtol)
    assert elapsed <= max_runtime, (
        f"Runtime regressed: current={elapsed:.3f}s vs baseline={bench['runtime_seconds']:.3f}s "
        f"(max allowed: {max_runtime:.3f}s)"
    )
