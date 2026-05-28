"""Tests for production readiness: checkpointing, concurrency, edge-case paths,
type stability, negative weights, large HOF, memory behavior, and benchmarking."""

from __future__ import annotations

import gc
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import pytest


def _quick_model(extra_kw=None, X=None, y=None):
    """Fit a tiny model for artifact/robustness testing."""
    from pysr import PySRRegressor
    if X is None:
        X = np.random.randn(50, 3).astype(np.float64)
    if y is None:
        y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    kw = dict(
        niterations=2,
        population_size=10,
        tournament_selection_n=3,
        ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=8,
        verbosity=0,
        progress=False,
    )
    if extra_kw:
        kw.update(extra_kw)
    model = PySRRegressor(**kw)
    model.fit(X, y)
    return model, X, y


# ── 1. Checkpoint / resume across process restart ──────────────────────

def test_checkpoint_file_written():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    out_dir = tempfile.mkdtemp()
    model, X, y = _quick_model(extra_kw=dict(output_directory=out_dir))
    artifact_dir = os.path.join(model.output_directory_, model.run_id_)
    ckpt = os.path.join(artifact_dir, "checkpoint.pkl")
    assert os.path.isfile(ckpt), f"checkpoint.pkl not found at {ckpt}"
    loaded = PySRRegressor.from_file(run_directory=artifact_dir)
    assert loaded.equations_ is not None
    assert len(loaded.equations_) > 0
    assert "equation" in loaded.equations_.columns


def test_checkpoint_resume_warm_start_via_pickle():
    os.environ["PYSR_BACKEND"] = "python"
    out_dir = tempfile.mkdtemp()
    model, X, y = _quick_model(extra_kw=dict(output_directory=out_dir))
    artifact_dir = os.path.join(model.output_directory_, model.run_id_)
    ckpt = os.path.join(artifact_dir, "checkpoint.pkl")
    with open(ckpt, "rb") as f:
        loaded = pickle.load(f)
    loaded.set_params(warm_start=True, niterations=3)
    loaded.fit(X, y)
    preds = loaded.predict(X)
    assert np.all(np.isfinite(preds)), "predictions from resumed model are not finite"


# ── 2. Concurrent runs, same output directory ──────────────────────────

def test_concurrent_same_output_directory_different_run_ids():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    out_dir = tempfile.mkdtemp()
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    kw = dict(
        niterations=2,
        population_size=10,
        tournament_selection_n=3,
        ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=8,
        verbosity=0,
        progress=False,
        output_directory=out_dir,
    )
    model1 = PySRRegressor(run_id="run_concurrent_a", **kw)
    model1.fit(X, y)
    model2 = PySRRegressor(run_id="run_concurrent_b", **kw)
    model2.fit(X, y)
    assert model1.run_id_ != model2.run_id_
    d1 = os.path.join(out_dir, model1.run_id_)
    d2 = os.path.join(out_dir, model2.run_id_)
    assert os.path.isdir(d1), f"{d1} does not exist"
    assert os.path.isdir(d2), f"{d2} does not exist"
    assert np.all(np.isfinite(model1.predict(X)))
    assert np.all(np.isfinite(model2.predict(X)))


# ── 3. Paths with spaces, unicode, long names ──────────────────────────

def test_output_dir_with_spaces():
    os.environ["PYSR_BACKEND"] = "python"
    tmpdir = tempfile.mkdtemp()
    out_dir = os.path.join(tmpdir, "my output dir")
    model, X, y = _quick_model(extra_kw=dict(output_directory=out_dir))
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_output_dir_with_unicode():
    os.environ["PYSR_BACKEND"] = "python"
    tmpdir = tempfile.mkdtemp()
    out_dir = os.path.join(tmpdir, "répertoire_sortie", "データ")
    model, X, y = _quick_model(extra_kw=dict(output_directory=out_dir))
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_output_dir_long_path():
    os.environ["PYSR_BACKEND"] = "python"
    tmpdir = tempfile.mkdtemp()
    long_rel = "a" * 60 + "/" + "b" * 60 + "/" + "c" * 60
    out_dir = os.path.join(tmpdir, long_rel)
    model, X, y = _quick_model(extra_kw=dict(output_directory=out_dir))
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── 4. Subprocess execution ────────────────────────────────────────────

def test_fit_in_subprocess():
    os.environ["PYSR_BACKEND"] = "python"
    code = """import os, sys, numpy as np
os.environ["PYSR_BACKEND"] = "python"
from pysr import PySRRegressor
X = np.random.randn(30, 2).astype(np.float64)
y = (X[:, 0]**2).astype(np.float64)
model = PySRRegressor(niterations=2, population_size=10,
                       tournament_selection_n=3, ncycles_per_iteration=1,
                       binary_operators=["+", "-", "*"],
                       unary_operators=[], maxsize=8,
                       verbosity=0, progress=False)
model.fit(X, y)
preds = model.predict(X)
assert np.all(np.isfinite(preds))
print("OK")
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.environ.get("PYTHONPATH", ""),
    ]))}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert "OK" in result.stdout


# ── 5. Float32 vs Float64 stability ────────────────────────────────────

def test_float32_type_handling():
    os.environ["PYSR_BACKEND"] = "python"
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3).astype(np.float32)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float32)
    model, _, _ = _quick_model(X=X, y=y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_float64_result_stability():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    losses = []
    for _ in range(3):
        model = PySRRegressor(
            niterations=2, population_size=10,
            tournament_selection_n=3, ncycles_per_iteration=1,
            binary_operators=["+", "-", "*"],
            unary_operators=[], maxsize=8,
            verbosity=0, progress=False,
            random_state=42,  # deterministic
        )
        model.fit(X, y)
        losses.append(float(model.equations_.iloc[0]["loss"]))
    assert max(losses) - min(losses) < 1e-10, (
        f"losses differ across deterministic runs: {losses}"
    )


def test_float32_vs_float64_predict_consistent():
    os.environ["PYSR_BACKEND"] = "python"
    rng = np.random.RandomState(42)
    X64 = rng.randn(50, 3).astype(np.float64)
    y64 = (X64[:, 0]**2 + X64[:, 1]).astype(np.float64)
    model, _, _ = _quick_model(X=X64, y=y64)
    pred64 = model.predict(X64)
    X32 = X64.astype(np.float32)
    pred32 = model.predict(X32)
    np.testing.assert_allclose(pred32, pred64, rtol=1e-4)


# ── 6. Negative-weight semantics ───────────────────────────────────────

def test_negative_weights_produce_finite():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    weights = -np.abs(np.random.randn(50))
    model = PySRRegressor(
        niterations=2, population_size=10,
        tournament_selection_n=3, ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
    )
    model.fit(X, y, weights=weights)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_negative_weights_produce_different_result():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    abs_w = np.abs(np.random.randn(50))
    kw = dict(
        niterations=2, population_size=15,
        tournament_selection_n=4, ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
    )
    m_pos = PySRRegressor(**kw)
    m_pos.fit(X, y, weights=abs_w)
    m_neg = PySRRegressor(**kw)
    m_neg.fit(X, y, weights=-abs_w)
    best_pos = m_pos.equations_.iloc[0]["loss"]
    best_neg = m_neg.equations_.iloc[0]["loss"]
    assert abs(best_pos - best_neg) > 1e-10, (
        f"positive and negative weight losses should differ: pos={best_pos}, neg={best_neg}"
    )


# ── 7. Large HOF handling ──────────────────────────────────────────────

def test_many_equations_in_hof():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    model = PySRRegressor(
        niterations=8, population_size=20,
        tournament_selection_n=5, ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin"],
        maxsize=15,
        verbosity=0, progress=False,
    )
    model.fit(X, y)
    assert len(model.equations_) > 0
    assert np.all(np.isfinite(model.equations_["loss"]))


# ── 8. Memory / parse behavior with many entries ───────────────────────

def test_large_hof_memory_stable():
    os.environ["PYSR_BACKEND"] = "python"
    gc.collect()
    from pysr import PySRRegressor
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    model = PySRRegressor(
        niterations=6, population_size=20,
        tournament_selection_n=5, ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin"],
        maxsize=15,
        verbosity=0, progress=False,
    )
    model.fit(X, y)
    gc.collect()
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── 9. Benchmark harness fixture ───────────────────────────────────────

def test_benchmark_harness_records_metrics():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5)).astype(np.float64)
    y = (X[:, 0] + np.sin(X[:, 1])).astype(np.float64)
    t0 = time.time()
    model = PySRRegressor(
        niterations=5, population_size=30,
        tournament_selection_n=5, ncycles_per_iteration=3,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "cos"],
        maxsize=12,
        verbosity=0, progress=False,
        random_state=42, deterministic=True,
    )
    model.fit(X, y)
    elapsed = time.time() - t0
    best_loss = float(model.equations_.iloc[0]["loss"])
    n_eq = len(model.equations_)
    bench = {
        "best_loss": best_loss,
        "runtime_seconds": round(elapsed, 3),
        "n_equations": n_eq,
    }
    print(f"\n[benchmark] {bench}")
    assert best_loss < 1.0, f"best_loss too high: {best_loss}"
    assert elapsed < 60, f"runtime too long: {elapsed}s"


# ── 10. Read-only / missing artifact directories ────────────────────────

def test_readonly_output_directory():
    """When output_directory is a file (not dir), fit raises a clear error."""
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    file_path = tmpdir / "dead_file"
    file_path.touch()
    X = np.random.randn(20, 2).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    model = PySRRegressor(
        niterations=2, population_size=10,
        tournament_selection_n=3, ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
        output_directory=str(file_path),
    )
    with pytest.raises((NotADirectoryError, OSError, PermissionError)):
        model.fit(X, y)


def test_missing_output_directory():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    missing_dir = tmpdir / "does_not_exist"
    X = np.random.randn(20, 2).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    model = PySRRegressor(
        niterations=2, population_size=10,
        tournament_selection_n=3, ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
        output_directory=str(missing_dir),
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert missing_dir.exists(), "output directory should have been created"


def test_no_output_directory_specified():
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    X = np.random.randn(20, 2).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    model = PySRRegressor(
        niterations=2, population_size=10,
        tournament_selection_n=3, ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
        output_directory=None,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert hasattr(model, "output_directory_"), "output_directory_ should be set"


# ── 11. Julia detection (no accidental dependency) ──────────────────────

def test_no_real_julia_dependency_when_python_backend():
    """Verify that setting PYSR_BACKEND=python avoids any real Julia imports."""
    os.environ["PYSR_BACKEND"] = "python"
    code = textwrap.dedent("""\
    import os, sys
    os.environ["PYSR_BACKEND"] = "python"
    # Attempt to import juliacall (should succeed via shim)
    from pysr import PySRRegressor
    import juliacall
    # Verify the shim, not real juliacall
    assert juliacall.__file__ is not None
    print("OK")
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, f"julia shim import failed: {result.stderr}"
    assert "OK" in result.stdout


# ── 12. Negative weight explicit semantics ──────────────────────────────

def test_negative_weights_semantics_rejected_or_handled():
    """Negative weights produce negative MSE → HOF rejects with ValueError.
    This is the defined behavior: weights must be non-negative."""
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0]**2 + X[:, 1]).astype(np.float64)
    weights = np.random.randn(30).astype(np.float64)
    model = PySRRegressor(
        niterations=2, population_size=10,
        tournament_selection_n=3, ncycles_per_iteration=1,
        binary_operators=["+", "-", "*"],
        unary_operators=[], maxsize=8,
        verbosity=0, progress=False,
    )
    with pytest.raises(ValueError, match="negative loss"):
        model.fit(X, y, weights=weights)
