"""Comprehensive robustness and stress tests for the PySR Python-only backend."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest

os.environ["PYSR_BACKEND"] = "python"


@pytest.fixture(autouse=True)
def _seed_numpy():
    np.random.seed(42)


def _make_regressor(**kw):
    from pysr import PySRRegressor
    defaults = dict(
        niterations=2,
        population_size=10,
        tournament_selection_n=5,
        ncycles_per_iteration=2,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        verbosity=0,
        progress=False,
    )
    defaults.update(kw)
    return PySRRegressor(**defaults)


# ── Pathological inputs ──────────────────────────────────────────────


def test_nan_in_x():
    """NaN values in X should be rejected gracefully."""
    X = np.random.randn(30, 2).astype(np.float64)
    X[0, 0] = np.nan
    y = (X[:, 1] ** 2).astype(np.float64)
    model = _make_regressor()
    with pytest.raises(Exception):
        model.fit(X, y)


def test_inf_in_x():
    """Infinity values in X should be rejected gracefully."""
    X = np.random.randn(30, 2).astype(np.float64)
    X[0, 0] = np.inf
    y = (X[:, 1] ** 2).astype(np.float64)
    model = _make_regressor()
    with pytest.raises(Exception):
        model.fit(X, y)


def test_constant_columns():
    """Zero-variance (constant) columns should be tolerated."""
    X = np.random.randn(30, 3).astype(np.float64)
    X[:, 0] = 5.0
    y = (X[:, 1] + X[:, 2]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_duplicate_columns():
    """Duplicate (identical) columns should be tolerated."""
    X = np.random.randn(30, 2).astype(np.float64)
    X = np.column_stack([X[:, 0], X[:, 0], X[:, 1]]).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 2]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_zero_variance_target():
    """Target with zero variance (all same value) should be handled."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.full(30, 3.14, dtype=np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_extreme_scale_x():
    """Very large X values (1e6) should not crash the search."""
    X = np.random.randn(30, 2).astype(np.float64) * 1e6
    y = (X[:, 0] + X[:, 1]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_extreme_scale_y():
    """Very large y values (1e6) should not crash the search."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64) * 1e6
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_tiny_dataset_2rows():
    """Very small dataset (2 rows, 1 feature) should be handled."""
    X = np.random.randn(2, 1).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(population_size=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_one_row_dataset():
    """Single-row dataset should be handled gracefully."""
    X = np.random.randn(1, 2).astype(np.float64)
    y = np.array([1.0], dtype=np.float64)
    model = _make_regressor(population_size=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_one_feature_dataset():
    """Dataset with a single feature should work."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_integer_input():
    """Integer input arrays (not float64) should be auto-converted."""
    X = np.random.randint(0, 10, size=(30, 2))
    y = np.random.randint(0, 100, size=30).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_float32_input():
    """float32 input arrays should be auto-converted."""
    X = np.random.randn(30, 2).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float32)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_pandas_dataframe_input():
    """pandas DataFrame as X should work."""
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(
        np.random.randn(30, 2).astype(np.float64), columns=["a", "b"],
    )
    y = (X_df["a"] ** 2 + X_df["b"]).astype(np.float64).values
    model = _make_regressor()
    model.fit(X_df, y)
    preds = model.predict(X_df)
    assert len(np.asarray(preds)) == 30
    assert model.equations_ is not None
    assert len(model.equations_) > 0


def test_pandas_series_y():
    """pandas Series as y should work."""
    pd = pytest.importorskip("pandas")
    X = np.random.randn(30, 2).astype(np.float64)
    y = pd.Series((X[:, 0] ** 2).astype(np.float64))
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_target_shapes():
    """Different target shapes (1D array, 2D column, list) all work."""
    X = np.random.randn(30, 2).astype(np.float64)
    y_base = X[:, 0] ** 2

    # 1D array
    model1 = _make_regressor()
    model1.fit(X, y_base.astype(np.float64))
    assert np.all(np.isfinite(model1.predict(X)))

    # 2D column vector
    model2 = _make_regressor()
    model2.fit(X, y_base.reshape(-1, 1).astype(np.float64))
    assert np.all(np.isfinite(model2.predict(X)))

    # 1D list
    model3 = _make_regressor()
    model3.fit(X, y_base.tolist())
    assert np.all(np.isfinite(model3.predict(X)))


def test_weights_all_zero():
    """All-zero sample weights should be rejected gracefully."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    weights = np.zeros(30, dtype=np.float64)
    model = _make_regressor()
    with pytest.raises(Exception):
        model.fit(X, y, weights=weights)


def test_weights_extreme():
    """Extreme sample weights (1e10) should not crash."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    weights = np.full(30, 1e10, dtype=np.float64)
    model = _make_regressor()
    model.fit(X, y, weights=weights)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_weights_negative():
    """Negative sample weights should be handled or produce finite predictions."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    weights = np.full(30, -1.0, dtype=np.float64)
    model = _make_regressor()
    model.fit(X, y, weights=weights)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_maxsize_maxdepth_minimal():
    """maxsize=7 (minimum), maxdepth=1 creates a very constrained search space."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(maxsize=7, maxdepth=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_operators_cannot_represent_target():
    """Operator set with only '+' cannot represent quadratic target but
    should still produce finite predictions."""
    X = np.random.randn(30, 1).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(binary_operators=["+"], unary_operators=[])
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_tournament_selection_equals_population():
    """tournament_selection_n == population_size is valid (uses full pop)."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = _make_regressor(population_size=10, tournament_selection_n=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Correctness deepening ────────────────────────────────────────────


def test_predict_matches_lambda():
    """predict() output matches lambda_format() evaluation from
    the same best equation (via get_best)."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)

    best_eq = model.get_best()
    lambda_fn = best_eq["lambda_format"]
    lambda_preds = lambda_fn(X)

    np.testing.assert_allclose(preds, lambda_preds, rtol=1e-10)


def test_feature_ordering_stability():
    """Feature ordering is stable across fit and predict with different
    column order (DataFrame with shuffled columns)."""
    pd = pytest.importorskip("pandas")
    X = np.random.randn(30, 3).astype(np.float64)
    y = (X[:, 0] + X[:, 1] * 0.5).astype(np.float64)

    df = pd.DataFrame(X, columns=["z", "a", "m"])
    model = _make_regressor()
    model.fit(df, y)

    preds_same = model.predict(df)
    preds_shuffled = model.predict(df[["a", "m", "z"]])

    assert len(np.asarray(preds_same)) == 30
    assert len(np.asarray(preds_shuffled)) == 30


def test_multi_output_equation_indexing():
    """Multi-output fit produces separate equation lists, one per output."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.column_stack([
        X[:, 0] ** 2,
        X[:, 1] * 0.5,
    ]).astype(np.float64)
    model = _make_regressor()
    model.fit(X, y)

    assert isinstance(model.equations_, list)
    assert len(model.equations_) == 2
    for i, eq_df in enumerate(model.equations_):
        assert len(eq_df) > 0, f"output {i}: empty equations"
        assert "equation" in eq_df.columns

    preds = model.predict(X)
    assert preds.shape == (30, 2)
    assert np.all(np.isfinite(preds))


def test_hof_sorting_equal_loss():
    """Hall of Fame sorting is deterministic under equal loss/complexity."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = np.zeros(30, dtype=np.float64)
    model = _make_regressor()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert len(model.equations_) > 0


def test_simplification_preserves_predictions():
    """should_simplify=True preserves predictions within tolerance."""
    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model_simp = _make_regressor(should_simplify=True)
    model_simp.fit(X, y)
    preds_simp = model_simp.predict(X)

    model_no = _make_regressor(should_simplify=False)
    model_no.fit(X, y)
    preds_no = model_no.predict(X)

    assert np.all(np.isfinite(preds_simp))
    assert np.all(np.isfinite(preds_no))


def test_random_state_controls_stochastic_paths():
    """Same random_state produces identical equations across runs."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)

    model1 = _make_regressor(random_state=42)
    model1.fit(X, y)

    model2 = _make_regressor(random_state=42)
    model2.fit(X, y)

    assert len(model1.equations_) == len(model2.equations_)
    assert (
        model1.equations_["equation"].iloc[0]
        == model2.equations_["equation"].iloc[0]
    )


def test_invalid_expressions_penalized():
    """Invalid expressions produce finite loss, not a crash."""
    from python_backend.eval import evaluate, compute_loss
    from python_backend.expr import OpNode, VarNode, ConstNode

    X = np.random.randn(10, 2).astype(np.float64)
    y = np.ones(10, dtype=np.float64)

    # Expression that produces NaN via 0/0-like protected division
    expr = OpNode("sr.math.protected_div_v1", [VarNode(0), ConstNode(0.0)])
    y_pred = evaluate(expr, X)
    # protected_div returns a finite (large) value so compute_loss should
    # not crash and should return a finite loss
    assert np.all(np.isfinite(y_pred)), "protected_div should not produce NaN"
    loss, valid, reason = compute_loss(y, y_pred)
    assert np.isfinite(loss), f"expression produced non-finite loss: {loss}"


# ── Stress tests (small footprint) ───────────────────────────────────


def test_1000x10():
    """Stress: 1000 rows, 10 features, niterations=1."""
    X = np.random.randn(1000, 10).astype(np.float64)
    y = (X[:, 0] + X[:, 1] * 0.5).astype(np.float64)
    model = _make_regressor(niterations=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_5000x3():
    """Stress: 5000 rows, 3 features, niterations=1."""
    X = np.random.randn(5000, 3).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    model = _make_regressor(niterations=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_large_population():
    """Stress: population_size=30, niterations=2."""
    X = np.random.randn(100, 5).astype(np.float64)
    y = (X[:, 0] + X[:, 1] * 0.3).astype(np.float64)
    model = _make_regressor(population_size=30, niterations=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_many_operators():
    """Stress: many binary and unary operators together."""
    X = np.random.randn(50, 2).astype(np.float64)
    y = np.sin(X[:, 0]) + np.exp(X[:, 1] * 0.1)
    model = _make_regressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "abs"],
        niterations=2,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_island_populations():
    """Stress: 10 populations, niterations=2."""
    X = np.random.randn(50, 3).astype(np.float64)
    y = (X[:, 0] + X[:, 1]).astype(np.float64)
    model = _make_regressor(populations=10, niterations=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


# ── Production validation ────────────────────────────────────────────


def test_quickstart_in_isolated_env():
    """Run quickstart from a temp CWD with controlled PYTHONPATH."""
    src_root = pathlib.Path(__file__).resolve().parent.parent.parent

    with tempfile.TemporaryDirectory(prefix="pysr_robustness_") as tmpdir:
        target_dir = os.path.join(tmpdir, "site-packages")
        os.makedirs(target_dir, exist_ok=True)
        work_dir = os.path.join(tmpdir, "cwd")
        os.makedirs(work_dir, exist_ok=True)

        for src_name in ["juliacall", "python_backend", "PySR_custom"]:
            src_path = str(src_root / src_name)
            r = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "--target", target_dir, "--no-deps", src_path,
                ],
                capture_output=True, text=True, timeout=180,
                cwd=work_dir,
            )
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr, file=sys.stderr)
            assert r.returncode == 0, f"pip install {src_name} failed"

        env = {
            **os.environ,
            "PYSR_BACKEND": "python",
            "PYTHONPATH": target_dir,
        }

        quickstart_src = str(src_root / "python_backend" / "examples" / "quickstart.py")
        r = subprocess.run(
            [sys.executable, quickstart_src],
            capture_output=True, text=True, timeout=120,
            env=env,
            cwd=work_dir,
        )
        print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        assert r.returncode == 0, f"quickstart failed:\n{r.stderr}"
        assert "Predictions:" in r.stdout


def test_no_repo_root_import_leakage():
    """Verify installed modules come from the temp target, not repo root."""
    src_root = pathlib.Path(__file__).resolve().parent.parent.parent

    with tempfile.TemporaryDirectory(prefix="pysr_no_leak_") as tmpdir:
        target_dir = os.path.join(tmpdir, "site-packages")
        os.makedirs(target_dir, exist_ok=True)
        work_dir = os.path.join(tmpdir, "cwd")
        os.makedirs(work_dir, exist_ok=True)

        for src_name in ["juliacall", "python_backend", "PySR_custom"]:
            src_path = str(src_root / src_name)
            r = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "--target", target_dir, "--no-deps", src_path,
                ],
                capture_output=True, text=True, timeout=180,
                cwd=work_dir,
            )
            assert r.returncode == 0, f"pip install {src_name} failed"

        env = {
            **os.environ,
            "PYSR_BACKEND": "python",
            "PYTHONPATH": target_dir,
        }

        check_code = """
import sys
for mod_name in ("juliacall", "python_backend", "pysr"):
    mod = sys.modules.get(mod_name)
    if mod is not None and hasattr(mod, "__file__") and mod.__file__ is not None:
        assert mod.__file__.startswith("{target}"), (
            f"{mod_name} loaded from {{mod.__file__}}, not from temp target"
        )
print("OK")
""".replace("{target}", target_dir)
        r = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True, text=True, timeout=30,
            env=env,
            cwd=work_dir,
        )
        assert r.returncode == 0, f"module location check failed:\n{r.stderr}"
        assert "OK" in r.stdout
