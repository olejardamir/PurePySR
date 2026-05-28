from __future__ import annotations

import json
import subprocess
import sys
import tempfile

import pytest

from python_backend.option_gate import (
    COVERAGE_TABLE,
    IGNORED,
    PASS_THROUGH,
    REJECTED,
    SUPPORTED,
    check_option_coverage,
)
from python_backend.validator import validate_options_coverage


def test_all_supported_options_pass():
    opts = {k: None for k, v in COVERAGE_TABLE.items() if v["status"] == SUPPORTED}
    results = check_option_coverage(opts)
    for r in results:
        assert r["status"] == SUPPORTED, f"{r['option']}: {r}"
    assert len(results) == len(opts)


def test_rejected_option_flagged():
    opts = {"cluster_manager": "dummy"}
    results = check_option_coverage(opts)
    assert results[0]["status"] == REJECTED
    assert results[0]["code"] != ""


def test_expression_spec_rejected():
    """expression_spec is REJECTED — non-None value produces SR-ERR-OPT-001."""
    opts = {"expression_spec": "dummy"}
    with pytest.warns(UserWarning, match="SR-ERR-OPT-001"):
        results = check_option_coverage(opts)
    assert results[0]["status"] == REJECTED
    assert results[0]["code"] == "SR-ERR-OPT-001"


def test_expression_spec_warning_has_message():
    """Warning for expression_spec includes the descriptive error_message."""
    with pytest.warns(UserWarning, match="templates require Julia"):
        results = check_option_coverage({"expression_spec": "custom"})
    assert results[0]["status"] == REJECTED


def test_expression_spec_object_rejected_via_pysr():
    """A raw expression_spec object (list of dicts) fails through PySRRegressor."""
    import numpy as np
    from pysr import PySRRegressor
    X = np.random.randn(30, 2)
    y = X[:, 0] ** 2
    model = PySRRegressor(
        niterations=2, population_size=10, tournament_selection_n=5,
        binary_operators=["+", "-", "*"], unary_operators=[], maxsize=10,
        verbosity=0, progress=False,
        expression_spec=[{"feature": "x1", "expression": "x1**2"}],
    )
    with pytest.raises(ValueError) as exc:
        model.fit(X, y)
    assert "expression_spec templates require Julia" in str(exc.value)


def test_unknown_option_flagged():
    opts = {"nonexistent_option": 42}
    results = check_option_coverage(opts)
    assert results[0]["status"] == "unknown"


def test_pass_through_known_recorded():
    results = check_option_coverage({}, pass_through={"random_state": 42})
    pt = [r for r in results if r["status"] == PASS_THROUGH]
    assert len(pt) == 1
    assert pt[0]["option"] == "random_state"
    assert pt[0]["code"] == ""


def test_pass_through_unknown_flagged():
    results = check_option_coverage({}, pass_through={"future_opt": "val"})
    unknown = [r for r in results if r["status"] == "unknown"]
    assert len(unknown) == 1
    assert unknown[0]["option"] == "future_opt"


def test_mixed_results():
    opts = {
        "binary_operators": ["+"],
        "cluster_manager": "dummy",
        "turbo": True,
        "unknown_opt": "x",
    }
    results = check_option_coverage(opts)
    statuses = {r["option"]: r["status"] for r in results}
    assert statuses["binary_operators"] == SUPPORTED
    assert statuses["cluster_manager"] == REJECTED
    assert statuses["turbo"] == IGNORED
    assert statuses["unknown_opt"] == "unknown"


def test_validate_options_coverage_passes():
    opts = {"binary_operators": ["+"], "unary_operators": []}
    validate_options_coverage(opts)


def test_validate_options_coverage_rejected_does_not_raise():
    # REJECTED options are documented as unsupported but do not
    # cause a hard validation error (the validator is for trace
    # artifact integrity, not policy enforcement).
    opts = {"cluster_manager": "dummy"}
    validate_options_coverage(opts)


def test_validate_options_coverage_unknown_raises():
    opts = {"foo": "bar"}
    with pytest.raises(ValueError) as exc:
        validate_options_coverage(opts)
    assert "SR-ERR-OPT-004" in str(exc.value)


def _write_json(content: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f)


def test_cli_all_supported():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        _write_json(
            {"binary_operators": ["+"], "unary_operators": [], "niterations": 10},
            opt_path,
        )
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"stdout={cp.stdout}, stderr={cp.stderr}"
        assert "PASS" in cp.stdout


def test_cli_rejected_fails():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        _write_json({"cluster_manager": "dummy"}, opt_path)
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1
        assert "FAIL" in cp.stdout


def test_cli_unknown_fails():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        _write_json({"foo_bar_baz": "val"}, opt_path)
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1
        assert "FAIL" in cp.stdout
        assert "SR-ERR-OPT-004" in cp.stdout


def test_cli_with_pass_through():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        pt_path = f"{tmp}/pt.json"
        _write_json({"binary_operators": ["+"]}, opt_path)
        _write_json({"future_opt": True}, pt_path)
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path, "--pass-through", pt_path],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1
        assert "unknown" in cp.stdout
        assert "future_opt" in cp.stdout


def test_cli_with_pass_through_known():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        pt_path = f"{tmp}/pt.json"
        _write_json({"binary_operators": ["+"]}, opt_path)
        _write_json({"random_state": 42}, pt_path)
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path, "--pass-through", pt_path],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0
        assert "pass_through" in cp.stdout
        assert "random_state" in cp.stdout


def test_cli_with_known_as():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        _write_json({"binary_operators": ["+"]}, opt_path)
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path, "--known-as", "test-caller"],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0
        assert "test-caller" in cp.stdout


def test_cli_deterministic_output():
    with tempfile.TemporaryDirectory() as tmp:
        opt_path = f"{tmp}/opts.json"
        _write_json(
            {"binary_operators": ["+"], "populations": 4, "turbo": True},
            opt_path,
        )
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path],
            capture_output=True, text=True,
        )
        out1 = cp.stdout

        cp2 = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_check_options",
             "--options", opt_path],
            capture_output=True, text=True,
        )
        out2 = cp2.stdout
        assert out1 == out2, "output not deterministic"
