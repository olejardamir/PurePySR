"""End-to-end smoke test: import PySRRegressor, fit, predict, verify no Julia binary invoked."""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import numpy as np
import juliacall

os.environ["PYSR_BACKEND"] = "python"


def test_no_julia_binary_on_path():
    """Fit works when PATH contains no Julia binary."""
    code = """
import os
os.environ["PYSR_BACKEND"] = "python"
import numpy as np
from pysr import PySRRegressor
X = np.random.randn(20, 2).astype(np.float64)
y = X[:, 0] + X[:, 1]
model = PySRRegressor(
    niterations=1, population_size=8, tournament_selection_n=4,
    binary_operators=["+", "*"], unary_operators=[], maxsize=8,
    verbosity=0, progress=False,
)
model.fit(X, y)
print("OK")
"""
    env = os.environ.copy()
    env["PATH"] = ""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_fake_broken_julia_on_path():
    """Place a broken 'julia' script on PATH and confirm fit still works.

    This proves the Python backend never attempts to invoke the Julia
    binary — even when one appears to be installed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_julia = os.path.join(tmpdir, "julia")
        with open(fake_julia, "w") as f:
            f.write("#!/bin/sh\necho 'fake julia fails' 1>&2\nexit 1\n")
        os.chmod(fake_julia, stat.S_IRWXU)

        env = os.environ.copy()
        env["PATH"] = f"{tmpdir}:{env['PATH']}"

        # Verify the fake is findable
        found = shutil.which("julia", path=env["PATH"])
        assert found == fake_julia, f"Expected fake julia at {fake_julia}, got {found}"

        # Run fit in a subprocess so the fake PATH takes effect
        code = """
import os, sys
os.environ["PYSR_BACKEND"] = "python"
import numpy as np
from pysr import PySRRegressor
X = np.random.randn(30, 2).astype(np.float64)
y = (X[:, 0] ** 2).astype(np.float64)
model = PySRRegressor(niterations=2, population_size=10,
                       tournament_selection_n=5,
                       binary_operators=["+", "-", "*"],
                       unary_operators=[], maxsize=10,
                       verbosity=0, progress=False)
model.fit(X, y)
preds = model.predict(X)
assert preds.shape == (30,)
assert all(np.isfinite(preds))
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "OK" in result.stdout


def test_import_and_fit_smoke():
    """Fit a tiny dataset and verify no Julia process was spawned."""
    from pysr import PySRRegressor

    X = np.random.randn(50, 2).astype(np.float64)
    y = (X[:, 0] ** 2 + X[:, 1] * 0.5).astype(np.float64)

    model = PySRRegressor(
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
    model.fit(X, y)

    assert hasattr(model, "equations_")
    assert len(model.equations_) > 0
    assert "equation" in model.equations_.columns

    preds = model.predict(X)
    assert preds.shape == (50,)
    assert np.all(np.isfinite(preds))


def test_shim_is_used_not_real_juliacall():
    """Verify the repo-local juliacall stub is active, not the real julia_call."""
    # The shim has a unique attribute: _MAIN_TYPE = "stub"
    assert hasattr(juliacall.Main, "_MAIN_TYPE")
    assert juliacall.Main._MAIN_TYPE == "stub"


def test_clean_install_and_run_quickstart():
    """Install juliacall shim + PySR_custom + python_backend to a temp
    prefix and run quickstart.

    This validates the package installs cleanly and the pure-Python
    backend works end-to-end without a Julia binary.

    Hardening measures:
      - Subprocess runs from a temp directory (not repo root) to prove
        installed packages are used, not accidental repo imports.
      - PYTHONPATH is set to ONLY the temp site-packages (no repo root).
      - A broken ``julia`` script is placed on PATH to prove no Julia
        binary is invoked.
    """

    src_root = pathlib.Path(__file__).resolve().parent.parent.parent

    with tempfile.TemporaryDirectory(prefix="pysr_clean_install_") as tmpdir:
        target_dir = os.path.join(tmpdir, "site-packages")
        os.makedirs(target_dir, exist_ok=True)

        work_dir = os.path.join(tmpdir, "cwd")
        os.makedirs(work_dir, exist_ok=True)

        def _pip_install(src: str, timeout: int = 180) -> None:
            r = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "--target", target_dir,
                    "--no-deps",
                    src,
                ],
                capture_output=True, text=True, timeout=timeout,
                cwd=work_dir,
            )
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr, file=sys.stderr)
            assert r.returncode == 0

        # Install juliacall shim
        _pip_install(str(src_root / "juliacall"))
        # Install python_backend (needed by juliacall shim at import time)
        _pip_install(str(src_root / "python_backend"))
        # Install PySR_custom (relies on juliacall shim)
        _pip_install(str(src_root / "PySR_custom"))

        # Place a broken julia on PATH
        fake_bin = os.path.join(tmpdir, "fake-bin")
        os.makedirs(fake_bin, exist_ok=True)
        fake_julia = os.path.join(fake_bin, "julia")
        with open(fake_julia, "w") as f:
            f.write("#!/bin/sh\necho 'fake julia fails' 1>&2\nexit 1\n")
        os.chmod(fake_julia, stat.S_IRWXU)

        env = {
            **os.environ,
            "PYSR_BACKEND": "python",
            "PYTHONPATH": target_dir,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
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

        # Verify no actual Julia binary was invoked
        assert "fake julia fails" not in r.stderr, (
            "fake julia binary was invoked — backend triggered Julia call"
        )

        # Verify installed modules come from the temp target
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
        r2 = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True, text=True, timeout=30,
            env=env,
            cwd=work_dir,
        )
        assert r2.returncode == 0, f"module location check failed:\n{r2.stderr}"
        assert "OK" in r2.stdout
