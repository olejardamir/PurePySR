from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from python_backend.run_artifacts import run_and_write_artifacts


def test_run_artifacts_lin_001() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = run_and_write_artifacts(
            problem_id="GOLDEN-LIN-001",
            seed=0,
            out_dir=tmp,
        )

        assert paths["trace"].endswith("trace.jsonl")
        assert paths["digests"].endswith("digests.json")
        assert paths["policy"].endswith("policy.json")
        assert paths["dataset"].endswith("dataset.json")
        assert paths["archive"].endswith("archive.json")

        with open(paths["trace"]) as f:
            lines = f.readlines()
        assert len(lines) >= 2

        with open(paths["digests"]) as f:
            digests = json.load(f)
        assert "operator_manifest_digest" in digests
        assert "policy_digest" in digests
        assert "step_trace_digest" in digests
        assert "dataset_digest" in digests
        assert "archive_digest" in digests

        with open(paths["policy"]) as f:
            policy = json.load(f)
        assert policy["eps_denom"] == "1e-08"
        assert policy["prng"] == "PCG64"

        with open(paths["dataset"]) as f:
            dataset = json.load(f)
        assert dataset["dataset_digest"] == digests["dataset_digest"]
        assert dataset["n_samples"] == 200
        assert dataset["n_features"] == 1
        assert dataset["dtype"] == "float64"
        assert "generator" in dataset

        with open(paths["archive"]) as f:
            archive = json.load(f)
        assert isinstance(archive, list)

        # Check archive digest matches
        from python_backend.trace import canonical_json
        from python_backend.digests import _sha256
        computed = _sha256(
            canonical_json(archive).encode("utf-8")
        )
        assert computed == digests["archive_digest"], (
            f"archive digest mismatch: computed={computed}, "
            f"expected={digests['archive_digest']}"
        )


def test_cli_run_golden_lin_001() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_run_golden",
             "--problem", "GOLDEN-LIN-001",
             "--seed", "0",
             "--out", tmp],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, (
            f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        )
        assert "OK" in cp.stdout, (
            f"expected OK in stdout, got: {cp.stdout}"
        )
        assert "trace:" in cp.stdout
        assert "digests:" in cp.stdout
        assert "policy:" in cp.stdout
        assert "dataset:" in cp.stdout
        assert "archive:" in cp.stdout


def test_cli_run_golden_repeatable_digests() -> None:
    """The pure-Python golden CLI produces deterministic digests across two runs."""
    import json
    import tempfile
    from pathlib import Path
    import subprocess
    import sys
    import os

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])

    problem_id = "GOLDEN-LIN-001"
    seed = 0

    with tempfile.TemporaryDirectory() as d1:
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_run_golden",
             "--problem", problem_id,
             "--seed", str(seed),
             "--out", d1],
            env=env,
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        digests_path = Path(d1) / "digests.json"
        assert digests_path.is_file(), f"Missing digests.json in {d1}"
        d1_digests = json.loads(digests_path.read_text())

    with tempfile.TemporaryDirectory() as d2:
        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_run_golden",
             "--problem", problem_id,
             "--seed", str(seed),
             "--out", d2],
            env=env,
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        digests_path = Path(d2) / "digests.json"
        assert digests_path.is_file(), f"Missing digests.json in {d2}"
        d2_digests = json.loads(digests_path.read_text())

    # All digests should be identical across reproducible runs
    for key in d1_digests:
        assert key in d2_digests, f"run2 missing {key}"
        assert d1_digests[key] == d2_digests[key], (
            f"digest mismatch for {key!r}: {d1_digests[key]} vs {d2_digests[key]}"
        )
