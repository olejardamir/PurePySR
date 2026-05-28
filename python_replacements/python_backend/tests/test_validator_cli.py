from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from python_backend.backend import PythonSRBackend
from python_backend.options import BackendOptions
from python_backend.trace import dump_jsonl
from python_backend.ops import operator_manifest_bytes
from python_backend.digests import operator_manifest_digest, policy_digest


def _tiny_run() -> dict:
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
    backend = PythonSRBackend()
    return backend.equation_search(X, y, options=opts, seed=0)


def _write_trace_and_digests(result: dict, tmp: str) -> tuple[Path, Path]:
    trace_path = Path(tmp) / "trace.jsonl"
    digests_path = Path(tmp) / "digests.json"
    dump_jsonl(result["trace_records"], str(trace_path))
    with open(digests_path, "w") as f:
        json.dump(result["digests"], f, sort_keys=True)
    return trace_path, digests_path


def test_cli_validates_clean_trace():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path, digests_path = _write_trace_and_digests(result, tmp)

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, (
            f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        )
        assert "OK" in cp.stdout


def test_cli_rejects_corrupted_trace():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        records = list(result["trace_records"])
        for rec in records:
            if rec.get("record_type") == "search_step":
                rec.pop("score", None)
                break

        trace_path = Path(tmp) / "trace.jsonl"
        digests_path = Path(tmp) / "digests.json"
        dump_jsonl(records, str(trace_path))
        with open(digests_path, "w") as f:
            json.dump(result["digests"], f, sort_keys=True)

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1, (
            f"CLI should have exited 1 but got {cp.returncode}: "
            f"stdout={cp.stdout}, stderr={cp.stderr}"
        )
        assert "SR-ERR-TRACE-001" in cp.stderr, (
            f"expected SR-ERR-TRACE-001 in stderr, got: {cp.stderr}"
        )


def test_cli_validates_without_digests_file():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.jsonl"
        dump_jsonl(result["trace_records"], str(trace_path))

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, (
            f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        )
        assert "OK" in cp.stdout


def test_cli_rejects_digest_mismatch():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path, digests_path = _write_trace_and_digests(result, tmp)

        bad_digests = dict(result["digests"])
        bad_digests["step_trace_digest"] = (
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        )
        with open(digests_path, "w") as f:
            json.dump(bad_digests, f, sort_keys=True)

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1, (
            f"CLI should have exited 1 but got {cp.returncode}: "
            f"stdout={cp.stdout}, stderr={cp.stderr}"
        )
        assert "SR-ERR-TRACE-003" in cp.stderr, (
            f"expected SR-ERR-TRACE-003 in stderr, got: {cp.stderr}"
        )


def test_cli_rejects_corrupt_rng_fingerprint():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        records = list(result["trace_records"])
        for rec in records:
            if rec.get("record_type") == "search_step":
                rec["rng_fingerprint"] = "bad"
                break

        trace_path, digests_path = _write_trace_and_digests(result, tmp)
        dump_jsonl(records, str(trace_path))

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1, (
            f"CLI should have exited 1 but got {cp.returncode}: "
            f"stdout={cp.stdout}, stderr={cp.stderr}"
        )
        assert "SR-ERR-TRACE-002" in cp.stderr, (
            f"expected SR-ERR-TRACE-002 in stderr, got: {cp.stderr}"
        )


def test_cli_rejects_non_null_score():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        records = list(result["trace_records"])
        for rec in records:
            if rec.get("record_type") == "search_step":
                rec["score"] = "0.5"
                break

        trace_path, digests_path = _write_trace_and_digests(result, tmp)
        dump_jsonl(records, str(trace_path))

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 1, (
            f"CLI should have exited 1 but got {cp.returncode}: "
            f"stdout={cp.stdout}, stderr={cp.stderr}"
        )
        assert "SR-ERR-TRACE-002" in cp.stderr, (
            f"expected SR-ERR-TRACE-002 in stderr, got: {cp.stderr}"
        )


def test_cli_validates_with_policy_and_manifest():
    result = _tiny_run()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path, digests_path = _write_trace_and_digests(result, tmp)

        policy_path = Path(tmp) / "policy.json"
        with open(policy_path, "w") as f:
            json.dump(
                {
                    "seed_space": "uint64",
                    "prng": "PCG64",
                    "dtype": "float64",
                    "fast_math": "forbidden",
                    "stable_sort": True,
                    "deterministic_ties": True,
                    "invalid_policy": "SR-INV-NONFINITE-001",
                    "objective": "(loss, complexity, structural_hash)",
                    "eps_denom": "1e-08",
                    "fraction_replaced": "0.1",
                    "fraction_replaced_hof": "0.1",
                    "optimize_probability": "1.0",
                    "optimizer_iterations": "8",
                    "optimizer_nrestarts": "2",
                    "max_evals": "0",
                    "timeout_in_seconds": "0",
                    "fast_cycle": "False",
                    "turbo": "False",
                    "early_stop_condition": "",
                    "precision": "16",
                    "hof_migration": "True",
                    "parsimony": "0.0",
                    "complexity_of_constants": "1",
                    "complexity_of_operators": "1",
                    "complexity_of_variables": "1",
                    "adaptive_parsimony_scaling": "0.0",
                    "guesses": "0",
                    "fraction_replaced_guesses": "0.0",
                },
                f, sort_keys=True,
            )

        manifest_path = Path(tmp) / "sr-operator-registry.yaml"
        manifest_data = operator_manifest_bytes()
        manifest_path.write_bytes(manifest_data)

        # Verify the digests match before running the CLI
        assert result["digests"]["operator_manifest_digest"] == (
            operator_manifest_digest(manifest_data)
        ), "operator manifest digest pre-check failed"
        actual_digest = result["digests"]["policy_digest"]
        expected_digest = policy_digest({
            "seed_space": "uint64",
            "prng": "PCG64",
            "dtype": "float64",
            "fast_math": "forbidden",
            "stable_sort": True,
            "deterministic_ties": True,
            "invalid_policy": "SR-INV-NONFINITE-001",
            "objective": "(loss, complexity, structural_hash)",
            "eps_denom": "1e-08",
            "fraction_replaced": "0.1",
            "fraction_replaced_hof": "0.1",
            "optimize_probability": "1.0",
            "optimizer_iterations": "8",
            "optimizer_nrestarts": "2",
            "max_evals": "0",
            "timeout_in_seconds": "0",
            "fast_cycle": "False",
            "turbo": "False",
            "early_stop_condition": "",
            "precision": "16",
            "hof_migration": "True",
            "parsimony": "0.0",
            "complexity_of_constants": "1",
            "complexity_of_operators": "1",
            "complexity_of_variables": "1",
            "adaptive_parsimony_scaling": "0.0",
            "guesses": "0",
            "fraction_replaced_guesses": "0.0",
        })
        assert actual_digest == expected_digest, \
            f"policy digest pre-check failed: {actual_digest} != {expected_digest}"

        cp = subprocess.run(
            [sys.executable, "-m", "python_backend.cli_validate_trace",
             "--trace", str(trace_path),
             "--digests", str(digests_path),
             "--policy", str(policy_path),
             "--operator-manifest", str(manifest_path)],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, (
            f"CLI exited {cp.returncode}: stderr={cp.stderr}"
        )
        assert "OK" in cp.stdout
