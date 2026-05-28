from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import yaml

from python_backend.backend import PythonSRBackend
from python_backend.golden import (
    generate_dataset,
    load_problem,
    load_problem_capability_level,
    load_problem_operator_ids,
    load_search_options,
)
from python_backend.options import BackendOptions
from python_backend.trace import canonical_json, dump_jsonl


def run_and_write_artifacts(
    problem_id: str,
    *,
    seed: int,
    out_dir: str,
) -> dict[str, str]:
    problem = load_problem(problem_id)
    binary_ids, unary_ids = load_problem_operator_ids(problem)

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X, y = generate_dataset(problem, rng=rng)

    options = _make_options(problem)
    backend = PythonSRBackend()
    result = backend.equation_search(X, y, options=options, seed=seed)

    trace_path = out_path / "trace.jsonl"
    dump_jsonl(result["trace_records"], str(trace_path))

    digests_path = out_path / "digests.json"
    with open(digests_path, "w") as f:
        json.dump(result["digests"], f, sort_keys=True)

    policy_path = out_path / "policy.json"
    with open(policy_path, "w") as f:
        json.dump(result["policy_dict"], f, sort_keys=True)

    dataset_manifest = _build_dataset_manifest(
        result["dataset_manifest"],
        result["digests"]["dataset_digest"],
        problem,
    )
    dataset_path = out_path / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_manifest, f, sort_keys=True)

    archive_path = out_path / "archive.json"
    archive_entries = [
        {"hash": e["hash"], "loss": str(e["loss"]), "complexity": e["complexity"]}
        for e in result["hall_of_fame"]
    ]
    archive_path.write_text(canonical_json(archive_entries))

    return {
        "trace": str(trace_path),
        "digests": str(digests_path),
        "policy": str(policy_path),
        "dataset": str(dataset_path),
        "archive": str(archive_path),
    }


def _make_options(problem: dict[str, Any]) -> BackendOptions:
    so = load_search_options(problem)
    return BackendOptions(
        binary_operators=problem["operators"]["binary"],
        unary_operators=problem["operators"].get("unary", []),
        **so,
    )


def _build_dataset_manifest(
    base: dict[str, object],
    dataset_digest: str,
    problem: dict[str, Any],
) -> dict[str, object]:
    return {
        "dataset_digest": dataset_digest,
        **base,
        "generator": problem["generator"],
    }
