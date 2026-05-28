from __future__ import annotations

import json
from typing import Any


OPTIONAL_NUMERIC_NULL = None
NUMERIC_STRING_FIELDS: frozenset[str] = frozenset([
    "loss",
    "best_loss_after_step",
])

REQUIRED_RUN_START_KEYS: frozenset[str] = frozenset([
    "record_type",
    "run_id",
    "spec_version",
    "compatibility_level",
    "seed",
    "prng_family",
    "operator_manifest_digest",
    "dataset_digest",
    "numeric_policy_digest",
    "environment_profile",
    "evaluation_backend",
    "start_time_policy",
    "completion_status",
    "termination_reason",
])

REQUIRED_SEARCH_STEP_KEYS: frozenset[str] = frozenset([
    "record_type",
    "t",
    "rng_fingerprint",
    "population_id",
    "selected_parent_hashes",
    "proposal_operator",
    "mutation_or_crossover_type",
    "candidate_hash_before",
    "candidate_hash_after",
    "validity_status",
    "loss",
    "complexity",
    "invalid_reason_code",
    "score",
    "archive_update_status",
    "accepted_or_inserted",
    "best_hash_after_step",
    "best_loss_after_step",
    "best_complexity_after_step",
])

REQUIRED_RUN_END_KEYS: frozenset[str] = frozenset([
    "record_type",
    "run_id",
    "completion_status",
    "termination_reason",
    "final_result_digest",
    "archive_digest",
    "step_trace_digest",
])


def run_start_record(
    run_id: str,
    spec_version: str = "EQC-SR-v1.0.3",
    compatibility_level: str = "SR-L1",
    seed: str = "0",
    prng_family: str = "PCG64",
    operator_manifest_digest: str = "",
    dataset_digest: str = "",
    numeric_policy_digest: str = "",
    environment_profile: str = "cpu-deterministic-v1",
    evaluation_backend: str = "unimplemented",
    start_time_policy: str = "recorded",
    completion_status: str = "in_progress",
    termination_reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record = dict(
        record_type="run_start",
        run_id=run_id,
        spec_version=spec_version,
        compatibility_level=compatibility_level,
        seed=seed,
        prng_family=prng_family,
        operator_manifest_digest=operator_manifest_digest,
        dataset_digest=dataset_digest,
        numeric_policy_digest=numeric_policy_digest,
        environment_profile=environment_profile,
        evaluation_backend=evaluation_backend,
        start_time_policy=start_time_policy,
        completion_status=completion_status,
        termination_reason=termination_reason,
    )
    record.update(extra)
    return record


def search_step_record(
    t: int,
    rng_fingerprint: str,
    population_id: int,
    selected_parent_hashes: list[str],
    proposal_operator: str,
    mutation_or_crossover_type: str,
    candidate_hash_after: str,
    validity_status: str,
    loss: str | None = None,
    complexity: int | None = None,
    archive_update_status: str = "unchanged",
    accepted_or_inserted: bool = False,
    best_hash_after_step: str | None = None,
    best_loss_after_step: str | None = None,
    best_complexity_after_step: int | None = None,
    invalid_reason_code: str | None = None,
    candidate_hash_before: str | None = None,
    score: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if loss is not None:
        loss = str(loss)
    if best_loss_after_step is not None:
        best_loss_after_step = str(best_loss_after_step)
    record = dict(
        record_type="search_step",
        t=t,
        rng_fingerprint=rng_fingerprint,
        population_id=population_id,
        selected_parent_hashes=selected_parent_hashes,
        proposal_operator=proposal_operator,
        mutation_or_crossover_type=mutation_or_crossover_type,
        candidate_hash_before=candidate_hash_before,
        candidate_hash_after=candidate_hash_after,
        validity_status=validity_status,
        loss=loss,
        complexity=complexity,
        invalid_reason_code=invalid_reason_code,
        score=score,
        archive_update_status=archive_update_status,
        accepted_or_inserted=accepted_or_inserted,
        best_hash_after_step=best_hash_after_step,
        best_loss_after_step=best_loss_after_step,
        best_complexity_after_step=best_complexity_after_step,
    )
    record.update(extra)
    return record


def run_end_record(
    run_id: str,
    completion_status: str = "success",
    termination_reason: str = "max_iterations",
    final_result_digest: str = "",
    archive_digest: str = "",
    step_trace_digest: str = "",
    **extra: Any,
) -> dict[str, Any]:
    record = dict(
        record_type="run_end",
        run_id=run_id,
        completion_status=completion_status,
        termination_reason=termination_reason,
        final_result_digest=final_result_digest,
        archive_digest=archive_digest,
        step_trace_digest=step_trace_digest,
    )
    record.update(extra)
    return record


def canonical_json(obj: Any) -> str:
    """Stable-key JSON serializer.

    - ``None`` → ``null``
    - ``bool`` → ``true`` / ``false``
    - ``int`` → integer (not string)
    - ``float`` → string (``"0.001"``)
    - ``str`` → JSON string
    """
    def _encode(val: Any) -> str:
        if isinstance(val, dict):
            items = sorted((k, _encode(v)) for k, v in val.items())
            return "{" + ",".join(f"{json.dumps(k)}:{v}" for k, v in items) + "}"
        if isinstance(val, list):
            return "[" + ",".join(_encode(v) for v in val) + "]"
        if isinstance(val, bool):
            return "true" if val else "false"
        if val is None:
            return "null"
        if isinstance(val, int) and not isinstance(val, bool):
            return json.dumps(val)
        if isinstance(val, float):
            return json.dumps(str(val))
        return json.dumps(val)

    return _encode(obj)


def dump_jsonl(records: list[dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
