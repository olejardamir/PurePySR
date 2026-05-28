from __future__ import annotations

import json
import re
from typing import Any

from python_backend.trace import (
    REQUIRED_RUN_START_KEYS,
    REQUIRED_SEARCH_STEP_KEYS,
    REQUIRED_RUN_END_KEYS,
    canonical_json,
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"SR-ERR-TRACE-004: invalid JSON on line {i}: {e}"
                ) from e
    return records


def validate_required_keys(records: list[dict[str, Any]]) -> None:
    starts = [r for r in records if r.get("record_type") == "run_start"]
    ends = [r for r in records if r.get("record_type") == "run_end"]

    if len(starts) != 1:
        raise ValueError(
            f"SR-ERR-TRACE-001: expected exactly 1 run_start, found {len(starts)}"
        )
    if len(ends) != 1:
        raise ValueError(
            f"SR-ERR-TRACE-001: expected exactly 1 run_end, found {len(ends)}"
        )

    for i, rec in enumerate(records):
        rt = rec.get("record_type")
        if rt == "run_start":
            missing = REQUIRED_RUN_START_KEYS - rec.keys()
            if missing:
                raise ValueError(
                    f"SR-ERR-TRACE-001: run_start missing keys: {sorted(missing)}"
                )

            cs = rec.get("completion_status")
            if cs != "in_progress":
                raise ValueError(
                    f"SR-ERR-TRACE-002: run_start completion_status={cs!r}, "
                    f"expected \"in_progress\""
                )

        elif rt == "search_step":
            missing = REQUIRED_SEARCH_STEP_KEYS - rec.keys()
            if missing:
                raise ValueError(
                    f"SR-ERR-TRACE-001: search_step #{rec.get('t', i)} "
                    f"missing keys: {sorted(missing)}"
                )

            if rec.get("score") is not None:
                raise ValueError(
                    f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                    f"score={rec['score']!r}, must be null"
                )

            ff = rec.get("rng_fingerprint", "")
            if not re.match(r"^u64:[0-9a-f]{16}$", ff):
                raise ValueError(
                    f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                    f"rng_fingerprint={ff!r}, expected u64:<16-hex>"
                )

            valid_statuses = {"valid", "invalid", "error"}
            vs = rec.get("validity_status")
            if vs not in valid_statuses:
                raise ValueError(
                    f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                    f"validity_status={vs!r}, expected one of {valid_statuses}"
                )

            update_statuses = {"inserted", "replaced", "unchanged"}
            aus = rec.get("archive_update_status")
            if aus not in update_statuses:
                raise ValueError(
                    f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                    f"archive_update_status={aus!r}, "
                    f"expected one of {update_statuses}"
                )

            loss = rec.get("loss")
            invalid_code = rec.get("invalid_reason_code")

            if vs == "valid":
                if loss is None:
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"validity_status=valid but loss is None"
                    )
                if not isinstance(loss, str):
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"loss={loss!r} must be a string"
                    )
                try:
                    float(loss)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"validity_status=valid but loss={loss!r} "
                        f"is not a parseable numeric string"
                    ) from None

                if invalid_code is not None:
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"validity_status=valid but "
                        f"invalid_reason_code={invalid_code!r}, must be null"
                    )

            elif vs == "invalid":
                if loss is not None:
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"validity_status=invalid but loss={loss!r}, must be null"
                    )

                if not isinstance(invalid_code, str) or not invalid_code:
                    raise ValueError(
                        f"SR-ERR-TRACE-002: search_step #{rec.get('t', i)} "
                        f"validity_status=invalid but "
                        f"invalid_reason_code={invalid_code!r}, "
                        f"expected non-empty string"
                    )

        elif rt == "run_end":
            missing = REQUIRED_RUN_END_KEYS - rec.keys()
            if missing:
                raise ValueError(
                    f"SR-ERR-TRACE-001: run_end missing keys: {sorted(missing)}"
                )

            cs = rec.get("completion_status")
            if cs not in ("success", "failed"):
                raise ValueError(
                    f"SR-ERR-TRACE-002: run_end completion_status={cs!r}, "
                    f"expected \"success\" or \"failed\""
                )

            tr = rec.get("termination_reason")
            if not isinstance(tr, str) or not tr:
                raise ValueError(
                    f"SR-ERR-TRACE-002: run_end termination_reason={tr!r}, "
                    f"expected non-empty string"
                )

        else:
            raise ValueError(
                f"SR-ERR-TRACE-002: unknown record_type={rt!r} at index {i}"
            )


def compute_step_trace_digest(records: list[dict[str, Any]]) -> str:
    from python_backend.digests import _sha256

    steps = [r for r in records if r.get("record_type") == "search_step"]
    lines = "\n".join(canonical_json(r) for r in steps)
    return _sha256(lines.encode("utf-8"))


def validate_digests(
    records: list[dict[str, Any]],
    digests: dict[str, str],
    *,
    operator_manifest_bytes_data: bytes | None = None,
    policy_dict: dict[str, Any] | None = None,
) -> None:
    from python_backend.digests import _sha256, policy_digest, operator_manifest_digest

    ends = [r for r in records if r.get("record_type") == "run_end"]
    if not ends:
        raise ValueError("SR-ERR-TRACE-001: no run_end record found")
    rec_digest = ends[0].get("step_trace_digest", "")
    computed = compute_step_trace_digest(records)

    if rec_digest != computed:
        raise ValueError(
            f"SR-ERR-TRACE-003: step_trace_digest mismatch: "
            f"record={rec_digest}, computed={computed}"
        )

    if digests:
        ref_digest = digests.get("step_trace_digest", "")
        if ref_digest and ref_digest != computed:
            raise ValueError(
                f"SR-ERR-TRACE-003: step_trace_digest mismatch with reference: "
                f"reference={ref_digest}, computed={computed}"
            )

    if operator_manifest_bytes_data is not None:
        rec_om_digest = (
            ends[0].get("operator_manifest_digest")
            or digests.get("operator_manifest_digest", "")
        )
        if rec_om_digest:
            computed_om = operator_manifest_digest(operator_manifest_bytes_data)
            if rec_om_digest != computed_om:
                raise ValueError(
                    f"SR-ERR-TRACE-003: operator_manifest_digest mismatch: "
                    f"expected={rec_om_digest}, computed={computed_om}"
                )

    if policy_dict is not None:
        rec_pol_digest = (
            ends[0].get("numeric_policy_digest")
            or digests.get("policy_digest", "")
        )
        if rec_pol_digest:
            computed_pol = policy_digest(policy_dict)
            if rec_pol_digest != computed_pol:
                raise ValueError(
                    f"SR-ERR-TRACE-003: policy_digest mismatch: "
                    f"expected={rec_pol_digest}, computed={computed_pol}"
                )


def validate_options_coverage(
    options_dict: dict[str, object],
    *,
    pass_through: dict[str, object] | None = None,
    known_as: str | None = None,
) -> None:
    from python_backend.option_gate import PASS_THROUGH, REJECTED, check_option_coverage

    results = check_option_coverage(
        options_dict,
        pass_through=pass_through,
        known_as=known_as,
    )
    errors: list[str] = []
    for r in results:
        if r["status"] == "unknown":
            errors.append(f"{r['code']}: {r['message']}")
        elif r["status"] == PASS_THROUGH and r.get("code", ""):
            # pass-through with a non-empty code means it was unrecognized
            errors.append(f"{r['code']}: {r['message']}")

    if errors:
        raise ValueError("\n".join(errors))
