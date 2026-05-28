from __future__ import annotations

import hashlib
import json
from typing import Any

from python_backend.trace import canonical_json


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def operator_manifest_digest(manifest_bytes: bytes) -> str:
    return _sha256(manifest_bytes)


def policy_digest(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical.encode("utf-8"))


def step_trace_digest(records: list[dict[str, Any]]) -> str:
    lines = "\n".join(canonical_json(r) for r in records)
    return _sha256(lines.encode("utf-8"))
