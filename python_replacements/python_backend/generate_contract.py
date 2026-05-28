#!/usr/bin/env python3
"""Auto-generate CONTRACT_PYSR_CUSTOM.md from option_gate.py coverage table."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from python_backend.option_gate import _COVERAGE_TABLE_RAW

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "CONTRACT_PYSR_CUSTOM.md"

_HEADER = """# CONTRACT_PYSR_CUSTOM (Auto-generated)

**Purpose:** A machine-auditable "fixed consumer contract" for a Python-only backend effort: enumerate what `PySR_custom` can pass into the backend, and require an explicit status for each option per `EQC-SR-v1.0.3` option coverage rules (§3.3).

**Key rule:** `PySR_custom` remains unchanged. The backend must either:
- support the option semantics, or
- reject with a clear structured error, or
- accept-but-ignore with an explicit warning code (only if declared).

Primary sources:
- `PySR_custom/pysr/sr.py` (public surface)
- `EQC-SR-v1.0.3-final.md` (§3.3–3.6, plus semantics sections)

---

## 1) Option coverage status vocabulary (EQC-SR §3.3)

Each option MUST be classified as exactly one of:

- `supported`
- `supported_with_deviation`
- `accepted_but_ignored_with_warning`
- `rejected_with_clear_error`
- `not_in_scope_for_level`
- `unresolved`

Minimum required fields per row (EQC-SR §3.3):

- `option_name`
- `source_api`
- `compatibility_level_required`
- `status`
- `semantic_owner_spec`
- `implementation_artifact`
- `test_coverage`
- `error_or_warning_code`
- `notes`

The backend MUST NOT silently ignore a PySR-style option unless this document explicitly marks it as `accepted_but_ignored_with_warning` and declares the warning code.

---

## 2) Capability profile (EQC-SR §3.5)

This document must also pin a capability profile (draft; to be finalized later):

```yaml
capability_profile:
  profile_id: "SR-CAP-PYSR_CUSTOM-FIXED-v1"
  expression_form: "tree"
  typed_expressions: false
  supported_regression_modes: ["scalar_real_regression"]
  search_strategy: ["mutation_only", "crossover_optional"]
  populations: "single_or_multi"
  constants: "ephemeral_and_optimized"
  constant_optimizer: "deterministic_local"
  evaluation_backends: ["numpy_vectorized", "numba_optional"]
  parallelism: "declared_deterministic_or_disabled"
  custom_operators: "registry_required"
  simplification: "declared_operator"
  exports: ["canonical", "python_callable", "sympy_optional"]
```

---

## 3) Public options inventory (from `option_gate.py`)

Below is the **option coverage matrix** auto-generated from `python_backend/option_gate.py`.
"""

_TABLE_HEADER = """| option_name | source_api | compatibility_level_required | status | error_or_warning_code |
|---|---:|---|---:|---:|
"""

_FOOTER = """
---

## 4) Next fill-in tasks (recommended)

1. Review any `not_in_scope_for_level` or `unresolved` options.
2. For each `supported` option, bind to:
   - a semantic owner spec section (EXPR/OPLIB/EVAL/RANK/SEARCH/TRACE)
   - at least one test (unit/property/golden trace)
3. Run `python_backend/cli_check_options.py` to validate runtime option coverage.

---

## 5) Initial test coverage checklist (starter)

- `TEST-SMOKE-001`: tiny dataset, `niterations=2`, serial, deterministic seed; asserts `equations_` exists and best loss decreases.
- `TEST-OPS-001`: binary/unary operator registry: parse, validate arity, complexity cost; protected ops behave deterministically on edge cases.
- `TEST-EXPR-001`: canonical serialization + structural hash stable; commutative ordering rules stable.
- `TEST-EVAL-001`: evaluation shape/dtype deterministic; non-finite predictions handled per policy.
- `TEST-LOSS-001`: MSE (and weighted variants if used) matches reference within tolerance.
- `TEST-RANK-001`: tie-break ordering stable; invalid candidates always worse than valid.
- `TEST-HOF-001`: hall-of-fame update deterministic; duplicate policy matches declared rule.
- `TEST-PARETO-001`: Pareto frontier deterministic ordering and selection fallback.
- `TEST-EXPORT-001`: SymPy/NumPy callable export equivalence on a held-out dataset (if export is enabled).
"""

_STATUS_MAP = {
    "supported": "supported",
    "accepted_but_ignored_with_warning": "accepted_but_ignored_with_warning",
    "rejected_with_clear_error": "rejected_with_clear_error",
    "pass_through": "supported_pass_through",
}


def _status_cell(entry: dict) -> str:
    raw = entry["status"]
    mapped = _STATUS_MAP.get(raw, raw)
    return mapped


def main() -> None:
    entries = sorted(_COVERAGE_TABLE_RAW, key=lambda e: e["option"])

    lines = [_HEADER, _TABLE_HEADER]
    for e in entries:
        opt = e["option"]
        level = e["level"]
        status = _status_cell(e)
        code = e.get("code", "") or ""
        lines.append(f"| `{opt}` | `PySRRegressor.__init__` | {level} | {status} | `{code}` |\n")

    lines.append(_FOOTER)

    content = "".join(lines)
    CONTRACT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {CONTRACT_PATH} ({len(entries)} options)")


if __name__ == "__main__":
    main()
