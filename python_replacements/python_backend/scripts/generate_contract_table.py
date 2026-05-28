"""Regenerate the CONTRACT_PYSR_CUSTOM.md option-coverage matrix from option_gate.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python_backend.option_gate import _COVERAGE_TABLE_RAW

STATUS_LABEL = {
    "supported": "supported",
    "rejected_with_clear_error": "rejected_with_clear_error",
    "accepted_but_ignored_with_warning": "accepted_but_ignored_with_warning",
    "pass_through": "supported_pass_through",
}


def error_or_warning_code(entry):
    code = entry.get("code", "")
    return f"`{code}`" if code else "``"


def status(entry):
    raw = entry["status"]
    mapped = STATUS_LABEL.get(raw)
    if mapped is not None:
        return mapped
    return raw


rows = sorted(_COVERAGE_TABLE_RAW, key=lambda e: e["option"])
lines = [
    "| option_name | source_api | compatibility_level_required | status | error_or_warning_code |",
    "|---|---:|---:|---|---:|",
]
for r in rows:
    option = f"`{r['option']}`"
    source = "`PySRRegressor.__init__`"
    level = r["level"]
    st = status(r)
    code = error_or_warning_code(r)
    lines.append(f"| {option} | {source} | {level} | {st} | {code} |")

print("\n".join(lines))
