"""Test that CONTRACT_PYSR_CUSTOM.md's option-coverage table is in sync with option_gate.py."""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CONTRACT_PATH = os.path.join(REPO_ROOT, "CONTRACT_PYSR_CUSTOM.md")
GENERATOR_PATH = os.path.join(
    REPO_ROOT, "python_backend", "scripts", "generate_contract_table.py"
)


def test_contract_table_in_sync():
    """Regenerate the table from option_gate.py and compare to the embedded table."""
    # Run the generator
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, GENERATOR_PATH],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Generator failed:\n{result.stderr}"
    )
    generated_lines = result.stdout.strip().splitlines()

    # Read the contract file and extract the table (lines 69-175)
    with open(CONTRACT_PATH) as f:
        contract_lines = f.read().splitlines()

    # Find the table: starts with header row + separator, ends before "---"
    header_idx = None
    for i, line in enumerate(contract_lines):
        if line.startswith("| option_name | source_api |"):
            header_idx = i
            break
    assert header_idx is not None, "Could not find table header in CONTRACT_PYSR_CUSTOM.md"

    # Collect all table rows until a non-table line
    table_lines = []
    for line in contract_lines[header_idx:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)

    # Compare: first two rows (header + separator) are identical
    assert generated_lines[0] == table_lines[0], (
        f"Header mismatch:\n  generated: {generated_lines[0]}\n  contract:  {table_lines[0]}"
    )
    assert generated_lines[1] == table_lines[1], (
        f"Separator mismatch:\n  generated: {generated_lines[1]}\n  contract:  {table_lines[1]}"
    )

    # Data rows: sort both for comparison to avoid ordering sensitivity
    gen_data = sorted(generated_lines[2:])
    contract_data = sorted(table_lines[2:])

    assert len(gen_data) == len(contract_data), (
        f"Row count mismatch: generated={len(gen_data)}, contract={len(contract_data)}"
    )

    for g, c in zip(gen_data, contract_data):
        assert g == c, (
            f"Row mismatch:\n  generated: {g}\n  contract:  {c}"
        )
