from __future__ import annotations

import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_CONTRACT_PATH = _HERE.parent.parent / "CONTRACT_PYSR_CUSTOM.md"

_OPTION_NAME_COL = 0
_STATUS_COL = 3
_ERROR_CODE_COL = 4


def _parse_option_rows() -> list[list[str]]:
    text = _CONTRACT_PATH.read_text()
    lines = text.splitlines()

    rows: list[list[str]] = []
    found_header = False

    for line in lines:
        if not line.startswith("|"):
            found_header = False
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            continue
        if not found_header:
            if cells[0] == "option_name":
                found_header = True
            continue
        if any("---" in c for c in cells):
            continue
        rows.append(cells)

    return rows


def test_contract_matrix_minimum_rows():
    rows = _parse_option_rows()
    assert len(rows) >= 100, (
        f"Expected >= 100 option rows, got {len(rows)}"
    )


def test_contract_no_silent_ignores():
    rows = _parse_option_rows()
    silent: list[str] = []
    for row in rows:
        status = row[_STATUS_COL] if len(row) > _STATUS_COL else ""
        err_code = row[_ERROR_CODE_COL] if len(row) > _ERROR_CODE_COL else ""
        if status == "accepted_but_ignored_with_warning" and not err_code:
            silent.append(row[_OPTION_NAME_COL] if row else "unknown")
    assert not silent, (
        f"Silent ignores (accepted_but_ignored_with_warning without "
        f"error/warning code): {silent}"
    )
