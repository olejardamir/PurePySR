from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from python_backend.option_gate import (
    PASS_THROUGH,
    REJECTED,
    check_option_coverage,
)


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a PySR options dict against the coverage table",
    )
    parser.add_argument(
        "--options", required=True,
        help="Path to JSON file containing the options dict",
    )
    parser.add_argument(
        "--pass-through", required=False, default=None,
        help="Optional path to JSON file containing pass-through options",
    )
    parser.add_argument(
        "--known-as", required=False, default=None,
        help="Optional label describing the caller context",
    )
    args = parser.parse_args()

    options = _load_json(args.options)
    pass_through: dict[str, Any] | None = None
    if args.pass_through:
        pass_through = _load_json(args.pass_through)

    results = check_option_coverage(
        options,
        pass_through=pass_through,
        known_as=args.known_as,
    )

    n_rejected = 0
    n_unknown = 0
    n_ignored = 0

    print("=== Option Coverage Report ===")
    if args.known_as:
        print(f"known_as: {args.known_as}")
    print()

    hdr = f"{'option':25s} {'status':35s} {'code':20s} message"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        option = r["option"]
        status = r["status"]
        code = r["code"]
        msg = r["message"]

        if status == REJECTED:
            n_rejected += 1
        elif status == "unknown":
            n_unknown += 1
        elif status == PASS_THROUGH and r.get("code", ""):
            n_unknown += 1
        elif status == "accepted_but_ignored_with_warning":
            n_ignored += 1

        print(f"{option:25s} {status:35s} {code:20s} {msg}")

    print()
    total_issues = n_rejected + n_unknown
    if total_issues == 0:
        print(f"result: PASS ({n_rejected} rejected, {n_unknown} unknown, "
              f"{n_ignored} ignored)")
        sys.exit(0)
    else:
        print(f"result: FAIL ({n_rejected} rejected, {n_unknown} unknown, "
              f"{n_ignored} ignored)")
        sys.exit(1)


if __name__ == "__main__":
    main()
