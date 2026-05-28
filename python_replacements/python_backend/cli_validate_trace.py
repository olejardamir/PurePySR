from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a governed trace against TRACE_SCHEMA.md"
    )
    parser.add_argument(
        "--trace", required=True, help="Path to trace.jsonl",
    )
    parser.add_argument(
        "--digests", required=False, default=None,
        help="Optional path to digests.json for digest comparison",
    )
    parser.add_argument(
        "--policy", required=False, default=None,
        help="Optional path to policy.json to validate policy_digest",
    )
    parser.add_argument(
        "--operator-manifest", required=False, default=None,
        help="Optional path to sr-operator-registry.yaml "
             "(default: repo-root sr-operator-registry.yaml)",
    )
    args = parser.parse_args()

    from python_backend.validator import (
        load_jsonl,
        validate_required_keys,
        validate_digests,
    )

    try:
        records = load_jsonl(args.trace)
        validate_required_keys(records)

        digests_ref: dict[str, str] = {}
        if args.digests:
            with open(args.digests) as f:
                digests_ref = json.load(f)

        manifest_bytes: bytes | None = None
        if args.operator_manifest:
            manifest_bytes = pathlib.Path(args.operator_manifest).read_bytes()
        elif args.digests and digests_ref.get("operator_manifest_digest"):
            mpath = (
                pathlib.Path(__file__).resolve().parent
                / "data" / "sr-operator-registry.yaml"
            )
            if mpath.exists():
                manifest_bytes = mpath.read_bytes()

        policy_dict: dict | None = None
        if args.policy:
            with open(args.policy) as f:
                policy_dict = json.load(f)

        validate_digests(
            records, digests_ref,
            operator_manifest_bytes_data=manifest_bytes,
            policy_dict=policy_dict,
        )

        print("OK")
        sys.exit(0)

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
