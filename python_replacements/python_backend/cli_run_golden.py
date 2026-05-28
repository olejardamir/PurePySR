from __future__ import annotations

import argparse
import subprocess
import sys

from python_backend.run_artifacts import run_and_write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a golden problem and write governed artifacts",
    )
    parser.add_argument(
        "--problem", required=True, help="Golden problem ID (e.g. GOLDEN-LIN-001)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for artifacts",
    )
    args = parser.parse_args()

    paths = run_and_write_artifacts(
        problem_id=args.problem,
        seed=args.seed,
        out_dir=args.out,
    )

    print(f"trace:        {paths['trace']}")
    print(f"digests:      {paths['digests']}")
    print(f"policy:       {paths['policy']}")
    print(f"dataset:      {paths['dataset']}")
    print(f"archive:      {paths['archive']}")
    print()

    # Auto-validate with the CLI validator
    validator_args = [
        sys.executable,
        "-m",
        "python_backend.cli_validate_trace",
        "--trace", paths["trace"],
        "--digests", paths["digests"],
        "--policy", paths["policy"],
    ]

    print("Running trace validation...")
    cp = subprocess.run(validator_args, capture_output=True, text=True)
    if cp.returncode != 0:
        print(f"Validation FAILED (exit {cp.returncode})", file=sys.stderr)
        print(cp.stderr, file=sys.stderr)
        sys.exit(cp.returncode)

    print(cp.stdout.strip())


if __name__ == "__main__":
    main()
