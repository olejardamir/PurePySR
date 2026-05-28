#!/usr/bin/env python3
"""Compare Python backend results against a local SymbolicRegression.jl clone.

This is an empirical parity harness, not part of the Python-only runtime.
It runs the same small synthetic problems through PySR_custom's Python backend
and, when a Julia executable is available, through a checked-out
SymbolicRegression.jl package.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Case:
    case_id: str
    X: np.ndarray
    y: np.ndarray
    binary_operators: tuple[str, ...]
    unary_operators: tuple[str, ...]


@dataclass(frozen=True)
class ParityCheck:
    case_id: str
    seed: int
    python_loss: float
    julia_loss: float
    loss_ratio: float
    python_complexity: int
    julia_complexity: int
    passed: bool
    reason: str


def build_cases() -> list[Case]:
    rng = np.random.default_rng(123)
    X_quad = rng.uniform(-2.0, 2.0, size=(80, 2)).astype(np.float64)
    y_quad = (X_quad[:, 0] ** 2 + 0.5 * X_quad[:, 1]).astype(np.float64)

    X_sin = rng.uniform(-3.0, 3.0, size=(80, 1)).astype(np.float64)
    y_sin = (np.sin(X_sin[:, 0]) + 0.25 * X_sin[:, 0]).astype(np.float64)

    X_mixed = rng.uniform(-1.5, 1.5, size=(80, 2)).astype(np.float64)
    y_mixed = (np.exp(0.25 * X_mixed[:, 0]) + X_mixed[:, 1] ** 2).astype(np.float64)

    return [
        Case("quadratic", X_quad, y_quad, ("+", "-", "*", "/"), ()),
        Case("sin_linear", X_sin, y_sin, ("+", "-", "*"), ("sin",)),
        Case("exp_quad", X_mixed, y_mixed, ("+", "-", "*"), ("exp",)),
    ]


def parse_seeds(seed: int, seeds: str | None) -> list[int]:
    if seeds is None:
        return [seed]
    parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--seeds must contain at least one integer seed")
    return parsed


def run_python_case(case: Case, seed: int, niterations: int) -> dict[str, object]:
    os.environ["PYSR_BACKEND"] = "python"
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        population_size=30,
        populations=2,
        tournament_selection_n=10,
        ncycles_per_iteration=5,
        binary_operators=list(case.binary_operators),
        unary_operators=list(case.unary_operators),
        maxsize=14,
        maxdepth=8,
        random_state=seed,
        deterministic=True,
        verbosity=0,
        progress=False,
    )
    model.fit(case.X, case.y)
    best = model.get_best()
    return {
        "case": case.case_id,
        "seed": seed,
        "backend": "python",
        "loss": float(best["loss"]),
        "complexity": int(best["complexity"]),
        "equation": str(best["equation"]),
    }


def write_result_csv(results: list[dict[str, object]]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["case", "seed", "backend", "loss", "complexity", "equation"])
    for result in results:
        writer.writerow(
            [
                result["case"],
                result["seed"],
                result["backend"],
                f"{float(result['loss']):.17g}",
                result["complexity"],
                result["equation"],
            ]
        )


def evaluate_parity(
    python_results: list[dict[str, object]],
    julia_results: list[dict[str, object]],
    max_loss_ratio: float,
    absolute_loss_slack: float,
    complexity_slack: int,
) -> list[ParityCheck]:
    python_by_case = {
        (str(result["case"]), int(result["seed"])): result for result in python_results
    }
    julia_by_case = {
        (str(result["case"]), int(result["seed"])): result for result in julia_results
    }
    checks: list[ParityCheck] = []
    for (case_id, seed), py_result in python_by_case.items():
        jl_result = julia_by_case[(case_id, seed)]
        python_loss = float(py_result["loss"])
        julia_loss = float(jl_result["loss"])
        python_complexity = int(py_result["complexity"])
        julia_complexity = int(jl_result["complexity"])
        denominator = max(abs(julia_loss), 1e-12)
        loss_ratio = python_loss / denominator
        loss_limit = max(julia_loss * max_loss_ratio, julia_loss + absolute_loss_slack)
        complexity_limit = julia_complexity + complexity_slack
        loss_ok = python_loss <= loss_limit
        complexity_ok = python_complexity <= complexity_limit
        if loss_ok and complexity_ok:
            reason = "ok"
        else:
            failures = []
            if not loss_ok:
                failures.append(
                    f"python loss {python_loss:.6g} exceeds limit {loss_limit:.6g}"
                )
            if not complexity_ok:
                failures.append(
                    f"python complexity {python_complexity} exceeds limit {complexity_limit}"
                )
            reason = "; ".join(failures)
        checks.append(
            ParityCheck(
                case_id=case_id,
                seed=seed,
                python_loss=python_loss,
                julia_loss=julia_loss,
                loss_ratio=loss_ratio,
                python_complexity=python_complexity,
                julia_complexity=julia_complexity,
                passed=loss_ok and complexity_ok,
                reason=reason,
            )
        )
    return checks


def write_parity_csv(checks: list[ParityCheck]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow([])
    writer.writerow(
        [
            "case",
            "seed",
            "parity",
            "python_loss",
            "julia_loss",
            "loss_ratio",
            "python_complexity",
            "julia_complexity",
            "reason",
        ]
    )
    for check in checks:
        writer.writerow(
            [
                check.case_id,
                check.seed,
                "PASS" if check.passed else "FAIL",
                f"{check.python_loss:.17g}",
                f"{check.julia_loss:.17g}",
                f"{check.loss_ratio:.6g}",
                check.python_complexity,
                check.julia_complexity,
                check.reason,
            ]
        )


def julia_operator_list(operators: tuple[str, ...]) -> str:
    mapping = {
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "^": "^",
        "sin": "sin",
        "cos": "cos",
        "exp": "exp",
        "abs": "abs",
    }
    return "[" + ", ".join(mapping[op] for op in operators) + "]"


def write_julia_runner(path: Path, cases: list[Case]) -> None:
    blocks: list[str] = []
    for case in cases:
        x_rows = ["[" + ", ".join(f"{v:.17g}" for v in row) + "]" for row in case.X]
        y_vals = ", ".join(f"{v:.17g}" for v in case.y)
        blocks.append(
            f"""
if case_id == "{case.case_id}"
    X_rows = [{'; '.join(x_rows)}]
    X = permutedims(reshape(X_rows, :, {case.X.shape[1]}))
    y = [{y_vals}]
    binary_ops = {julia_operator_list(case.binary_operators)}
    unary_ops = {julia_operator_list(case.unary_operators)}
end
"""
        )

    path.write_text(
        f"""
using Pkg
julia_pkg_path = ARGS[1]
parity_env = mktempdir()
Pkg.activate(parity_env)
Pkg.develop(PackageSpec(path=julia_pkg_path))
Pkg.instantiate()
using SymbolicRegression

case_id = ARGS[2]
seed = parse(Int, ARGS[3])
niterations = parse(Int, ARGS[4])

X = nothing
y = nothing
binary_ops = nothing
unary_ops = nothing

{''.join(blocks)}

if X === nothing
    error("unknown case: " * case_id)
end

options = Options(;
    binary_operators=binary_ops,
    unary_operators=unary_ops,
    maxsize=14,
    maxdepth=8,
    population_size=30,
    populations=2,
    tournament_selection_n=10,
    ncycles_per_iteration=5,
    deterministic=true,
    seed=seed,
    verbosity=0,
    progress=false,
)

hof = equation_search(
    X,
    y;
    niterations=niterations,
    options=options,
    parallelism=:serial,
    runtests=false,
)

members = filter(m -> isfinite(m.loss), hof.members)
best = members[argmin([m.loss for m in members])]
eq = replace(string(best.tree), '\\t' => ' ', '\\n' => ' ')
println("RESULT\\t", case_id, "\\tjulia\\t", best.loss, "\\t", compute_complexity(best, options), "\\t", eq)
""",
        encoding="utf-8",
    )


def run_julia_case(
    case: Case,
    seed: int,
    niterations: int,
    julia: str,
    symbolic_regression_path: Path,
    runner: Path,
) -> dict[str, object]:
    cp = subprocess.run(
        [
            julia,
            "--project=" + str(symbolic_regression_path),
            str(runner),
            str(symbolic_regression_path),
            case.case_id,
            str(seed),
            str(niterations),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stdout + "\n" + cp.stderr)
    result_lines = [line for line in cp.stdout.splitlines() if line.startswith("RESULT\t")]
    if not result_lines:
        raise RuntimeError("Julia run did not emit RESULT line:\n" + cp.stdout + cp.stderr)
    _, case_id, backend, loss, complexity, equation = result_lines[-1].split("\t", 5)
    return {
        "case": case_id,
        "seed": seed,
        "backend": backend,
        "loss": float(loss),
        "complexity": int(complexity),
        "equation": equation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbolic-regression-path",
        type=Path,
        default=Path("/home/glompy/Desktop/ASTROCYTECH/SymbolicRegression.jl"),
    )
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list. Overrides --seed when provided.",
    )
    parser.add_argument("--niterations", type=int, default=5)
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument(
        "--max-loss-ratio",
        type=float,
        default=5.0,
        help="Fail if Python loss is worse than this multiple of Julia loss.",
    )
    parser.add_argument(
        "--absolute-loss-slack",
        type=float,
        default=0.25,
        help="Minimum absolute loss slack allowed when Julia loss is very small.",
    )
    parser.add_argument(
        "--complexity-slack",
        type=int,
        default=6,
        help="Fail if Python complexity exceeds Julia complexity by more than this.",
    )
    args = parser.parse_args()

    cases = build_cases()
    seeds = parse_seeds(args.seed, args.seeds)
    python_results = [
        run_python_case(case, seed, args.niterations)
        for seed in seeds
        for case in cases
    ]

    if args.python_only:
        write_result_csv(python_results)
        return 0
    if not args.julia:
        print("ERROR: Julia executable not found. Install Julia 1.10+ or pass --julia.", file=sys.stderr)
        return 2
    if not args.symbolic_regression_path.exists():
        print(
            f"ERROR: SymbolicRegression.jl clone not found: {args.symbolic_regression_path}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="sr_julia_parity_") as tmp:
        runner = Path(tmp) / "run_symbolic_regression_case.jl"
        write_julia_runner(runner, cases)
        julia_results = []
        for seed in seeds:
            for case in cases:
                result = run_julia_case(
                    case,
                    seed,
                    args.niterations,
                    args.julia,
                    args.symbolic_regression_path,
                    runner,
                )
                julia_results.append(result)

    write_result_csv([*python_results, *julia_results])
    checks = evaluate_parity(
        python_results,
        julia_results,
        max_loss_ratio=args.max_loss_ratio,
        absolute_loss_slack=args.absolute_loss_slack,
        complexity_slack=args.complexity_slack,
    )
    write_parity_csv(checks)
    failed = [check for check in checks if not check.passed]
    if failed:
        print(f"\nPARITY FAILED: {len(failed)} of {len(checks)} cases failed.", file=sys.stderr)
        return 1
    print(f"\nPARITY PASSED: {len(checks)} of {len(checks)} cases passed.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
