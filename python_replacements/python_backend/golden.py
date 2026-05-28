from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import yaml


def load_problems() -> list[dict[str, Any]]:
    path = (
        pathlib.Path(__file__).resolve().parent
        / "data" / "sr-golden-problems.yaml"
    )
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["problems"]


def load_problem(problem_id: str) -> dict[str, Any]:
    for p in load_problems():
        if p["problem_id"] == problem_id:
            return p
    raise KeyError(f"golden problem {problem_id!r} not found")


def load_default_search_options() -> dict[str, Any]:
    path = (
        pathlib.Path(__file__).resolve().parent
        / "data" / "sr-golden-problems.yaml"
    )
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("default_search_options", {})


def load_search_options(problem: dict[str, Any]) -> dict[str, Any]:
    defaults = load_default_search_options()
    overrides = problem.get("search_options", {})
    return {**defaults, **overrides}


def generate_dataset(
    problem: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gen = problem["generator"]
    if rng is None:
        rng = np.random.default_rng(gen["seed"])

    n_samples = gen["n_samples"]
    n_features = gen["n_features"]
    x_lo, x_hi = gen["x_range"]
    noise_std = gen.get("noise_std", 0.0)

    X = rng.uniform(x_lo, x_hi, (n_samples, n_features)).astype(np.float64)
    y = _eval_target(problem["target"]["expression"], X)

    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, n_samples)

    return X, y


def load_problem_capability_level(problem: dict) -> str:
    return problem.get("capability_level_required", "SR-L0")


def load_problem_operator_ids(
    problem: dict,
) -> tuple[list[str], list[str]]:
    from python_backend.ops import resolve_operator_tokens

    ops = problem.get("operators", {})
    binary_ids = resolve_operator_tokens(ops.get("binary", []))
    unary_ids = resolve_operator_tokens(ops.get("unary", []))
    return binary_ids, unary_ids


def _eval_target(expr_str: str, X: np.ndarray) -> np.ndarray:
    import numpy as np
    from python_backend.policy import EPS_DENOM

    env: dict[str, object] = {}
    for i in range(X.shape[1]):
        env[f"x{i}"] = X[:, i]
    env["sin"] = np.sin
    env["cos"] = np.cos
    env["abs"] = np.abs
    env["safe_log"] = lambda x: np.log(np.abs(x) + EPS_DENOM)
    env["c"] = 1.0
    s = expr_str.replace("^", "**")
    return eval(s, {"__builtins__": {}}, env)  # noqa: S307
