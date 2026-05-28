from __future__ import annotations

import re
from typing import Callable

import numpy as np

LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) ** 2


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true)


def _huber(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> np.ndarray:
    diff = y_pred - y_true
    return np.where(np.abs(diff) <= delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta))


def _log_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


_NAMED_LOSSES: dict[str, LossFn] = {
    "mse": _mse,
    "mae": _mae,
    "huber": _huber,
    "log_loss": _log_loss,
}

_LAMBDA_PATTERN = re.compile(r"^\s*(?:\([^)]*\)\s*->\s*)?(.+)$", re.DOTALL)


def resolve_loss(name_or_expr: str | None) -> LossFn:
    if name_or_expr is None:
        return _mse
    lower = name_or_expr.strip().lower()
    if lower in _NAMED_LOSSES:
        return _NAMED_LOSSES[lower]
    return _compile_expression(name_or_expr)


def _compile_expression(expr: str) -> LossFn:
    import re

    m = _LAMBDA_PATTERN.match(expr)
    if m is None:
        raise ValueError(f"cannot parse loss expression: {expr!r}")
    body = m.group(1).strip()
    body = body.replace("^", "**")

    _NP_FNS = ["abs", "log", "exp", "sqrt", "sin", "cos",
               "tanh", "max", "min", "sign"]
    for fn in _NP_FNS:
        body = re.sub(rf"(?<![.\w]){fn}\(", f"np.{fn}(", body)
    safe_dict: dict[str, object] = {
        "y_pred": None,
        "y_true": None,
        "np": np,
    }

    def loss_fn(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        safe_dict["y_pred"] = y_pred
        safe_dict["y_true"] = y_true
        return eval(body, {"__builtins__": {}}, safe_dict)

    return loss_fn
