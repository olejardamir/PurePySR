from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import scipy.special as _sp_special

from python_backend.errors import BackendOptionError, SR_ERR_OPT_001

TOKEN_TO_OP_ID: dict[str, str] = {
    "+": "sr.arith.add_v1",
    "add": "sr.arith.add_v1",
    "-": "sr.arith.sub_v1",
    "sub": "sr.arith.sub_v1",
    "*": "sr.arith.mul_v1",
    "mul": "sr.arith.mul_v1",
    "/": "sr.math.protected_div_v1",
    "div": "sr.math.protected_div_v1",
    "^": "sr.math.pow_v1",
    "pow": "sr.math.pow_v1",
    "sin": "sr.math.sin_v1",
    "cos": "sr.math.cos_v1",
    "tan": "sr.math.tan_v1",
    "abs": "sr.math.abs_v1",
    "log": "sr.math.safe_log_v1",
    "safe_log": "sr.math.safe_log_v1",
    "exp": "sr.math.exp_v1",
    "factorial": "sr.math.factorial_v1",
    "sqrt": "sr.math.sqrt_v1",
    "logistic": "sr.math.logistic_v1",
    "sigmoid": "sr.math.logistic_v1",
    "step": "sr.math.step_v1",
    "sgn": "sr.math.sign_v1",
    "sign": "sr.math.sign_v1",
    "gauss": "sr.math.gauss_v1",
    "tanh": "sr.math.tanh_v1",
    "erf": "sr.math.erf_v1",
    "erfc": "sr.math.erfc_v1",
    "equal": "sr.bool.equal_v1",
    "<": "sr.arith.less_v1",
    "less": "sr.arith.less_v1",
    "less_or_equal": "sr.bool.less_equal_v1",
    "greater": "sr.bool.greater_v1",
    "greater_or_equal": "sr.bool.greater_equal_v1",
    "and": "sr.bool.and_v1",
    "or": "sr.bool.or_v1",
    "xor": "sr.bool.xor_v1",
    "not": "sr.bool.not_v1",
    "min": "sr.math.min_v1",
    "max": "sr.math.max_v1",
    "mod": "sr.math.mod_v1",
    "floor": "sr.math.floor_v1",
    "ceil": "sr.math.ceil_v1",
    "round": "sr.math.round_v1",
    "asin": "sr.math.asin_v1",
    "acos": "sr.math.acos_v1",
    "atan": "sr.math.atan_v1",
    "atan2": "sr.math.atan2_v1",
    "neg": "sr.arith.neg_v1",
    "delay": "sr.ts.delay_v1",
    "sma": "sr.ts.sma_v1",
    "wma": "sr.ts.wma_v1",
    "mma": "sr.ts.mma_v1",
    "median": "sr.ts.median_v1",
    "moving_median": "sr.ts.median_v1",
}

OP_ID_TO_ARITY: dict[str, int] = {
    "sr.arith.add_v1": 2,
    "sr.arith.sub_v1": 2,
    "sr.arith.mul_v1": 2,
    "sr.math.protected_div_v1": 2,
    "sr.math.pow_v1": 2,
    "sr.math.sin_v1": 1,
    "sr.math.cos_v1": 1,
    "sr.math.tan_v1": 1,
    "sr.math.abs_v1": 1,
    "sr.math.safe_log_v1": 1,
    "sr.math.exp_v1": 1,
    "sr.math.factorial_v1": 1,
    "sr.math.sqrt_v1": 1,
    "sr.math.logistic_v1": 1,
    "sr.math.step_v1": 1,
    "sr.math.sign_v1": 1,
    "sr.math.gauss_v1": 1,
    "sr.math.tanh_v1": 1,
    "sr.math.erf_v1": 1,
    "sr.math.erfc_v1": 1,
    "sr.bool.equal_v1": 2,
    "sr.arith.less_v1": 2,
    "sr.bool.less_equal_v1": 2,
    "sr.bool.greater_v1": 2,
    "sr.bool.greater_equal_v1": 2,
    "sr.bool.and_v1": 2,
    "sr.bool.or_v1": 2,
    "sr.bool.xor_v1": 2,
    "sr.bool.not_v1": 1,
    "sr.math.min_v1": 2,
    "sr.math.max_v1": 2,
    "sr.math.mod_v1": 2,
    "sr.math.floor_v1": 1,
    "sr.math.ceil_v1": 1,
    "sr.math.round_v1": 1,
    "sr.math.asin_v1": 1,
    "sr.math.acos_v1": 1,
    "sr.math.atan_v1": 1,
    "sr.math.atan2_v1": 2,
    "sr.arith.neg_v1": 1,
    "sr.ts.delay_v1": 2,
    "sr.ts.sma_v1": 2,
    "sr.ts.wma_v1": 2,
    "sr.ts.mma_v1": 2,
    "sr.ts.median_v1": 2,
}

OP_ID_TO_TOKEN: dict[str, str] = {}
for _token, _op_id in TOKEN_TO_OP_ID.items():
    OP_ID_TO_TOKEN.setdefault(_op_id, _token)
OP_ID_TO_TOKEN.update({
    "sr.arith.add_v1": "+",
    "sr.arith.sub_v1": "-",
    "sr.arith.mul_v1": "*",
    "sr.math.protected_div_v1": "/",
    "sr.math.pow_v1": "^",
    "sr.math.safe_log_v1": "safe_log",
    "sr.math.logistic_v1": "logistic",
    "sr.math.sign_v1": "sgn",
    "sr.arith.less_v1": "<",
})

OP_NAMESPACE: dict[str, str] = {
    "sr.arith.add_v1": "sr.arith",
    "sr.arith.sub_v1": "sr.arith",
    "sr.arith.mul_v1": "sr.arith",
    "sr.math.protected_div_v1": "sr.math",
    "sr.math.pow_v1": "sr.math",
    "sr.math.sin_v1": "sr.math",
    "sr.math.cos_v1": "sr.math",
    "sr.math.tan_v1": "sr.math",
    "sr.math.abs_v1": "sr.math",
    "sr.math.safe_log_v1": "sr.math",
    "sr.math.exp_v1": "sr.math",
    "sr.math.factorial_v1": "sr.math",
    "sr.math.sqrt_v1": "sr.math",
    "sr.math.logistic_v1": "sr.math",
    "sr.math.step_v1": "sr.math",
    "sr.math.sign_v1": "sr.math",
    "sr.math.gauss_v1": "sr.math",
    "sr.math.tanh_v1": "sr.math",
    "sr.math.erf_v1": "sr.math",
    "sr.math.erfc_v1": "sr.math",
    "sr.arith.less_v1": "sr.arith",
    "sr.bool.equal_v1": "sr.bool",
    "sr.bool.less_equal_v1": "sr.bool",
    "sr.bool.greater_v1": "sr.bool",
    "sr.bool.greater_equal_v1": "sr.bool",
    "sr.bool.and_v1": "sr.bool",
    "sr.bool.or_v1": "sr.bool",
    "sr.bool.xor_v1": "sr.bool",
    "sr.bool.not_v1": "sr.bool",
    "sr.math.min_v1": "sr.math",
    "sr.math.max_v1": "sr.math",
    "sr.math.mod_v1": "sr.math",
    "sr.math.floor_v1": "sr.math",
    "sr.math.ceil_v1": "sr.math",
    "sr.math.round_v1": "sr.math",
    "sr.math.asin_v1": "sr.math",
    "sr.math.acos_v1": "sr.math",
    "sr.math.atan_v1": "sr.math",
    "sr.math.atan2_v1": "sr.math",
    "sr.arith.neg_v1": "sr.arith",
    "sr.ts.delay_v1": "sr.ts",
    "sr.ts.sma_v1": "sr.ts",
    "sr.ts.wma_v1": "sr.ts",
    "sr.ts.mma_v1": "sr.ts",
    "sr.ts.median_v1": "sr.ts",
}


def _protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) < 1e-8, 1.0, x / y)


def _protected_div_out(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(x, y, out=out)
        out[np.abs(y) < 1e-8] = 1.0


def _safe_log(x: np.ndarray) -> np.ndarray:
    from python_backend.policy import EPS_DENOM

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.abs(x) + EPS_DENOM)


def _safe_log_out(x: np.ndarray, out: np.ndarray) -> None:
    from python_backend.policy import EPS_DENOM

    with np.errstate(divide="ignore", invalid="ignore"):
        np.abs(x, out=out)
        out += EPS_DENOM
        np.log(out, out=out)


def _safe_pow(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(x < 0, np.nan, np.power(x, y))
        return np.where(np.isfinite(result), result, np.nan)


def _safe_pow_out(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        neg = x < 0
        np.power(x, y, out=out)
        out[neg] = np.nan
        out[~np.isfinite(out)] = np.nan


def _less_v1_out(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    np.less(x, y, out=out)  # writes bool cast to float64 (1.0/0.0)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.where(x >= 0.0, x, np.nan))


def _safe_sqrt_out(x: np.ndarray, out: np.ndarray) -> None:
    with np.errstate(invalid="ignore"):
        np.sqrt(np.where(x >= 0.0, x, np.nan), out=out)


def _gauss(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-(x * x))


def _gauss_out(x: np.ndarray, out: np.ndarray) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        np.multiply(x, x, out=out)
        np.negative(out, out=out)
        np.exp(out, out=out)


def _safe_mod(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) < 1e-8, 0.0, np.mod(x, y))


def _safe_mod_out(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        np.mod(x, y, out=out)
        out[np.abs(y) < 1e-8] = 0.0


def _safe_asin(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.arcsin(np.clip(x, -1.0, 1.0))


def _safe_acos(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.arccos(np.clip(x, -1.0, 1.0))


def _bool_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where((x > 0.0) & (y > 0.0), 1.0, 0.0)


def _bool_or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where((x > 0.0) | (y > 0.0), 1.0, 0.0)


def _bool_xor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(((x > 0.0) ^ (y > 0.0)), 1.0, 0.0)


def _equal_out(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.isclose(x, y)


def _window_size(window_arg: np.ndarray) -> int:
    values = np.asarray(window_arg, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1
    return max(1, int(round(abs(float(finite[0])))))


def _delay(x: np.ndarray, window_arg: np.ndarray) -> np.ndarray:
    n = _window_size(window_arg)
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    if values.size == 0:
        return out
    out[:n] = values[0]
    if n < values.size:
        out[n:] = values[:-n]
    return out


def _rolling_apply(x: np.ndarray, window_arg: np.ndarray, reducer: Callable[[np.ndarray], float]) -> np.ndarray:
    n = _window_size(window_arg)
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    for i in range(values.size):
        start = max(0, i - n + 1)
        out[i] = reducer(values[start : i + 1])
    return out


def _sma(x: np.ndarray, window_arg: np.ndarray) -> np.ndarray:
    return _rolling_apply(x, window_arg, lambda v: float(np.mean(v)))


def _wma(x: np.ndarray, window_arg: np.ndarray) -> np.ndarray:
    def _weighted(values: np.ndarray) -> float:
        weights = np.arange(1, values.size + 1, dtype=np.float64)
        return float(np.average(values, weights=weights))

    return _rolling_apply(x, window_arg, _weighted)


def _mma(x: np.ndarray, window_arg: np.ndarray) -> np.ndarray:
    n = _window_size(window_arg)
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    if values.size == 0:
        return out
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def _moving_median(x: np.ndarray, window_arg: np.ndarray) -> np.ndarray:
    return _rolling_apply(x, window_arg, lambda v: float(np.median(v)))


def _copy_out(fn: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    out[:] = fn(x, y)


OP_ID_TO_FN: dict[str, Callable[..., np.ndarray]] = {
    "sr.arith.add_v1": lambda x, y: x + y,
    "sr.arith.sub_v1": lambda x, y: x - y,
    "sr.arith.mul_v1": lambda x, y: x * y,
    "sr.math.protected_div_v1": _protected_div,
    "sr.math.pow_v1": _safe_pow,
    "sr.math.sin_v1": np.sin,
    "sr.math.cos_v1": np.cos,
    "sr.math.tan_v1": np.tan,
    "sr.math.abs_v1": np.abs,
    "sr.math.safe_log_v1": _safe_log,
    "sr.math.exp_v1": np.exp,
    "sr.math.factorial_v1": lambda x: _sp_special.gamma(x + 1.0),
    "sr.math.sqrt_v1": _safe_sqrt,
    "sr.math.logistic_v1": _sp_special.expit,
    "sr.math.step_v1": lambda x: np.where(x > 0.0, 1.0, 0.0),
    "sr.math.sign_v1": np.sign,
    "sr.math.gauss_v1": _gauss,
    "sr.math.tanh_v1": np.tanh,
    "sr.math.erf_v1": _sp_special.erf,
    "sr.math.erfc_v1": _sp_special.erfc,
    "sr.bool.equal_v1": lambda x, y: np.where(np.isclose(x, y), 1.0, 0.0),
    "sr.arith.less_v1": lambda x, y: np.where(x < y, 1.0, 0.0),
    "sr.bool.less_equal_v1": lambda x, y: np.where(x <= y, 1.0, 0.0),
    "sr.bool.greater_v1": lambda x, y: np.where(x > y, 1.0, 0.0),
    "sr.bool.greater_equal_v1": lambda x, y: np.where(x >= y, 1.0, 0.0),
    "sr.bool.and_v1": _bool_and,
    "sr.bool.or_v1": _bool_or,
    "sr.bool.xor_v1": _bool_xor,
    "sr.bool.not_v1": lambda x: np.where(x > 0.0, 0.0, 1.0),
    "sr.math.min_v1": np.minimum,
    "sr.math.max_v1": np.maximum,
    "sr.math.mod_v1": _safe_mod,
    "sr.math.floor_v1": np.floor,
    "sr.math.ceil_v1": np.ceil,
    "sr.math.round_v1": np.rint,
    "sr.math.asin_v1": _safe_asin,
    "sr.math.acos_v1": _safe_acos,
    "sr.math.atan_v1": np.arctan,
    "sr.math.atan2_v1": np.arctan2,
    "sr.arith.neg_v1": np.negative,
    "sr.ts.delay_v1": _delay,
    "sr.ts.sma_v1": _sma,
    "sr.ts.wma_v1": _wma,
    "sr.ts.mma_v1": _mma,
    "sr.ts.median_v1": _moving_median,
}

OP_ID_TO_FN_OUT: dict[str, Callable[..., None]] = {
    "sr.arith.add_v1": lambda x, y, out: np.add(x, y, out=out),
    "sr.arith.sub_v1": lambda x, y, out: np.subtract(x, y, out=out),
    "sr.arith.mul_v1": lambda x, y, out: np.multiply(x, y, out=out),
    "sr.math.protected_div_v1": _protected_div_out,
    "sr.math.pow_v1": _safe_pow_out,
    "sr.math.sin_v1": lambda x, out: np.sin(x, out=out),
    "sr.math.cos_v1": lambda x, out: np.cos(x, out=out),
    "sr.math.tan_v1": lambda x, out: np.tan(x, out=out),
    "sr.math.abs_v1": lambda x, out: np.abs(x, out=out),
    "sr.math.safe_log_v1": _safe_log_out,
    "sr.math.exp_v1": lambda x, out: np.exp(x, out=out),
    "sr.math.factorial_v1": lambda x, out: _sp_special.gamma(x + 1.0, out=out),
    "sr.math.sqrt_v1": _safe_sqrt_out,
    "sr.math.logistic_v1": lambda x, out: _sp_special.expit(x, out=out),
    "sr.math.step_v1": lambda x, out: np.greater(x, 0.0, out=out),
    "sr.math.sign_v1": lambda x, out: np.sign(x, out=out),
    "sr.math.gauss_v1": _gauss_out,
    "sr.math.tanh_v1": lambda x, out: np.tanh(x, out=out),
    "sr.math.erf_v1": lambda x, out: _sp_special.erf(x, out=out),
    "sr.math.erfc_v1": lambda x, out: _sp_special.erfc(x, out=out),
    "sr.bool.equal_v1": _equal_out,
    "sr.arith.less_v1": _less_v1_out,
    "sr.bool.less_equal_v1": lambda x, y, out: np.less_equal(x, y, out=out),
    "sr.bool.greater_v1": lambda x, y, out: np.greater(x, y, out=out),
    "sr.bool.greater_equal_v1": lambda x, y, out: np.greater_equal(x, y, out=out),
    "sr.bool.and_v1": lambda x, y, out: np.logical_and(x > 0.0, y > 0.0, out=out),
    "sr.bool.or_v1": lambda x, y, out: np.logical_or(x > 0.0, y > 0.0, out=out),
    "sr.bool.xor_v1": lambda x, y, out: np.logical_xor(x > 0.0, y > 0.0, out=out),
    "sr.bool.not_v1": lambda x, out: np.less_equal(x, 0.0, out=out),
    "sr.math.min_v1": lambda x, y, out: np.minimum(x, y, out=out),
    "sr.math.max_v1": lambda x, y, out: np.maximum(x, y, out=out),
    "sr.math.mod_v1": _safe_mod_out,
    "sr.math.floor_v1": lambda x, out: np.floor(x, out=out),
    "sr.math.ceil_v1": lambda x, out: np.ceil(x, out=out),
    "sr.math.round_v1": lambda x, out: np.rint(x, out=out),
    "sr.math.asin_v1": lambda x, out: np.arcsin(np.clip(x, -1.0, 1.0), out=out),
    "sr.math.acos_v1": lambda x, out: np.arccos(np.clip(x, -1.0, 1.0), out=out),
    "sr.math.atan_v1": lambda x, out: np.arctan(x, out=out),
    "sr.math.atan2_v1": lambda x, y, out: np.arctan2(x, y, out=out),
    "sr.arith.neg_v1": lambda x, out: np.negative(x, out=out),
    "sr.ts.delay_v1": lambda x, y, out: _copy_out(_delay, x, y, out),
    "sr.ts.sma_v1": lambda x, y, out: _copy_out(_sma, x, y, out),
    "sr.ts.wma_v1": lambda x, y, out: _copy_out(_wma, x, y, out),
    "sr.ts.mma_v1": lambda x, y, out: _copy_out(_mma, x, y, out),
    "sr.ts.median_v1": lambda x, y, out: _copy_out(_moving_median, x, y, out),
}


def resolve_operator_tokens(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return []
    ids: list[str] = []
    for tok in tokens:
        oid = TOKEN_TO_OP_ID.get(tok)
        if oid is None:
            raise BackendOptionError(
                SR_ERR_OPT_001,
                f"unsupported operator token {tok!r}",
            )
        ids.append(oid)
    return ids


def operator_manifest_bytes() -> bytes:
    path = (
        pathlib.Path(__file__).resolve().parent
        / "data" / "sr-operator-registry.yaml"
    )
    if path.exists():
        return path.read_bytes()
    return b"".join(
        sorted(f"{k}={v}\n".encode() for k, v in TOKEN_TO_OP_ID.items())
    )
