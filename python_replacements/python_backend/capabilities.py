from __future__ import annotations

from python_backend.ops import OP_ID_TO_ARITY, resolve_operator_tokens
from python_backend.errors import BackendOptionError, SR_ERR_OPT_001

CAPABILITY_LEVEL = "SR-L2"

_CAPABILITY_ORDER = [
    "SR-L0",
    "SR-L1",
    "SR-L2",
    "SR-L3",
    "SR-L4",
    "SR-L5",
]

_CAPABILITY_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(_CAPABILITY_ORDER)
}

_ALL_REGISTERED_IDS: frozenset[str] = frozenset(OP_ID_TO_ARITY.keys())

_LEVEL_BINARY_ALLOWLIST: dict[str, frozenset[str]] = {
    "SR-L0": frozenset(),
    "SR-L1": frozenset([
        "sr.arith.add_v1",
        "sr.arith.sub_v1",
        "sr.arith.mul_v1",
        "sr.math.protected_div_v1",
    ]),
    "SR-L2": frozenset([
        "sr.arith.add_v1",
        "sr.arith.sub_v1",
        "sr.arith.mul_v1",
        "sr.math.protected_div_v1",
        "sr.math.pow_v1",
        "sr.arith.less_v1",
        "sr.bool.equal_v1",
        "sr.bool.less_equal_v1",
        "sr.bool.greater_v1",
        "sr.bool.greater_equal_v1",
        "sr.bool.and_v1",
        "sr.bool.or_v1",
        "sr.bool.xor_v1",
        "sr.math.min_v1",
        "sr.math.max_v1",
        "sr.math.mod_v1",
        "sr.math.atan2_v1",
        "sr.ts.delay_v1",
        "sr.ts.sma_v1",
        "sr.ts.wma_v1",
        "sr.ts.mma_v1",
        "sr.ts.median_v1",
    ]),
}

_LEVEL_UNARY_ALLOWLIST: dict[str, frozenset[str]] = {
    "SR-L0": frozenset(),
    "SR-L1": frozenset([
        "sr.math.sin_v1",
        "sr.math.cos_v1",
    ]),
    "SR-L2": frozenset([
        "sr.math.sin_v1",
        "sr.math.cos_v1",
        "sr.math.abs_v1",
        "sr.math.safe_log_v1",
        "sr.math.exp_v1",
        "sr.math.tan_v1",
        "sr.math.factorial_v1",
        "sr.math.sqrt_v1",
        "sr.math.logistic_v1",
        "sr.math.step_v1",
        "sr.math.sign_v1",
        "sr.math.gauss_v1",
        "sr.math.tanh_v1",
        "sr.math.erf_v1",
        "sr.math.erfc_v1",
        "sr.bool.not_v1",
        "sr.math.floor_v1",
        "sr.math.ceil_v1",
        "sr.math.round_v1",
        "sr.math.asin_v1",
        "sr.math.acos_v1",
        "sr.math.atan_v1",
        "sr.arith.neg_v1",
    ]),
}


def _capability_le(a: str, b: str) -> bool:
    if a not in _CAPABILITY_INDEX:
        raise BackendOptionError(
            SR_ERR_OPT_001,
            f"unknown capability level {a!r}",
        )
    if b not in _CAPABILITY_INDEX:
        raise BackendOptionError(
            SR_ERR_OPT_001,
            f"unknown capability level {b!r}",
        )
    return _CAPABILITY_INDEX[a] <= _CAPABILITY_INDEX[b]


def _supported_binary_ids(level: str) -> frozenset[str]:
    if level in _LEVEL_BINARY_ALLOWLIST:
        return _LEVEL_BINARY_ALLOWLIST[level]
    return _ALL_REGISTERED_IDS


def _supported_unary_ids(level: str) -> frozenset[str]:
    if level in _LEVEL_UNARY_ALLOWLIST:
        return _LEVEL_UNARY_ALLOWLIST[level]
    return _ALL_REGISTERED_IDS


def assert_operators_supported(
    binary_ids: list[str], unary_ids: list[str],
) -> None:
    for oid in binary_ids:
        if oid not in _supported_binary_ids(CAPABILITY_LEVEL):
            raise BackendOptionError(
                SR_ERR_OPT_001,
                f"operator {oid!r} not supported at capability level "
                f"{CAPABILITY_LEVEL}",
            )
    for oid in unary_ids:
        if oid not in _supported_unary_ids(CAPABILITY_LEVEL):
            raise BackendOptionError(
                SR_ERR_OPT_001,
                f"operator {oid!r} not supported at capability level "
                f"{CAPABILITY_LEVEL}",
            )


def assert_capability_level_sufficient(required_level: str) -> None:
    if not _capability_le(required_level, CAPABILITY_LEVEL):
        raise BackendOptionError(
            SR_ERR_OPT_001,
            f"problem requires capability level {required_level}, "
            f"but backend is at {CAPABILITY_LEVEL}",
        )


def assert_problem_supported(problem_spec: dict) -> None:
    required = problem_spec.get("capability_level_required", "SR-L0")
    assert_capability_level_sufficient(required)

    ops = problem_spec.get("operators", {})
    binary_ids = resolve_operator_tokens(ops.get("binary", []))
    unary_ids = resolve_operator_tokens(ops.get("unary", []))

    assert_operators_supported(binary_ids, unary_ids)
