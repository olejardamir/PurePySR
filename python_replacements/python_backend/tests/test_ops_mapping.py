from __future__ import annotations

import pytest

from python_backend.ops import TOKEN_TO_OP_ID, OP_ID_TO_TOKEN, resolve_operator_tokens
from python_backend.errors import BackendOptionError, SR_ERR_OPT_001


def test_known_binary_mappings():
    assert TOKEN_TO_OP_ID["+"] == "sr.arith.add_v1"
    assert TOKEN_TO_OP_ID["-"] == "sr.arith.sub_v1"
    assert TOKEN_TO_OP_ID["*"] == "sr.arith.mul_v1"
    assert TOKEN_TO_OP_ID["/"] == "sr.math.protected_div_v1"


def test_known_unary_mappings():
    assert TOKEN_TO_OP_ID["sin"] == "sr.math.sin_v1"
    assert TOKEN_TO_OP_ID["cos"] == "sr.math.cos_v1"
    assert TOKEN_TO_OP_ID["abs"] == "sr.math.abs_v1"
    assert TOKEN_TO_OP_ID["safe_log"] == "sr.math.safe_log_v1"
    assert TOKEN_TO_OP_ID["exp"] == "sr.math.exp_v1"


def test_reverse_mapping_is_consistent():
    for token, oid in TOKEN_TO_OP_ID.items():
        canonical = OP_ID_TO_TOKEN[oid]
        assert TOKEN_TO_OP_ID[canonical] == oid, (
            f"canonical reverse mapping failed for {token!r} <-> {oid!r}"
        )


def test_unknown_token_raises():
    with pytest.raises(BackendOptionError) as excinfo:
        resolve_operator_tokens(["foo"])
    assert SR_ERR_OPT_001 in str(excinfo.value)


def test_unknown_token_in_empty_list():
    assert resolve_operator_tokens([]) == []
    assert resolve_operator_tokens(None) == []


def test_exp_overflow_returns_inf():
    """exp on large input produces inf, not crash (inf_policy=invalid_candidate)."""
    import numpy as np
    from python_backend.ops import OP_ID_TO_FN
    fn = OP_ID_TO_FN["sr.math.exp_v1"]
    x = np.array([1.0, 10.0, 100.0, 800.0])
    result = fn(x)
    assert np.any(~np.isfinite(result)), "exp(800) should overflow to inf"
    assert not np.any(np.isnan(result)), "exp should never produce NaN"

def test_known_tokens_resolve_successfully():
    ids = resolve_operator_tokens(["+", "*", "sin", "cos", "exp", "safe_log"])
    assert ids == [
        "sr.arith.add_v1",
        "sr.arith.mul_v1",
        "sr.math.sin_v1",
        "sr.math.cos_v1",
        "sr.math.exp_v1",
        "sr.math.safe_log_v1",
    ]
