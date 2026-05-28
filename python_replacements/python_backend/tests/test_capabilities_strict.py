from __future__ import annotations

import pytest

from python_backend.capabilities import (
    CAPABILITY_LEVEL,
    assert_capability_level_sufficient,
    assert_operators_supported,
    assert_problem_supported,
)
from python_backend.errors import BackendOptionError, SR_ERR_OPT_001


def test_unknown_required_level_raises():
    with pytest.raises(BackendOptionError) as excinfo:
        assert_capability_level_sufficient("SR-L99")
    assert SR_ERR_OPT_001 in str(excinfo.value)
    assert "SR-L99" in str(excinfo.value)


def test_unknown_backend_level_raises(monkeypatch):
    monkeypatch.setattr(
        "python_backend.capabilities.CAPABILITY_LEVEL", "SR-L99",
    )
    with pytest.raises(BackendOptionError) as excinfo:
        assert_capability_level_sufficient("SR-L1")
    assert SR_ERR_OPT_001 in str(excinfo.value)
    assert "SR-L99" in str(excinfo.value)


def test_unknown_both_levels_raises(monkeypatch):
    monkeypatch.setattr(
        "python_backend.capabilities.CAPABILITY_LEVEL", "SR-L99",
    )
    with pytest.raises(BackendOptionError) as excinfo:
        assert_capability_level_sufficient("SR-L99")
    assert SR_ERR_OPT_001 in str(excinfo.value)


def test_unknown_operator_token_via_problem_raises():
    problem = {
        "problem_id": "FAKE",
        "operators": {"binary": ["+"], "unary": ["bogus"]},
    }
    with pytest.raises(BackendOptionError) as excinfo:
        assert_problem_supported(problem)
    assert SR_ERR_OPT_001 in str(excinfo.value)
    assert "bogus" in str(excinfo.value)


def test_known_levels_pass():
    assert_capability_level_sufficient("SR-L0")
    assert_capability_level_sufficient("SR-L1")


def test_assert_operators_supported_unsupported_raises():
    with pytest.raises(BackendOptionError) as excinfo:
        assert_operators_supported([], ["sr.math.unregistered_v1"])
    assert SR_ERR_OPT_001 in str(excinfo.value)
    assert "unregistered" in str(excinfo.value)


def test_assert_operators_supported_known_pass():
    assert_operators_supported(
        ["sr.arith.add_v1", "sr.arith.mul_v1"], [],
    )
