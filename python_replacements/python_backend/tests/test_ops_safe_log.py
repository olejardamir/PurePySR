from __future__ import annotations

import numpy as np

from python_backend.ops import _safe_log
from python_backend.policy import EPS_DENOM


def test_safe_log_zero():
    result = _safe_log(np.array([0.0]))[0]
    assert result == np.log(EPS_DENOM), (
        f"safe_log(0) = {result}, expected {np.log(EPS_DENOM)}"
    )


def test_safe_log_symmetric_negative():
    pos = _safe_log(np.array([1.0]))[0]
    neg = _safe_log(np.array([-1.0]))[0]
    assert pos == neg, (
        f"safe_log(-1) = {neg} != safe_log(1) = {pos}"
    )


def test_safe_log_finite_for_all_finite_inputs():
    inputs = np.linspace(-100.0, 100.0, 501)
    outputs = _safe_log(inputs)
    assert np.all(np.isfinite(outputs)), (
        f"non-finite outputs at indices: "
        f"{np.where(~np.isfinite(outputs))[0]}"
    )


def test_safe_log_positive_input_matches_log():
    x = np.array([2.0, 10.0, 100.0])
    expected = np.log(x + EPS_DENOM)
    result = _safe_log(x)
    assert np.allclose(result, expected), (
        f"safe_log(x) = {result}, expected {expected}"
    )
