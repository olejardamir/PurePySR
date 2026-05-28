from __future__ import annotations

import numpy as np
import pytest

from python_backend.loss_functions import resolve_loss


def test_resolve_none_returns_mse():
    fn = resolve_loss(None)
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = (y_pred - y_true) ** 2
    np.testing.assert_array_equal(result, expected)


def test_resolve_mse_named():
    fn = resolve_loss("mse")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = (y_pred - y_true) ** 2
    np.testing.assert_array_equal(result, expected)


def test_resolve_mae_named():
    fn = resolve_loss("mae")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = np.abs(y_pred - y_true)
    np.testing.assert_array_equal(result, expected)


def test_resolve_mae_case_insensitive():
    fn = resolve_loss("MAE")
    y_pred = np.array([1.0, 2.0])
    y_true = np.array([1.5, 2.5])
    result = fn(y_pred, y_true)
    expected = np.abs(y_pred - y_true)
    np.testing.assert_array_equal(result, expected)


def test_resolve_lambda_expression():
    fn = resolve_loss("(y_pred, y_true) -> abs(y_pred - y_true)")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = np.abs(y_pred - y_true)
    np.testing.assert_array_equal(result, expected)


def test_resolve_bare_expression():
    fn = resolve_loss("(y_pred - y_true)^2")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = (y_pred - y_true) ** 2
    np.testing.assert_array_equal(result, expected)


def test_resolve_huber():
    fn = resolve_loss("huber")
    y_pred = np.array([0.0, 2.0, 5.0])
    y_true = np.array([1.0, 1.0, 1.0])
    result = fn(y_pred, y_true)
    diff = y_pred - y_true
    expected = np.where(np.abs(diff) <= 1.0, 0.5 * diff ** 2, 1.0 * (np.abs(diff) - 0.5))
    np.testing.assert_array_equal(result, expected)


def test_resolve_log_loss():
    fn = resolve_loss("log_loss")
    y_pred = np.array([0.1, 0.5, 0.9])
    y_true = np.array([0.0, 1.0, 1.0])
    result = fn(y_pred, y_true)
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    expected = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    np.testing.assert_array_almost_equal(result, expected)


def test_custom_expression_with_np():
    fn = resolve_loss("sqrt((y_pred - y_true)**2)")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    result = fn(y_pred, y_true)
    expected = np.sqrt((y_pred - y_true) ** 2)
    np.testing.assert_array_equal(result, expected)


def test_resolve_invalid_expression_raises():
    with pytest.raises(Exception):
        resolve_loss("")("", "")


def test_resolve_full_lambda_syntax():
    fn = resolve_loss("(y_true, y_pred) -> (y_pred - y_true)^2")
    y_pred = np.array([2.0, 3.0])
    y_true = np.array([1.0, 1.0])
    result = fn(y_pred, y_true)
    expected = (y_pred - y_true) ** 2
    np.testing.assert_array_equal(result, expected)
