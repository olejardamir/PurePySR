from __future__ import annotations

import numpy as np
import scipy.special as _sp_special

from python_backend.eval import compute_loss
from python_backend.expr import ConstNode, Node, OpNode, VarNode
from python_backend.ops import OP_ID_TO_FN
from python_backend.policy import EPS_DENOM


def _collect_consts(node: Node) -> list[ConstNode]:
    if isinstance(node, ConstNode):
        return [node]
    if isinstance(node, OpNode):
        consts: list[ConstNode] = []
        for c in node.children:
            consts.extend(_collect_consts(c))
        return consts
    return []


def _evaluate_with_gradient(
    node: Node,
    X: np.ndarray,
    consts: list[ConstNode],
) -> tuple[np.ndarray, np.ndarray]:
    n_consts = len(consts)
    n_samples = X.shape[0]

    if isinstance(node, VarNode):
        return (
            X[:, node.index].astype(np.float64),
            np.zeros((n_consts, n_samples), dtype=np.float64),
        )

    if isinstance(node, ConstNode):
        val = np.full(n_samples, node.value, dtype=np.float64)
        grad = np.zeros((n_consts, n_samples), dtype=np.float64)
        for i, c in enumerate(consts):
            if c is node:
                grad[i, :] = 1.0
        return val, grad

    if isinstance(node, OpNode):
        child_results = [_evaluate_with_gradient(c, X, consts) for c in node.children]
        child_vals = [r[0] for r in child_results]
        child_grads = [r[1] for r in child_results]

        val = OP_ID_TO_FN[node.op_id](*child_vals)

        if node.op_id == "sr.arith.add_v1":
            grad = child_grads[0] + child_grads[1]
        elif node.op_id == "sr.arith.sub_v1":
            grad = child_grads[0] - child_grads[1]
        elif node.op_id == "sr.arith.mul_v1":
            a, b = child_vals
            da, db = child_grads
            grad = da * b + a * db
        elif node.op_id == "sr.math.protected_div_v1":
            a, b = child_vals
            da, db = child_grads
            inv_b2 = 1.0 / np.maximum(b * b, 1e-16)
            grad = (da * b - a * db) * inv_b2
        elif node.op_id == "sr.math.pow_v1":
            a, b = child_vals
            da, db = child_grads
            valid = (a > 0) & np.isfinite(val)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                log_a_safe = np.log(np.maximum(a, 1e-300))
                a_pow_b_minus_1 = np.exp((b - 1) * log_a_safe)
                grad = np.where(
                    valid,
                    b * a_pow_b_minus_1 * da + val * log_a_safe * db,
                    0.0,
                )
            grad = np.where(np.isfinite(grad), grad, 0.0)
        elif node.op_id == "sr.math.sin_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = np.cos(a) * da
        elif node.op_id == "sr.math.cos_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = -np.sin(a) * da
        elif node.op_id == "sr.math.tan_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = (1.0 / np.cos(a) ** 2) * da
        elif node.op_id == "sr.math.abs_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = np.sign(a) * da
        elif node.op_id == "sr.math.exp_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = val * da
        elif node.op_id == "sr.math.factorial_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = val * _sp_special.digamma(a + 1.0) * da
        elif node.op_id == "sr.math.sqrt_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = np.where(a > 1e-12, 0.5 / np.sqrt(a) * da, 0.0)
        elif node.op_id == "sr.math.logistic_v1":
            da = child_grads[0]
            grad = val * (1.0 - val) * da
        elif node.op_id in (
            "sr.math.step_v1",
            "sr.math.sign_v1",
            "sr.math.floor_v1",
            "sr.math.ceil_v1",
            "sr.math.round_v1",
            "sr.bool.not_v1",
        ):
            grad = np.zeros((n_consts, n_samples), dtype=np.float64)
        elif node.op_id == "sr.math.gauss_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = -2.0 * a * val * da
        elif node.op_id == "sr.math.tanh_v1":
            da = child_grads[0]
            grad = (1.0 - val * val) * da
        elif node.op_id == "sr.math.erf_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = (2.0 / np.sqrt(np.pi)) * np.exp(-(a * a)) * da
        elif node.op_id == "sr.math.erfc_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = -(2.0 / np.sqrt(np.pi)) * np.exp(-(a * a)) * da
        elif node.op_id == "sr.math.safe_log_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = (np.sign(a) / (np.abs(a) + EPS_DENOM)) * da
        elif node.op_id == "sr.math.asin_v1":
            a = child_vals[0]
            da = child_grads[0]
            denom = np.sqrt(np.maximum(1.0 - np.clip(a, -1.0, 1.0) ** 2, 1e-12))
            grad = da / denom
        elif node.op_id == "sr.math.acos_v1":
            a = child_vals[0]
            da = child_grads[0]
            denom = np.sqrt(np.maximum(1.0 - np.clip(a, -1.0, 1.0) ** 2, 1e-12))
            grad = -da / denom
        elif node.op_id == "sr.math.atan_v1":
            a = child_vals[0]
            da = child_grads[0]
            grad = da / (1.0 + a * a)
        elif node.op_id == "sr.math.atan2_v1":
            a, b = child_vals
            da, db = child_grads
            denom = np.maximum(a * a + b * b, 1e-12)
            grad = (b * da - a * db) / denom
        elif node.op_id == "sr.arith.neg_v1":
            grad = -child_grads[0]
        elif node.op_id in (
            "sr.arith.less_v1",
            "sr.bool.equal_v1",
            "sr.bool.less_equal_v1",
            "sr.bool.greater_v1",
            "sr.bool.greater_equal_v1",
            "sr.bool.and_v1",
            "sr.bool.or_v1",
            "sr.bool.xor_v1",
        ):
            grad = np.zeros((n_consts, n_samples), dtype=np.float64)
        elif node.op_id == "sr.math.min_v1":
            a, b = child_vals
            da, db = child_grads
            grad = np.where(a <= b, da, db)
        elif node.op_id == "sr.math.max_v1":
            a, b = child_vals
            da, db = child_grads
            grad = np.where(a >= b, da, db)
        elif node.op_id == "sr.math.mod_v1":
            grad = child_grads[0]
        elif node.op_id in (
            "sr.ts.delay_v1",
            "sr.ts.sma_v1",
            "sr.ts.wma_v1",
            "sr.ts.mma_v1",
            "sr.ts.median_v1",
        ):
            grad = np.zeros((n_consts, n_samples), dtype=np.float64)
        else:
            raise NotImplementedError(f"gradient not implemented for {node.op_id}")

        return val, grad

    raise ValueError(f"unknown node type: {type(node)}")


def loss_gradient(
    params: np.ndarray,
    expr: Node,
    X: np.ndarray,
    y: np.ndarray,
    consts: list[ConstNode],
    loss_fn=None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = X.shape[0]

    for c, p in zip(consts, params):
        c.value = float(p)

    y_pred, dy_dc = _evaluate_with_gradient(expr, X, consts)

    if not np.all(np.isfinite(y_pred)):
        return np.zeros(len(consts), dtype=np.float64)

    if loss_fn is None:
        dloss_dy = 2.0 * (y_pred - y) / n_samples
    else:
        eps = 1e-8
        loss_plus = loss_fn(y_pred + eps, y)
        loss_minus = loss_fn(y_pred - eps, y)
        dloss_dy = (loss_plus - loss_minus) / (2 * eps)

    if weights is not None:
        dloss_dy = dloss_dy * weights / np.mean(weights)

    grad = np.sum(dloss_dy * dy_dc, axis=1)
    if not np.all(np.isfinite(grad)):
        return np.zeros(len(consts), dtype=np.float64)
    return grad
