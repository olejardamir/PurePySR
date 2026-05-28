from __future__ import annotations

import copy

import numpy as np

from python_backend.eval import clear_tree_cache, evaluate, compute_loss, check_constraints, LossFn
from python_backend.expr import ConstNode, Node, OpNode
from python_backend.gradients import _collect_consts, loss_gradient


def _eval_loss(
    expr: Node, X: np.ndarray, y: np.ndarray,
    loss_fn: LossFn | None = None,
    weights: np.ndarray | None = None,
) -> float:
    clear_tree_cache(expr)
    try:
        y_pred = evaluate(expr, X)
    except Exception:
        return float("inf")
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    loss, valid, _ = compute_loss(y, y_pred, loss_fn=loss_fn, weights=weights)
    return loss if valid else float("inf")


def _hill_climb(
    expr: Node,
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int,
    loss_fn: LossFn | None = None,
    weights: np.ndarray | None = None,
    f_calls_limit: int = 0,
) -> tuple[Node, float]:
    consts = _collect_consts(expr)
    if not consts:
        return expr, _eval_loss(expr, X, y, loss_fn=loss_fn, weights=weights)

    best = expr
    best_loss = _eval_loss(best, X, y, loss_fn=loss_fn, weights=weights)
    if not np.isfinite(best_loss):
        return expr, float("inf")

    f_calls = 0
    steps = [max(abs(c.value) * 0.1, 0.01) for c in consts]

    for _ in range(n_iterations):
        if f_calls_limit and f_calls >= f_calls_limit:
            break
        improved = False
        for i, const in enumerate(consts):
            orig = const.value
            step = steps[i]

            const.value = orig + step
            loss_up = _eval_loss(best, X, y, loss_fn=loss_fn, weights=weights)
            f_calls += 1

            const.value = orig - step
            loss_down = _eval_loss(best, X, y, loss_fn=loss_fn, weights=weights)
            f_calls += 1

            if loss_up < best_loss and loss_up <= loss_down:
                best_loss = loss_up
                const.value = orig + step
                steps[i] = min(steps[i] * 1.5, 10.0)
                improved = True
            elif loss_down < best_loss:
                best_loss = loss_down
                const.value = orig - step
                steps[i] = min(steps[i] * 1.5, 10.0)
                improved = True
            else:
                const.value = orig
                steps[i] = max(steps[i] * 0.5, 1e-12)

            if f_calls_limit and f_calls >= f_calls_limit:
                break

        if not improved:
            break

    return best, best_loss


def _perturb_constants(node: Node, rng: np.random.Generator | None = None) -> Node:
    if rng is None:
        rng = np.random.default_rng()
    result = copy.deepcopy(node)
    consts = _collect_consts(result)
    for c in consts:
        c.value *= 1.0 + 0.5 * rng.normal(0.0, 1.0)
    return result


def _scipy_optimize(
    expr: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: LossFn | None = None,
    weights: np.ndarray | None = None,
    f_calls_limit: int = 0,
    method: str = "NelderMead",
    autodiff_backend: bool = False,
) -> tuple[Node, float]:
    """Optimize constants using :mod:`scipy.optimize.minimize`.

    The expression is evaluated eagerly — each call to the objective
    function sets the :class:`ConstNode` values in‑place, evaluates,
    and returns the loss.

    When *autodiff_backend* is ``True``, exact gradients are computed
    via forward‑mode automatic differentiation through the expression
    tree and passed to the solver, avoiding finite‑difference
    approximation.
    """
    import scipy.optimize as _opt

    consts = _collect_consts(expr)
    if not consts:
        return expr, _eval_loss(expr, X, y, loss_fn=loss_fn, weights=weights)

    x0 = np.array([c.value for c in consts], dtype=np.float64)

    def _objective(params: np.ndarray) -> float:
        for c, p in zip(consts, params):
            c.value = float(p)
        return _eval_loss(expr, X, y, loss_fn=loss_fn, weights=weights)

    jac: callable | None = None
    if autodiff_backend:
        def _gradient(params: np.ndarray) -> np.ndarray:
            return loss_gradient(
                params, expr, X, y, consts,
                loss_fn=loss_fn, weights=weights,
            )
        jac = _gradient

    options: dict[str, object] = {}
    if f_calls_limit:
        method_upper = method.upper().replace("-", "")
        if method_upper in ("NELDERMEAD",):
            options["maxfev"] = f_calls_limit
        else:
            options["maxiter"] = f_calls_limit
    # NelderMead → Nelder-Mead (scipy name)
    # Map short names to scipy method names
    _method_map: dict[str, str] = {
        "NelderMead": "Nelder-Mead",
        "L-BFGS-B": "L-BFGS-B",
        "LBFGSB": "L-BFGS-B",
        "BFGS": "BFGS",
        "Newton": "Newton-CG",
    }
    scipy_method = _method_map.get(method, method)

    result = _opt.minimize(
        _objective,
        x0,
        method=scipy_method,
        jac=jac,
        options=options,
    )

    # Apply final params
    for c, p in zip(consts, result.x):
        c.value = float(p)

    final_loss = _eval_loss(expr, X, y, loss_fn=loss_fn, weights=weights)
    return expr, final_loss


def optimize_constants(
    expr: Node,
    X: np.ndarray,
    y: np.ndarray,
    maxsize: int,
    maxdepth: int,
    n_iterations: int = 8,
    loss_fn: LossFn | None = None,
    constraints: dict[str, int | tuple[int, ...]] | None = None,
    nested_constraints: dict[str, dict[str, int]] | None = None,
    weights: np.ndarray | None = None,
    nrestarts: int = 2,
    f_calls_limit: int = 0,
    algorithm: str = "L-BFGS-B",
    rng: np.random.Generator | None = None,
    autodiff_backend: bool = False,
) -> Node:
    consts = _collect_consts(expr)
    if not consts:
        return expr

    best = copy.deepcopy(expr)

    if not check_constraints(
        best, maxsize, maxdepth,
        constraints=constraints, nested_constraints=nested_constraints,
    )[0]:
        return expr

    _HAS_SCIPY: bool = True
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        _HAS_SCIPY = False

    if algorithm not in ("NelderMead",) and not _HAS_SCIPY:
        algorithm = "NelderMead"

    primary: callable
    if _HAS_SCIPY:
        primary = lambda e: _scipy_optimize(
            e, X, y, loss_fn=loss_fn, weights=weights,
            f_calls_limit=f_calls_limit, method=algorithm,
            autodiff_backend=autodiff_backend,
        )
    else:
        primary = lambda e: _hill_climb(
            e, X, y, n_iterations,
            loss_fn=loss_fn, weights=weights, f_calls_limit=f_calls_limit,
        )

    best, best_loss = primary(best)
    if not np.isfinite(best_loss):
        return expr

    for _ in range(nrestarts - 1):
        candidate = _perturb_constants(expr)
        candidate, cand_loss = primary(candidate)
        if np.isfinite(cand_loss) and cand_loss < best_loss:
            best, best_loss = candidate, cand_loss

    if not check_constraints(
        best, maxsize, maxdepth,
        constraints=constraints, nested_constraints=nested_constraints,
    )[0]:
        return expr
    return best
