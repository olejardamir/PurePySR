from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from python_backend.expr import Node, VarNode, ConstNode, OpNode
from python_backend.ops import OP_ID_TO_ARITY, OP_ID_TO_FN
from python_backend.eval import check_constraints, compute_complexity, evaluate


# ── Backsolve support ──────────────────────────────────────────────
# Unary operator inverse mapping: op_id -> (inverse_fn or None)
_UNARY_INVERSES: dict[str, Callable[[np.ndarray], np.ndarray] | None] = {
    "sr.math.sin_v1": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
    "sr.math.cos_v1": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    "sr.math.abs_v1": None,
    "sr.math.safe_log_v1": lambda x: np.where(np.isfinite(x), np.exp(x), np.nan),
    "sr.math.inv_v1": lambda x: np.where(np.abs(x) > 1e-16, 1.0 / x, np.nan),
}

# Binary operator inverses for left-child target (target = first child):
_BINARY_INVERSES_LEFT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "sr.arith.add_v1": lambda y, r: y - r,
    "sr.arith.sub_v1": lambda y, r: y + r,
    "sr.arith.mul_v1": lambda y, r: np.where(np.abs(r) > 1e-16, y / r, np.nan),
    "sr.math.protected_div_v1": lambda y, r: y * r,
}

# Binary operator inverses for right-child target (target = second child):
_BINARY_INVERSES_RIGHT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "sr.arith.add_v1": lambda y, l: y - l,
    "sr.arith.sub_v1": lambda y, l: l - y,
    "sr.arith.mul_v1": lambda y, l: np.where(np.abs(l) > 1e-16, y / l, np.nan),
    "sr.math.protected_div_v1": lambda y, l: np.where(np.abs(y) > 1e-16, l / y, np.nan),
}


def _node_contains(root: Node, target: Node) -> bool:
    """Check if *target* is a descendant of *root* (or root itself)."""
    if root is target:
        return True
    if isinstance(root, OpNode):
        return any(_node_contains(c, target) for c in root.children)
    return False


def _find_path_to_node(
    root: Node, target: Node,
) -> list[tuple[Node, int]] | None:
    """Return the path from root to target as a list of ``(node, child_index)`` pairs
    where ``node.children[child_index]`` is the next node on the path.
    Returns ``None`` if target is not in root."""
    if root is target:
        return []
    if not isinstance(root, OpNode):
        return None
    for i, child in enumerate(root.children):
        if _node_contains(child, target):
            rest = _find_path_to_node(child, target)
            if rest is not None:
                return [(root, i)] + rest
    return None


def _is_bad_array(x: np.ndarray) -> bool:
    return bool(np.any(np.isnan(x)) or np.any(np.isinf(x)))


def _eval_inverse_tree_array(
    tree: Node,
    X: np.ndarray,
    node_to_invert_at: Node,
    y: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Walk from root to *node_to_invert_at*, inverting each operator.
    Returns ``(target_values, success)``."""
    if tree is node_to_invert_at:
        return y.copy(), True
    if not isinstance(tree, OpNode):
        return y, False

    path = _find_path_to_node(tree, node_to_invert_at)
    if path is None:
        return y, False

    current_y = y.copy().astype(np.float64)

    for node, child_idx in path:
        if node is node_to_invert_at:
            break
        if not isinstance(node, OpNode):
            return current_y, False

        # Evaluate the sibling children and invert
        siblings_evaluated = []
        for i, child in enumerate(node.children):
            if i == child_idx:
                continue
            val = evaluate(child, X)
            if _is_bad_array(val):
                return current_y, False
            siblings_evaluated.append(val)

        if len(node.children) == 1:
            # Unary operator
            inv_fn = _UNARY_INVERSES.get(node.op_id)
            if inv_fn is None:
                return current_y, False
            current_y = inv_fn(current_y)
            if _is_bad_array(current_y):
                return current_y, False
        elif len(node.children) == 2:
            # Binary operator
            if child_idx == 0:
                inv_fn = _BINARY_INVERSES_LEFT.get(node.op_id)
                if inv_fn is None:
                    return current_y, False
                current_y = inv_fn(current_y, siblings_evaluated[0])
            else:
                inv_fn = _BINARY_INVERSES_RIGHT.get(node.op_id)
                if inv_fn is None:
                    return current_y, False
                current_y = inv_fn(current_y, siblings_evaluated[0])
            if _is_bad_array(current_y):
                return current_y, False

    return current_y, True


def _solve_library(theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear solve with pseudo-inverse fallback."""
    try:
        return np.linalg.lstsq(theta, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.linalg.pinv(theta) @ y


def stlsq(
    theta: np.ndarray,
    y: np.ndarray,
    lambda_: float = 0.01,
    max_iter: int = 10,
) -> tuple[np.ndarray, bool]:
    """Sequential Thresholded Least Squares (STLSQ).
    Returns ``(coefficients, success)``."""
    n_samples, n_features = theta.shape
    tol = np.finfo(np.float64).eps
    threshold = float(lambda_)

    if len(y) != n_samples:
        return np.zeros(n_features), False

    col_norms = np.sqrt(np.sum(theta ** 2, axis=0))
    col_norms = np.maximum(col_norms, tol)
    theta_normalised = theta / col_norms

    coefficients = _solve_library(theta_normalised, y)

    for _ in range(max_iter):
        active = np.abs(coefficients) >= threshold
        if not np.any(active):
            return np.zeros(n_features), False

        theta_active = theta_normalised[:, active]
        coeffs_active = _solve_library(theta_active, y)

        coefficients_new = np.zeros(n_features)
        coefficients_new[active] = coeffs_active

        if np.linalg.norm(coefficients_new - coefficients) < tol * 10:
            coefficients = coefficients_new
            break
        coefficients = coefficients_new

    coefficients /= col_norms
    success = bool(np.any(np.abs(coefficients) > tol * 100))
    return coefficients, success


def _string_tree(node: Node, binary_ids: list[str], unary_ids: list[str]) -> str:
    """Simplified string representation for deduplication (matching Julia's ``string_tree``)."""
    if isinstance(node, VarNode):
        return f"x{node.index}"
    if isinstance(node, ConstNode):
        return f"{node.value:.6f}"
    if isinstance(node, OpNode):
        parts = ",".join(_string_tree(c, binary_ids, unary_ids) for c in node.children)
        return f"{node.op_id}({parts})"
    return ""


def _collect_subtrees(node: Node) -> list[Node]:
    """Collect all descendant subtrees (including the root itself)."""
    result: list[Node] = [node]
    if isinstance(node, OpNode):
        for c in node.children:
            result.extend(_collect_subtrees(c))
    return result


def build_basis_library(
    prototype: Node,
    X: np.ndarray,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    population: list[tuple[Node, float, int, bool, str]] | None = None,
    max_library_size: int = 200,
    top_k: int = 10,
) -> tuple[list[Node], np.ndarray]:
    """Build a basis library: constant term, feature terms, and subtrees from top population members.
    Returns ``(valid_terms, evaluated_terms)`` where ``evaluated_terms`` is ``(n_samples, n_terms)``."""
    min_lib = 1 + n_features
    max_lib = max(max_library_size, min_lib)

    terms: list[Node] = []
    n_samples = X.shape[0]

    # Constant term
    terms.append(ConstNode(1.0))

    # Feature terms
    for i in range(n_features):
        terms.append(VarNode(i))

    # Subtrees from top population members
    if population is not None and len(population) > 0:
        sorted_pop = sorted(population, key=lambda e: e[1])  # sort by loss
        top_members = sorted_pop[:min(top_k, len(sorted_pop))]
        all_subtrees: list[Node] = []
        for member_node, _, _, _, _ in top_members:
            all_subtrees.extend(_collect_subtrees(member_node))

        # Deduplicate by string representation
        seen: set[str] = set()
        unique_subtrees: list[Node] = []
        for subtree in all_subtrees:
            s = _string_tree(subtree, binary_ids, unary_ids)
            if s not in seen:
                seen.add(s)
                unique_subtrees.append(subtree)

        n_to_add = min(len(unique_subtrees), max_lib - len(terms))
        for i in range(n_to_add):
            terms.append(copy.deepcopy(unique_subtrees[i]))

    # Evaluate all terms
    evaluated_list: list[np.ndarray] = []
    valid_terms: list[Node] = []
    for term in terms:
        try:
            val = evaluate(term, X)
            if not _is_bad_array(val):
                evaluated_list.append(val)
                valid_terms.append(term)
        except Exception:
            continue

    if not evaluated_list:
        return [], np.empty((n_samples, 0))

    evaluated_terms = np.column_stack(evaluated_list)
    return valid_terms, evaluated_terms


def _has_weighted_sum_operators(binary_ids: list[str]) -> bool:
    return "sr.arith.add_v1" in binary_ids and "sr.arith.mul_v1" in binary_ids


def combine_trees_weighted_sum(
    trees: list[Node],
    coefficients: np.ndarray,
    binary_ids: list[str],
) -> Node | None:
    """Combine expression terms into a weighted sum tree.
    Returns the combined tree or ``None`` if combination fails."""
    if not _has_weighted_sum_operators(binary_ids):
        return None
    add_id = "sr.arith.add_v1"
    mul_id = "sr.arith.mul_v1"

    tol = np.finfo(np.float64).eps * 100
    active_indices = [i for i, c in enumerate(coefficients) if abs(c) > tol]
    if not active_indices:
        return None

    active_trees = [trees[i] for i in active_indices]
    active_coeffs = coefficients[active_indices]

    if len(active_indices) == 1:
        tree = active_trees[0]
        coeff = active_coeffs[0]
        if abs(coeff - 1.0) < tol:
            return tree
        coeff_node = ConstNode(float(coeff))
        return OpNode(mul_id, [coeff_node, tree])

    weighted: list[Node] = []
    for tree, coeff in zip(active_trees, active_coeffs):
        if abs(coeff - 1.0) < tol:
            weighted.append(tree)
        else:
            coeff_node = ConstNode(float(coeff))
            weighted.append(OpNode(mul_id, [coeff_node, tree]))

    result = weighted[0]
    for i in range(1, len(weighted)):
        result = OpNode(add_id, [result, weighted[i]])
    return result


def fit_sparse_expression(
    target_values: np.ndarray,
    X: np.ndarray,
    n_features: int,
    binary_ids: list[str],
    unary_ids: list[str],
    population: list[tuple[Node, float, int, bool, str]] | None = None,
    max_library_size: int = 500,
    lambda_: float = 0.01,
    max_iter: int = 10,
) -> Node | None:
    """Fit a sparse expression to *target_values* using STLSQ.
    Returns the fitted tree or ``None`` if fitting fails."""
    if not _has_weighted_sum_operators(binary_ids):
        return None

    valid_terms, evaluated_terms = build_basis_library(
        VarNode(0), X, binary_ids, unary_ids, n_features,
        population=population, max_library_size=max_library_size,
    )
    if evaluated_terms.shape[1] < 2:
        return None

    coefficients, stlsq_success = stlsq(
        evaluated_terms, target_values,
        lambda_=lambda_, max_iter=max_iter,
    )
    if not stlsq_success:
        return None

    result_tree = combine_trees_weighted_sum(valid_terms, coefficients, binary_ids)
    if result_tree is None:
        return None

    # Verify evaluation
    try:
        predicted = evaluate(result_tree, X)
        if _is_bad_array(predicted):
            return None
    except Exception:
        return None

    return result_tree


def _find_parent(tree: Node, target: Node) -> tuple[Node, int] | None:
    """Find the parent of *target* in *tree*. Returns ``(parent, child_index)`` or ``None``."""
    if tree is target or not isinstance(tree, OpNode):
        return None
    for i, child in enumerate(tree.children):
        if child is target:
            return tree, i
        result = _find_parent(child, target)
        if result is not None:
            return result
    return None


def backsolve_rewrite_random_node(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    population: list[tuple[Node, float, int, bool, str]] | None = None,
    max_library_size: int = 500,
    lambda_: float = 0.01,
    max_iter: int = 10,
) -> Node:
    """Pick a random non-root node, backsolve for its target values,
    fit a sparse expression, and replace the node.
    Falls back to median if fitting fails."""
    if not isinstance(tree, OpNode):
        return tree

    all_nodes = _collect_nodes(tree)
    non_root_nodes = [n for n in all_nodes if n is not tree]
    if not non_root_nodes:
        return tree

    target_node = non_root_nodes[int(rng.integers(0, len(non_root_nodes)))]

    target_values, success = _eval_inverse_tree_array(tree, X, target_node, y)
    if not success or _is_bad_array(target_values):
        return tree

    new_node = fit_sparse_expression(
        target_values, X, n_features, binary_ids, unary_ids,
        population=population,
        max_library_size=max_library_size,
        lambda_=lambda_, max_iter=max_iter,
    )

    if new_node is not None:
        parent_info = _find_parent(tree, target_node)
        if parent_info is not None:
            parent, idx = parent_info
            new_children = list(parent.children)
            new_children[idx] = new_node
            new_parent = OpNode(parent.op_id, new_children)
            return _deep_replace(tree, parent, new_parent)
        # target_node is root (should not happen due to filter, but handle anyway)
        return new_node

    # Fallback: median of target_values as constant
    # (matching Julia's fallback)
    representative_val = float(np.median(target_values))
    new_const = ConstNode(representative_val)
    parent_info = _find_parent(tree, target_node)
    if parent_info is not None:
        parent, idx = parent_info
        new_children = list(parent.children)
        new_children[idx] = new_const
        new_parent = OpNode(parent.op_id, new_children)
        return _deep_replace(tree, parent, new_parent)
    return new_const


_backsolve_reserved = {"X", "y", "population", "binary_ids", "unary_ids", "n_features",
                        "max_library_size", "lambda_", "max_iter"}


def _random_tree_fixed_size(
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    size: int,
) -> Node | None:
    """Generate a tree with exactly *size* nodes (matching Julia's gen_random_tree_fixed_size)."""
    if size <= 0:
        return None
    if size == 1:
        return _random_terminal(rng, n_features)
    all_ids = binary_ids + unary_ids
    if not all_ids:
        return None
    op_id = str(rng.choice(all_ids))
    arity = OP_ID_TO_ARITY.get(op_id, 2)
    if arity == 1:
        child = _random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, size - 1)
        if child is None:
            return None
        return OpNode(op_id, [child])
    elif arity == 2:
        if size - 1 < 2:
            return None
        left_sz = int(rng.integers(1, size - 1))
        right_sz = size - 1 - left_sz
        left = _random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, left_sz)
        right = _random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, right_sz)
        if left is None or right is None:
            return None
        return OpNode(op_id, [left, right])
    else:
        parts = [1] * arity
        remaining = (size - 1) - arity
        if remaining < 0:
            return None
        for _ in range(remaining):
            parts[int(rng.integers(0, arity))] += 1
        children = [_random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, p) for p in parts]
        if any(c is None for c in children):
            return None
        return OpNode(op_id, children)


def generate_expression(
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    max_depth: int = 3,
) -> Node:
    all_ids = binary_ids + unary_ids

    if max_depth <= 0 or (not all_ids):
        return _random_terminal(rng, n_features)

    term_prob = 0.5 if max_depth >= 3 else (0.7 if max_depth >= 2 else 0.9)
    if rng.random() < term_prob:
        return _random_terminal(rng, n_features)

    use_unary = unary_ids and (not binary_ids or rng.random() < 0.35)

    if use_unary:
        op_id = str(rng.choice(unary_ids))
        child = generate_expression(
            rng, binary_ids, unary_ids, n_features, max_depth - 1,
        )
        return OpNode(op_id, [child])

    if binary_ids:
        op_id = str(rng.choice(binary_ids))
        left = generate_expression(
            rng, binary_ids, unary_ids, n_features, max_depth - 1,
        )
        right = generate_expression(
            rng, binary_ids, unary_ids, n_features, max_depth - 1,
        )
        return OpNode(op_id, [left, right])

    return _random_terminal(rng, n_features)


def _random_terminal(rng: np.random.Generator, n_features: int) -> Node:
    if rng.random() < 0.5:
        return VarNode(int(rng.integers(0, n_features)))
    s = 1.0 if rng.random() < 0.5 else -1.0
    return ConstNode(float(s * rng.uniform(0.1, 2.0)))


_MUTATION_TYPES = [
    "mutate_constant", "mutate_operator", "mutate_feature",
    "swap_operands", "rotate_tree", "add_node", "insert_node",
    "delete_node", "randomize", "backsolve",
    "form_connection", "break_connection",
]


def mutate(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    maxsize: int,
    mutation_type: str,
    maxdepth: int = 999,
    temperature: float = 1.0,
    probability_negate_constant: float = 0.01,
    perturbation_factor: float = 1.0,
    backsolve_context: dict | None = None,
) -> tuple[Node, str]:
    """Try up to 10 attempts to apply *mutation_type*.
    Returns ``(mutated_node, mutation_type)``.
    On complete failure returns the original root unchanged.
    """
    for attempt in range(10):
        result = _try_mutation(
            root, mutation_type, rng, binary_ids, unary_ids,
            n_features, maxsize, temperature,
            probability_negate_constant, perturbation_factor,
            backsolve_context=backsolve_context,
        )
        if result is not None:
            ok, _ = check_constraints(result, maxsize, maxdepth)
            if ok:
                return result, mutation_type
    return root, mutation_type


def _try_mutation(
    root: Node,
    mutation_type: str,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    maxsize: int,
    temperature: float,
    probability_negate_constant: float,
    perturbation_factor: float,
    backsolve_context: dict | None = None,
) -> Node | None:
    if mutation_type == "mutate_constant":
        return _mutate_constant(root, rng, temperature, probability_negate_constant, perturbation_factor)
    if mutation_type == "mutate_operator":
        return _mutate_operator(root, rng, binary_ids, unary_ids)
    if mutation_type == "mutate_feature":
        return _mutate_feature(root, rng, n_features)
    if mutation_type == "swap_operands":
        return _swap_operands(root, rng)
    if mutation_type == "rotate_tree":
        return _rotate_tree(root, rng)
    if mutation_type == "add_node":
        return _add_node(root, rng, binary_ids, unary_ids, n_features, maxsize)
    if mutation_type == "insert_node":
        return _insert_node(root, rng, binary_ids, unary_ids, n_features, maxsize)
    if mutation_type == "delete_node":
        return _delete_node(root, rng)
    if mutation_type == "randomize":
        target_size = int(rng.integers(1, maxsize + 1))
        return _random_tree_fixed_size(rng, binary_ids, unary_ids, n_features, target_size)
    if mutation_type == "backsolve":
        if backsolve_context is None:
            return None
        X = backsolve_context.get("X")
        y = backsolve_context.get("y")
        population = backsolve_context.get("population")
        max_library_size = backsolve_context.get("max_library_size", 500)
        lambda_ = backsolve_context.get("lambda_", 0.01)
        max_iter = backsolve_context.get("max_iter", 10)
        if X is None or y is None:
            return None
        return backsolve_rewrite_random_node(
            copy.deepcopy(root), X, y, rng, binary_ids, unary_ids,
            n_features, population=population,
            max_library_size=max_library_size,
            lambda_=lambda_, max_iter=max_iter,
        )
    if mutation_type == "form_connection":
        return _form_connection(root, rng, n_features, binary_ids, unary_ids)
    if mutation_type == "break_connection":
        return _break_connection(root, rng, n_features)
    return None


# ── mutate_constant ─────────────────────────────────────────────────


def _mutate_constant(
    root: Node,
    rng: np.random.Generator,
    temperature: float,
    probability_negate_constant: float,
    perturbation_factor: float,
) -> Node | None:
    """Pick a random constant leaf and perturb it multiplicatively."""
    consts = [n for n in _collect_nodes(root) if isinstance(n, ConstNode)]
    if not consts:
        return None
    target = consts[int(rng.integers(0, len(consts)))]
    target_val = float(target.value)
    max_change = perturbation_factor * temperature + 1.1
    factor = float(max_change ** rng.random())
    if rng.random() < 0.5:
        factor = 1.0 / factor
    if rng.random() > probability_negate_constant:
        factor *= -1.0
    return _deep_replace(root, target, ConstNode(target_val * factor))


# ── mutate_operator ──────────────────────────────────────────────────


def _mutate_operator(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
) -> Node | None:
    """Pick a random operator node and replace its op_id with another of the same arity."""
    ops = [n for n in _collect_nodes(root) if isinstance(n, OpNode)]
    if not ops:
        return None
    target = ops[int(rng.integers(0, len(ops)))]
    arity = len(target.children)
    candidates = [
        oid for oid in (binary_ids + unary_ids)
        if OP_ID_TO_ARITY.get(oid, arity) == arity
    ]
    if len(candidates) <= 1:
        return None
    new_op_id = str(rng.choice(candidates))
    return _deep_replace(root, target, OpNode(new_op_id, target.children))


# ── mutate_feature ───────────────────────────────────────────────────


def _mutate_feature(
    root: Node,
    rng: np.random.Generator,
    n_features: int,
) -> Node | None:
    """Pick a random variable leaf and change its feature index."""
    vars = [n for n in _collect_nodes(root) if isinstance(n, VarNode)]
    if not vars or n_features <= 1:
        return None
    target = vars[int(rng.integers(0, len(vars)))]
    new_idx = int(rng.integers(0, n_features))
    while new_idx == target.index and n_features > 1:
        new_idx = int(rng.integers(0, n_features))
    return _deep_replace(root, target, VarNode(new_idx))


# ── swap_operands ────────────────────────────────────────────────────


def _swap_operands(root: Node, rng: np.random.Generator) -> Node | None:
    """Pick a binary operator node and swap its children."""
    bins = [
        n for n in _collect_nodes(root)
        if isinstance(n, OpNode) and len(n.children) >= 2
    ]
    if not bins:
        return None
    target = bins[int(rng.integers(0, len(bins)))]
    new_node = OpNode(target.op_id, [target.children[1], target.children[0]])
    return _deep_replace(root, target, new_node)


# ── rotate_tree ──────────────────────────────────────────────────────


def _rotate_tree(root: Node, rng: np.random.Generator) -> Node | None:
    """Tree rotation matching Julia's ``randomly_rotate_tree!`` exactly.

    A "valid rotation root" is a binary node with a binary child.
    Probability 1/N to rotate at the actual tree root (N = number of
    valid rotation roots), otherwise pick a non-root valid rotation root.
    Swap the two nodes so the child becomes the subtree root.
    """
    all_nodes = _collect_nodes(root)

    valid_roots = [
        n for n in all_nodes
        if isinstance(n, OpNode) and len(n.children) == 2
        and any(isinstance(c, OpNode) and len(c.children) == 2 for c in n.children)
    ]
    if not valid_roots:
        return None

    num_valid = len(valid_roots)
    rotate_at_root = rng.random() < 1.0 / num_valid

    if rotate_at_root:
        if not (isinstance(root, OpNode) and len(root.children) == 2):
            return None
        rot_root = root
    else:
        non_roots = [n for n in valid_roots if n is not root]
        if not non_roots:
            return None
        rot_root = non_roots[int(rng.integers(0, len(non_roots)))]

    # Pick binary child of rot_root as pivot
    pivots = [
        (ci, c) for ci, c in enumerate(rot_root.children)
        if isinstance(c, OpNode) and len(c.children) == 2
    ]
    if not pivots:
        return None
    pivot_idx, pivot = pivots[int(rng.integers(0, len(pivots)))]

    # Pick grandchild of pivot
    gc_idx = int(rng.integers(0, len(pivot.children)))
    grandchild = pivot.children[gc_idx]

    # Build rotated subtree
    new_root_children = list(rot_root.children)
    new_root_children[pivot_idx] = grandchild
    new_rot_root = OpNode(rot_root.op_id, new_root_children)

    new_pivot_children = list(pivot.children)
    new_pivot_children[gc_idx] = new_rot_root
    new_pivot = OpNode(pivot.op_id, new_pivot_children)

    if root is rot_root:
        return new_pivot
    return _deep_replace(root, rot_root, new_pivot)


# ── add_node (leaf expansion — Julia's append_random_op) ─────────────


def _add_node(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    maxsize: int,
) -> Node | None:
    """50/50 leaf expansion or root wrapping (matching Julia's ``add_node``)."""
    if rng.random() < 0.5:
        return _expand_leaf(root, rng, binary_ids, unary_ids, n_features)
    return _wrap_root(root, rng, binary_ids, unary_ids, n_features)


def _expand_leaf(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
) -> Node | None:
    """Replace a random leaf with an operator + random terminals (matching Julia)."""
    leaves = [n for n in _collect_nodes(root) if not isinstance(n, OpNode)]
    if not leaves:
        return None
    target = leaves[int(rng.integers(0, len(leaves)))]
    all_ids = binary_ids + unary_ids
    if not all_ids:
        return None
    op_id = str(rng.choice(all_ids))
    arity = OP_ID_TO_ARITY.get(op_id, 2)
    arg_to_carry = int(rng.integers(0, arity))
    children = [_random_terminal(rng, n_features) for _ in range(arity)]
    children[arg_to_carry] = target
    return _deep_replace(root, target, OpNode(op_id, children))


def _wrap_root(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
) -> Node:
    """Wrap the whole tree as a child of a new root operator."""
    all_ids = binary_ids + unary_ids
    if not all_ids:
        return root
    op_id = str(rng.choice(all_ids))
    arity = OP_ID_TO_ARITY.get(op_id, 2)
    children = [_random_terminal(rng, n_features) for _ in range(arity)]
    children[int(rng.integers(0, arity))] = root
    return OpNode(op_id, children)


# ── insert_node ──────────────────────────────────────────────────────


def _insert_node(
    root: Node,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    maxsize: int,
) -> Node | None:
    """Pick any node and insert a new operator above it (matching Julia's ``insert_random_op``)."""
    nodes = _collect_nodes(root)
    if not nodes:
        return None
    target = nodes[int(rng.integers(0, len(nodes)))]
    all_ids = binary_ids + unary_ids
    if not all_ids:
        return None
    op_id = str(rng.choice(all_ids))
    arity = OP_ID_TO_ARITY.get(op_id, 2)
    children = [_random_terminal(rng, n_features) for _ in range(arity)]
    children[int(rng.integers(0, arity))] = target
    return _deep_replace(root, target, OpNode(op_id, children))


# ── delete_node ──────────────────────────────────────────────────────


def _delete_node(root: Node, rng: np.random.Generator) -> Node | None:
    """Pick a random operator node, splice a random child upward."""
    ops = [n for n in _collect_nodes(root) if isinstance(n, OpNode)]
    if not ops:
        return None
    target = ops[int(rng.integers(0, len(ops)))]
    carry_idx = int(rng.integers(0, len(target.children)))
    carry = target.children[carry_idx]
    return _deep_replace(root, target, carry)


# ── form_connection / break_connection ─────────────────────────────


def _form_connection(
    root: Node,
    rng: np.random.Generator,
    n_features: int,
    binary_ids: list[str],
    unary_ids: list[str],
) -> Node | None:
    """Pick a non-root OpNode and replace one child with a copy of a
    non-descendant node from elsewhere in the tree (forming a new
    connection)."""
    all_nodes = _collect_nodes(root)
    non_root_nodes = [n for n in all_nodes if n is not root]
    non_root_ops = [n for n in non_root_nodes if isinstance(n, OpNode)]
    if len(non_root_ops) < 1:
        return None
    target = non_root_ops[int(rng.integers(0, len(non_root_ops)))]

    descendants = set(_collect_nodes(target))
    candidates = [n for n in all_nodes if n is not target and n not in descendants]
    if not candidates:
        return None
    donor = copy.deepcopy(candidates[int(rng.integers(0, len(candidates)))])
    child_idx = int(rng.integers(0, len(target.children)))
    new_children = list(target.children)
    new_children[child_idx] = donor
    return _deep_replace(root, target, OpNode(target.op_id, new_children))


def _break_connection(
    root: Node,
    rng: np.random.Generator,
    n_features: int,
) -> Node | None:
    """Pick a random OpNode and replace one child with a random terminal
    (breaking the connection to that child's subtree)."""
    ops = [n for n in _collect_nodes(root) if isinstance(n, OpNode)]
    if not ops:
        return None
    target = ops[int(rng.integers(0, len(ops)))]
    child_idx = int(rng.integers(0, len(target.children)))
    new_children = list(target.children)
    new_children[child_idx] = _random_terminal(rng, n_features)
    return _deep_replace(root, target, OpNode(target.op_id, new_children))


# ── helpers ──────────────────────────────────────────────────────────


def _collect_nodes(node: Node) -> list[Node]:
    nodes: list[Node] = []

    def _walk(n: Node) -> None:
        nodes.append(n)
        if isinstance(n, OpNode):
            for c in n.children:
                _walk(c)

    _walk(node)
    return nodes


def _deep_replace(node: Node, target: Node, replacement: Node) -> Node:
    if node is target:
        return replacement
    if isinstance(node, OpNode):
        children = [_deep_replace(c, target, replacement) for c in node.children]
        return OpNode(node.op_id, children)
    return node


def generate_seeded_for_safe_log(
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    maxsize: int,
) -> list[Node]:
    if "sr.math.safe_log_v1" not in unary_ids:
        return []

    safe_log_op = "sr.math.safe_log_v1"
    abs_op = "sr.math.abs_v1" if "sr.math.abs_v1" in unary_ids else None
    add_op = "sr.arith.add_v1" if "sr.arith.add_v1" in binary_ids else None

    seeded: list[Node] = []
    if n_features < 1:
        return []

    v0: Node = VarNode(0)

    seeded.append(OpNode(safe_log_op, [v0]))

    if abs_op is not None:
        abs_expr = OpNode(abs_op, [v0])
        seeded.append(abs_expr)
        seeded.append(OpNode(safe_log_op, [abs_expr]))

    if add_op is not None:
        for const in [0.5, 1.0, 2.0]:
            add_expr = OpNode(add_op, [v0, ConstNode(const)])
            seeded.append(OpNode(safe_log_op, [add_expr]))
            if abs_op is not None:
                abs_add_expr = OpNode(add_op, [abs_expr, ConstNode(const)])
                seeded.append(OpNode(safe_log_op, [abs_add_expr]))

    return [s for s in seeded if compute_complexity(s) <= maxsize]


# ── crossover_trees ──────────────────────────────────────────────────


def crossover_trees(
    parent1: Node,
    parent2: Node,
    rng: np.random.Generator,
    maxsize: int,
    maxdepth: int,
    constraints: dict[str, int | tuple[int, ...]] | None = None,
    nested_constraints: dict[str, dict[str, int]] | None = None,
) -> tuple[Node, Node] | None:
    """Mutual subtree swap on deep copies: returns (child1, child2) or None."""
    t1 = copy.deepcopy(parent1)
    t2 = copy.deepcopy(parent2)
    nodes1 = _collect_nodes(t1)
    nodes2 = _collect_nodes(t2)
    if not nodes1 or not nodes2:
        return None

    target1 = nodes1[int(rng.integers(0, len(nodes1)))]
    target2 = nodes2[int(rng.integers(0, len(nodes2)))]

    child1 = _deep_replace(t1, target1, target2)
    child2 = _deep_replace(t2, target2, target1)
    ok1, _ = check_constraints(
        child1, maxsize, maxdepth,
        constraints=constraints, nested_constraints=nested_constraints,
    )
    ok2, _ = check_constraints(
        child2, maxsize, maxdepth,
        constraints=constraints, nested_constraints=nested_constraints,
    )
    if ok1 and ok2:
        return (child1, child2)
    return None
