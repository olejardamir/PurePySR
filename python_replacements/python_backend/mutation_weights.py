from __future__ import annotations

import dataclasses

import numpy as np

from python_backend.expr import Node, VarNode, ConstNode, OpNode
from python_backend.eval import compute_complexity


# ── 15-field taxonomy matching Julia ─────────────────────────────────


_WEIGHT_FIELDS = [
    "mutate_constant", "mutate_operator", "mutate_feature",
    "swap_operands", "rotate_tree", "add_node", "insert_node",
    "delete_node", "simplify", "randomize", "do_nothing",
    "optimize", "backsolve", "form_connection", "break_connection",
]


@dataclasses.dataclass
class MutationWeights:
    # Default values match Julia's tuned defaults exactly.
    mutate_constant: float = 0.0353
    mutate_operator: float = 3.63
    mutate_feature: float = 0.1
    swap_operands: float = 0.00608
    rotate_tree: float = 1.42
    add_node: float = 0.0771
    insert_node: float = 2.44
    delete_node: float = 0.369
    simplify: float = 0.00148
    randomize: float = 0.00695
    do_nothing: float = 0.431
    optimize: float = 0.0
    backsolve: float = 0.0
    form_connection: float = 0.5
    break_connection: float = 0.1


def weight_for(weights: MutationWeights, mutation_type: str) -> float:
    return getattr(weights, mutation_type, 0.0)


def set_weight(weights: MutationWeights, mutation_type: str, value: float) -> None:
    if hasattr(weights, mutation_type):
        setattr(weights, mutation_type, value)


# ── Applicability checks matching Julia's conditioning ───────────────


def _collect_nodes(node: Node) -> list[Node]:
    nodes: list[Node] = []

    def _walk(n: Node) -> None:
        nodes.append(n)
        if isinstance(n, OpNode):
            for c in n.children:
                _walk(c)

    _walk(node)
    return nodes


def _n_constants(member: Node) -> int:
    return sum(1 for n in _collect_nodes(member) if isinstance(n, ConstNode))


def condition_mutation_weights(
    weights: MutationWeights,
    member: Node,
    n_features: int,
    maxsize: int,
    should_simplify: bool = False,
) -> None:
    """Zero out weights for inapplicable mutations (matching Julia exactly)."""
    nodes = _collect_nodes(member)
    degree = len(nodes[0].children) if isinstance(nodes[0], OpNode) else 0

    # --- Graph ops (form_connection / break_connection) ---
    # break_connection needs at least one OpNode.
    # form_connection needs at least one non-root OpNode.
    n_ops = sum(1 for n in nodes if isinstance(n, OpNode))
    if n_ops < 1:
        weights.form_connection = 0.0
        weights.break_connection = 0.0
    else:
        n_non_root_ops = n_ops - (1 if isinstance(nodes[0], OpNode) else 0)
        if n_non_root_ops < 1:
            weights.form_connection = 0.0

    # --- Degree-0 tree (single leaf) ---
    if degree == 0:
        weights.mutate_operator = 0.0
        weights.swap_operands = 0.0
        weights.delete_node = 0.0
        weights.simplify = 0.0
        if isinstance(nodes[0], ConstNode):
            weights.optimize = 0.0
            weights.mutate_constant = 0.0
        elif isinstance(nodes[0], VarNode):
            weights.mutate_feature = 0.0
        return

    # --- No binary ops → disable swap_operands ---
    has_binary = any(
        isinstance(n, OpNode) and len(n.children) >= 2
        for n in nodes
    )
    if not has_binary:
        weights.swap_operands = 0.0

    # --- Scale mutate_constant by number of constants ---
    n_consts = _n_constants(member)
    weights.mutate_constant *= min(8, n_consts) / 8.0

    # --- Single feature → disable mutate_feature ---
    if n_features <= 1:
        weights.mutate_feature = 0.0

    # --- Complexity at maxsize → disable growth mutations ---
    complexity = compute_complexity(member)
    if complexity >= maxsize:
        weights.add_node = 0.0
        weights.insert_node = 0.0

    # --- Simplification disabled → disable simplify ---
    if not should_simplify:
        weights.simplify = 0.0


# ── Sampling (raw weights, no normalization) ─────────────────────────


def sample_mutation(
    rng: np.random.Generator,
    weights: MutationWeights,
) -> str:
    """Sample a mutation type from *weights* proportionally to weight values.
    Follows Julia's ``StatsBase.sample`` semantics — zero-weight types can
    never be picked.
    """
    candidates = [(f, getattr(weights, f)) for f in _WEIGHT_FIELDS if getattr(weights, f) > 0]
    if not candidates:
        return "do_nothing"
    total = sum(w for _, w in candidates)
    r = rng.random() * total
    cumulative = 0.0
    for name, w in candidates:
        cumulative += w
        if r <= cumulative:
            return name
    return candidates[-1][0]
