from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Callable

import numpy as np

from python_backend.expr import Node
from python_backend.mutation_weights import (
    MutationWeights, sample_mutation,
    condition_mutation_weights,
)
from python_backend.running_statistics import RunningSearchStatistics
from python_backend.search import (
    generate_expression,
    mutate,
    crossover_trees,
    _MUTATION_TYPES,
)


@dataclasses.dataclass(frozen=True)
class RegularizedEvolutionConfig:
    population_size: int
    tournament_size: int
    cycles: int
    maxsize: int
    maxdepth: int
    init_maxdepth: int = 3
    fresh_injection_prob: float = 0.0
    crossover_prob: float = 0.1
    mutation_weights: MutationWeights | None = None
    parsimony: float = 0.0
    fraction_replaced_guesses: float = 0.0
    use_frequency: bool = False
    use_frequency_in_tournament: bool = False
    adaptive_parsimony_scaling: float = 0.0
    baseline_loss: float = 0.01
    tournament_selection_p: float = 1.0
    running_stats: RunningSearchStatistics | None = None
    ncycles_per_iteration: int = 380
    annealing: bool = True
    alpha_annealing: float = 3.17
    skip_mutation_failures: bool = True
    should_simplify: bool = False
    bumper: bool = False
    warmup_maxsize_by: int = 0
    total_cycles: int = 0  # niterations * ncycles_per_iteration (for warmup calc)
    hof_migration: bool = False
    fraction_replaced_hof: float = 0.0
    early_stop_condition: Callable[[float, int], bool] | float | None = None
    probability_negate_constant: float = 0.01
    perturbation_factor: float = 1.0
    optimizer_probability: float = 1.0
    backsolve_context: dict | None = None


def _normalized_cost(
    loss: float, complexity: int, baseline_loss: float, parsimony: float,
) -> float:
    normalization = max(baseline_loss, 0.01)
    return loss / normalization + complexity * parsimony


def _get_cur_maxsize(config: RegularizedEvolutionConfig, cycles_elapsed: int) -> int:
    """Gradually increase maxsize during warmup (matching Julia's get_cur_maxsize)."""
    if config.warmup_maxsize_by <= 0 or config.total_cycles <= 0:
        return config.maxsize
    fraction_elapsed = cycles_elapsed / config.total_cycles
    if fraction_elapsed <= config.warmup_maxsize_by:
        return 3 + int(math.floor((config.maxsize - 3) * fraction_elapsed / config.warmup_maxsize_by))
    return config.maxsize


def _geometric_tournament_winner(
    rng: np.random.Generator,
    scores: list[tuple[float, int, str]],
    k: int,
    p: float,
) -> int:
    n = len(scores)
    k = min(k, n)
    idxs = list(rng.integers(0, n, k))
    sorted_idxs = sorted(idxs, key=lambda i: scores[i])
    if p >= 1.0:
        return sorted_idxs[0]
    weights = np.array([p * (1.0 - p) ** i for i in range(k)])
    weights /= weights.sum()
    place = rng.choice(k, p=weights)
    return sorted_idxs[place]


EvalFn = Callable[[Node], tuple[float, int, bool, str]]
TraceFn = Callable[[dict[str, object]], None]
OptimizeTreeFn = Callable[[Node], Node]

# A population entry for warm-start seeding:
# (node, loss, complexity, valid, reason, birth, frequency)
_PopEntry = tuple[Node, float, int, bool, str, int, int]


def _optimize_and_simplify_population(
    pop: list[Node],
    pop_data: list[tuple[float, int, bool, str]],
    evaluate: EvalFn,
    optimize_tree: OptimizeTreeFn | None,
    simplify_tree: OptimizeTreeFn | None,
    config: RegularizedEvolutionConfig,
    rng: np.random.Generator | None = None,
) -> None:
    for i in range(len(pop)):
        if config.should_simplify and simplify_tree is not None:
            s = simplify_tree(pop[i])
            if s is not pop[i]:
                pop[i] = s
                loss, complexity, valid, reason = evaluate(pop[i])
                pop_data[i] = (loss, complexity, valid, reason)
        if optimize_tree is not None and (rng is None or rng.random() < config.optimizer_probability):
            opt = optimize_tree(pop[i])
            if opt is not pop[i]:
                pop[i] = opt
                loss, complexity, valid, reason = evaluate(pop[i])
                pop_data[i] = (loss, complexity, valid, reason)


def _process_child(
    child: Node,
    parent_idx: int | None,
    orig_node: Node | None,
    p_loss: float | None,
    parent_cost: float | None,
    mutation_type: str,
    temperature: float,
    evaluate: EvalFn,
    optimize_tree: OptimizeTreeFn | None,
    config: RegularizedEvolutionConfig,
    rng: np.random.Generator,
    pop: list[Node],
    pop_data: list[tuple[float, int, bool, str]],
    birth: list[int],
    frequency: list[int],
    t: int,
    hall_of_fame: object,
    emit: TraceFn,
    parent_hashes: list[str],
    proposal_op: str,
    cand_before: str | None,
    simplify_tree: OptimizeTreeFn | None = None,
    target_idx: int | None = None,
) -> bool:
    """Evaluate, optionally simplify, optionally optimize, apply acceptance, and replace oldest.
    Returns ``True`` if accepted into the population."""
    child_loss, child_complexity, child_valid, child_reason = evaluate(child)

    if child_valid and config.should_simplify and simplify_tree is not None:
        child = simplify_tree(child)
        child_loss, child_complexity, child_valid, child_reason = evaluate(child)

    if child_valid and optimize_tree is not None:
        child = optimize_tree(child)
        child_loss, child_complexity, child_valid, child_reason = evaluate(child)

    child_hash = child.structural_hash()

    # ── Acceptance probability (simulated annealing + frequency) ──
    if child_valid and mutation_type == "mutation" and parent_cost is not None and parent_idx is not None:
        child_cost = _normalized_cost(child_loss, child_complexity, config.baseline_loss, config.parsimony)
        prob_change = 1.0
        if config.annealing:
            delta = child_cost - parent_cost
            prob_change *= np.exp(-delta / (temperature * config.alpha_annealing))
        if config.use_frequency and config.running_stats is not None:
            old_sz = pop_data[parent_idx][1]
            new_sz = child_complexity
            old_freq = (config.running_stats.normalized_frequencies[old_sz - 1]
                       if 0 < old_sz <= len(config.running_stats.normalized_frequencies)
                       else 1e-6)
            new_freq = (config.running_stats.normalized_frequencies[new_sz - 1]
                       if 0 < new_sz <= len(config.running_stats.normalized_frequencies)
                       else 1e-6)
            prob_change *= old_freq / new_freq
        if prob_change < rng.random():
            child = pop[parent_idx]
            child_loss, child_complexity, child_valid, child_reason = pop_data[parent_idx]

    # Replace oldest individual (aging)
    if target_idx is not None:
        oldest_idx = target_idx
    else:
        oldest_idx = min(range(len(pop)), key=lambda i: birth[i])
    replaced = False
    if child_valid or (pop_data[oldest_idx][2] is False):
        pop[oldest_idx] = child
        pop_data[oldest_idx] = (
            child_loss,
            child_complexity,
            child_valid,
            child_reason,
        )
        birth[oldest_idx] = len(pop) + t
        if config.use_frequency:
            frequency[oldest_idx] = 0
        # Track per-complexity frequency via RunningSearchStatistics
        rs = config.running_stats
        if config.use_frequency and rs is not None:
            rs.update_frequencies(child_complexity)
            if t % 100 == 0:
                rs.move_window()
                rs.normalize_frequencies()
        replaced = True

    # Update hall-of-fame if provided
    hof_status = "unchanged"
    accepted = replaced and child_valid
    if child_valid and hall_of_fame is not None:
        hof_status = hall_of_fame.consider(
            child, child_loss, child_complexity, child_hash,
        )

    emit(
        {
            "t": t,
            "child": child,
            "selected_parent_hash": cand_before,
            "candidate_hash_after": child_hash,
            "validity_status": "valid" if child_valid else "invalid",
            "loss": str(child_loss) if child_valid else None,
            "complexity": child_complexity,
            "archive_update_status": hof_status,
            "accepted_or_inserted": accepted,
            "invalid_reason_code": None if child_valid else child_reason,
            "parent_hashes": parent_hashes,
            "proposal_op": proposal_op,
            "mutation_type": mutation_type,
            "cand_before": cand_before,
        }
    )
    return accepted


def _poisson_sample(mean: float, rng: np.random.Generator) -> int:
    """Sample from Poisson distribution."""
    if mean <= 0.0:
        return 0
    L = math.exp(-mean)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(k - 1, 0)


def _migrate_into_population(
    rng: np.random.Generator,
    pop: list[Node],
    pop_data: list[tuple[float, int, bool, str]],
    birth: list[int],
    frequency: list[int],
    t: int,
    candidates: list[tuple[Node, float, int, bool, str]],
    frac: float,
    config: RegularizedEvolutionConfig,
) -> None:
    """Copy migrant candidates into the population at random locations (Julia's migrate!)."""
    if frac <= 0.0 or not candidates:
        return
    mean_replaced = len(pop) * frac
    num_replace = min(_poisson_sample(mean_replaced, rng), len(candidates), len(pop))
    if num_replace <= 0:
        return
    locations = rng.integers(0, len(pop), num_replace)
    migrant_indices = rng.integers(0, len(candidates), num_replace)
    for loc_idx, mig_idx in zip(locations, migrant_indices):
        node, loss, complexity, valid, reason = candidates[mig_idx]
        pop[loc_idx] = node
        pop_data[loc_idx] = (loss, complexity, valid, reason)
        birth[loc_idx] = len(pop) + t
        if config.use_frequency:
            frequency[loc_idx] = 0


def run_regularized_evolution(
    *,
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    config: RegularizedEvolutionConfig,
    evaluate: EvalFn,
    optimize_tree: OptimizeTreeFn | None = None,
    simplify_tree: OptimizeTreeFn | None = None,
    trace_cb: TraceFn | None = None,
    hall_of_fame: object = None,
    seed_members: list[Node] | None = None,
    seed_population: list[_PopEntry] | None = None,
) -> dict[str, object]:
    pop: list[Node] = []
    pop_data: list[tuple[float, int, bool, str]] = []
    birth: list[int] = []
    frequency: list[int] = []

    def _emit(rec: dict[str, object]) -> None:
        if trace_cb is not None:
            trace_cb(rec)

    # Initialize population
    if seed_population:
        for entry in seed_population:
            node, loss, complexity, valid, reason, b, freq = entry
            pop.append(node)
            pop_data.append((loss, complexity, valid, reason))
            birth.append(b)
            frequency.append(freq)
    else:
        for i in range(config.population_size):
            expr = generate_expression(
                rng,
                binary_ids,
                unary_ids,
                n_features,
                max_depth=min(config.init_maxdepth, config.maxdepth),
            )
            loss, complexity, valid, reason = evaluate(expr)
            pop.append(expr)
            pop_data.append((loss, complexity, valid, reason))
            birth.append(i)
            frequency.append(0)

    def _score(i: int) -> tuple[float, int, str]:
        loss, complexity, valid, _ = pop_data[i]
        if not valid:
            return (float("inf"), complexity, pop[i].structural_hash())
        cost_val = _normalized_cost(loss, complexity, config.baseline_loss, config.parsimony)
        if config.use_frequency_in_tournament and config.use_frequency and config.running_stats is not None:
            sz = complexity
            freq = (config.running_stats.normalized_frequencies[sz - 1]
                    if 0 < sz <= len(config.running_stats.normalized_frequencies)
                    else 0.0)
            cost_val *= np.exp(config.adaptive_parsimony_scaling * freq)
        return (cost_val, complexity, pop[i].structural_hash())

    _seed_idx = 0

    # ── Temperature schedule (simulated annealing) ──────────────────────
    max_temp = 1.0
    min_temp = 0.001  # never hit exactly 0 (avoid div-by-zero in annealing)
    if not config.annealing:
        min_temp = max_temp  # constant temperature=1.0
    temperatures = (
        np.linspace(max_temp, min_temp, config.ncycles_per_iteration).tolist()
        if config.ncycles_per_iteration > 1
        else [max_temp]
    )

    # ── Early stopping tracking ────────────────────────────────────────
    _best_loss = float("inf")
    _best_cycle = 0
    _cur_maxsize = config.maxsize

    for t in range(config.cycles):
        temp_idx = t % len(temperatures)
        temperature = temperatures[temp_idx]

        # Seed expression injection (guesses) — replaces worst members
        if seed_members and config.fraction_replaced_guesses > 0.0:
            seed_n = max(1, int(len(pop) * config.fraction_replaced_guesses))
            for _ in range(min(seed_n, len(seed_members))):
                loss, complexity, valid, reason = evaluate(
                    seed_members[_seed_idx % len(seed_members)],
                )
                if valid:
                    def _seed_cost(i: int) -> float:
                        l, c, _, _ = pop_data[i]
                        return _normalized_cost(l, c, config.baseline_loss, config.parsimony)
                    worst_idx = max(
                        range(len(pop)),
                        key=lambda i: (_seed_cost(i), pop_data[i][1]),
                    )
                    pop[worst_idx] = seed_members[_seed_idx % len(seed_members)]
                    pop_data[worst_idx] = (loss, complexity, valid, reason)
                    if config.use_frequency:
                        frequency[worst_idx] = 0
                _seed_idx += 1

        # ── Crossover vs Mutation decision (matching Julia) ─────────
        mw = config.mutation_weights
        do_crossover = rng.random() < config.crossover_prob and len(pop) >= 2

        # Fresh-injection (default off — Julia has none)
        if rng.random() < config.fresh_injection_prob:
            child = generate_expression(
                rng, binary_ids, unary_ids, n_features,
                max_depth=config.init_maxdepth,
            )
            parent_hashes: list[Any] = []
            proposal_op = "initial"
            mutation_type = "initial"
            cand_before: str | None = None
            children_data: list = [(child, None, None, None, None, None)]
        elif do_crossover:
            # ── Two-child crossover with mutual swap (matching Julia) ──
            k = min(config.tournament_size, len(pop))
            p_sel = config.tournament_selection_p
            p1_idx = _geometric_tournament_winner(
                rng, [_score(i) for i in range(len(pop))], k, p_sel,
            )
            p2_idx = _geometric_tournament_winner(
                rng, [_score(i) for i in range(len(pop))], k, p_sel,
            )
            if config.use_frequency:
                frequency[p1_idx] += 1
                frequency[p2_idx] += 1
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]
            parent_hashes = [parent1.structural_hash(), parent2.structural_hash()]
            proposal_op = "sr.crossover.swap_subtree_v1"
            mutation_type = "crossover"
            cand_before = None

            # Find two distinct oldest members
            oldest1 = min(range(len(pop)), key=lambda i: birth[i])
            oldest2 = min(range(len(pop)), key=lambda i: birth[i] if i != oldest1 else float('inf'))

            children_data = []
            for _ in range(10):
                result = crossover_trees(
                    parent1, parent2, rng,
                    _cur_maxsize, config.maxdepth,
                )
                if result is not None:
                    c1, c2 = result
                    children_data = [
                        (c1, p1_idx, parent1, None, None, "crossover", oldest1),
                        (c2, p2_idx, parent2, None, None, "crossover", oldest2),
                    ]
                    break
        else:
            # ── Mutation ────────────────────────────────────────
            k = min(config.tournament_size, len(pop))
            p_sel = config.tournament_selection_p
            parent_idx = _geometric_tournament_winner(
                rng, [_score(i) for i in range(len(pop))], k, p_sel,
            )
            if config.use_frequency:
                frequency[parent_idx] += 1
            parent = pop[parent_idx]
            parent_hash = parent.structural_hash()
            p_loss, p_complexity, p_valid, _ = pop_data[parent_idx]
            parent_cost = _normalized_cost(
                p_loss, p_complexity, config.baseline_loss, config.parsimony,
            )

            # Sample mutation type from weights (or uniformly)
            if mw is not None:
                mw_copy = copy.deepcopy(mw)
                condition_mutation_weights(
                    mw_copy, parent, n_features,
                    _cur_maxsize, config.should_simplify,
                )
                op = sample_mutation(rng, mw_copy)
            else:
                op = str(rng.choice(_MUTATION_TYPES))

            # Handle special types inline (simplify/optimize/do_nothing)
            if op == "simplify" and simplify_tree is not None:
                child = simplify_tree(parent)
                mutation_subtype = "simplify"
            elif op == "optimize" and optimize_tree is not None:
                child = optimize_tree(parent)
                mutation_subtype = "optimize"
            elif op == "do_nothing":
                child = parent
                mutation_subtype = "do_nothing"
            else:
                _backsolve_ctx = config.backsolve_context
                if _backsolve_ctx is not None:
                    _backsolve_ctx["population"] = [
                        (pop[i],) + pop_data[i] for i in range(len(pop))
                    ]
                child, mutation_subtype = mutate(
                    parent, rng, binary_ids, unary_ids,
                    n_features, _cur_maxsize,
                    mutation_type=op,
                    maxdepth=config.maxdepth,
                    temperature=temperature,
                    probability_negate_constant=config.probability_negate_constant,
                    perturbation_factor=config.perturbation_factor,
                    backsolve_context=_backsolve_ctx,
                )
            parent_hashes = [parent_hash]
            proposal_op = "sr.mutation.replace_subtree_v1"
            mutation_type = "mutation"
            cand_before = parent_hash

            # Skip cycle if mutation physically failed and skip_mutation_failures is set
            if (config.skip_mutation_failures
                and mutation_subtype not in ("simplify", "optimize", "do_nothing")
                and child is parent):
                if config.use_frequency:
                    frequency[parent_idx] = max(0, frequency[parent_idx] - 1)
                continue

            children_data = [(child, parent_idx, parent, p_loss, parent_cost, mutation_subtype)]

        # ── Process each child (evaluate, accept/reject, replace) ──
        for entry in children_data:
            if len(entry) == 7:
                child, _p_idx, orig_node, p_loss, parent_cost, m_subtype, tgt = entry
            else:
                child, _p_idx, orig_node, p_loss, parent_cost, m_subtype = entry
                tgt = None
            _process_child(
                child, _p_idx, orig_node, p_loss, parent_cost,
                mutation_type, temperature, evaluate, optimize_tree, config,
                rng, pop, pop_data, birth, frequency, t,
                hall_of_fame, _emit,
                parent_hashes, proposal_op, cand_before,
                simplify_tree=simplify_tree, target_idx=tgt,
            )

        # ── Early stopping ──────────────────────────────────────
        _esc = config.early_stop_condition
        if _esc is not None and (not isinstance(_esc, str) or _esc):
            if hall_of_fame is not None:
                hof_best = hall_of_fame.best()
                if hof_best is not None and hof_best["loss"] is not None:
                    cur_loss = hof_best["loss"]
                    cur_cplx = hof_best["complexity"]
                else:
                    cur_loss = None
                    cur_cplx = None
            else:
                best_idx = min(range(len(pop)), key=_score)
                cur_loss = pop_data[best_idx][0]
                cur_cplx = pop_data[best_idx][1] if len(pop_data[best_idx]) > 1 else 0

            if cur_loss is not None:
                if callable(config.early_stop_condition):
                    if config.early_stop_condition(float(cur_loss), int(cur_cplx)):
                        break
                else:
                    if float(cur_loss) <= float(config.early_stop_condition):
                        break
        else:
            # Fallback: convergence-based stop (no improvement for ncycles_per_iteration)
            if hall_of_fame is not None:
                hof_best = hall_of_fame.best()
                if hof_best is not None and hof_best["loss"] is not None:
                    if hof_best["loss"] < _best_loss:
                        _best_loss = hof_best["loss"]
                        _best_cycle = t
            else:
                best_idx = min(range(len(pop)), key=_score)
                b_loss = pop_data[best_idx][0]
                if b_loss < _best_loss:
                    _best_loss = b_loss
                    _best_cycle = t
            if t - _best_cycle > config.ncycles_per_iteration:
                break

        # ── Bumper restart mechanism ──────────────────────────────
        if config.bumper and t - _best_cycle > config.ncycles_per_iteration // 2 and t > 0:
            _stale_cycles = t - _best_cycle
            _refresh_n = max(1, len(pop) // 4)
            _worst_sorted = sorted(
                range(len(pop)),
                key=lambda i: (_score(i)[0] if pop_data[i][2] else float("inf"), -birth[i]),
                reverse=True,
            )
            for _idx in _worst_sorted[:_refresh_n]:
                _expr = generate_expression(
                    rng, binary_ids, unary_ids, n_features,
                    max_depth=min(config.init_maxdepth, config.maxdepth),
                )
                _loss, _cplx, _valid, _reason = evaluate(_expr)
                pop[_idx] = _expr
                pop_data[_idx] = (_loss, _cplx, _valid, _reason)
                birth[_idx] = len(pop) + t
                if config.use_frequency:
                    frequency[_idx] = 0

        # ── Per-iteration population optimization (Julia s_r_cycle) ──
        if (t + 1) % config.ncycles_per_iteration == 0 and (t + 1) < config.cycles:
            _optimize_and_simplify_population(
                pop, pop_data, evaluate, optimize_tree, simplify_tree, config, rng=rng,
            )
            # ── warmup_maxsize_by: increase maxsize after each iteration ──
            if config.warmup_maxsize_by > 0:
                _new_maxsize = _get_cur_maxsize(config, t + 1)
                if _new_maxsize > _cur_maxsize:
                    _cur_maxsize = _new_maxsize
            # ── Migration (HOF + seed members) ──
            if config.hof_migration and hall_of_fame is not None:
                _entries = hall_of_fame.entries() if hasattr(hall_of_fame, 'entries') else []
                if _entries:
                    _candidates = [
                        (e["expression"], e["loss"], e["complexity"], True, "")
                        for e in _entries if "expression" in e
                    ]
                    _migrate_into_population(
                        rng, pop, pop_data, birth, frequency, t,
                        _candidates, config.fraction_replaced_hof, config,
                    )
            if seed_members and config.fraction_replaced_guesses > 0.0:
                _seed_candidates = [
                    (n, 0.0, 0, True, "") for n in seed_members
                ]
                _migrate_into_population(
                    rng, pop, pop_data, birth, frequency, t,
                    _seed_candidates, config.fraction_replaced_guesses, config,
                )

    # ── Final population optimization (Julia-style) ─────────────
    _optimize_and_simplify_population(
        pop, pop_data, evaluate, optimize_tree, simplify_tree, config, rng=rng,
    )

    best_idx = min(range(len(pop)), key=_score)
    best = pop[best_idx]
    best_loss, best_complexity, best_valid, _ = pop_data[best_idx]
    return {
        "best": {
            "hash": best.structural_hash(),
            "loss": best_loss if best_valid else None,
            "complexity": best_complexity if best_valid else None,
            "canonical_expression": best.canonical(),
        },
        "population": pop,
        "population_data": pop_data,
        "birth": birth,
        "frequency": frequency,
    }
