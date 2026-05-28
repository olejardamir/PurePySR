from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

import numpy as np

from python_backend.options import BackendOptions
from python_backend.ops import (
    OP_ID_TO_ARITY,
    operator_manifest_bytes,
    resolve_operator_tokens,
)
from python_backend.expr import Node, parse_expression, parse_canonical
from python_backend.eval import (
    evaluate, compute_loss, compute_complexity, check_constraints, LossFn,
)
from python_backend.search import (
    generate_expression,
    generate_seeded_for_safe_log,
    mutate,
    _MUTATION_TYPES,
)
from python_backend.regularized_evolution import (
    RegularizedEvolutionConfig,
    run_regularized_evolution,
)
from python_backend.running_statistics import RunningSearchStatistics
from python_backend.mutation_weights import MutationWeights
from python_backend.policy import EPS_DENOM
from python_backend.capabilities import CAPABILITY_LEVEL
from python_backend.hof import HallOfFame
from python_backend.constant_optimization import optimize_constants
from python_backend.trace import (
    run_start_record,
    search_step_record,
    run_end_record,
)
from python_backend.trace import canonical_json
from python_backend.digests import (
    operator_manifest_digest,
    policy_digest,
    step_trace_digest,
    _sha256,
)
from python_backend.errors import BackendOptionError, SR_ERR_OPT_001
from python_backend.capabilities import assert_operators_supported
from python_backend.loss_functions import resolve_loss


class PythonSRBackend:
    def equation_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        options: BackendOptions,
        extra_options: dict[str, Any] | None = None,
        seed: int = 0,
        weights: np.ndarray | None = None,
        saved_state: dict[str, Any] | None = None,
    ) -> dict:
        from python_backend.validator import validate_options_coverage

        # Extract data-dependent items from extra_options before validation
        _x_units: list[str] | None = None
        _y_units: str | None = None
        if extra_options is not None:
            if weights is None:
                raw_w = extra_options.pop("weights", None)
                if raw_w is not None:
                    weights = np.asarray(raw_w, dtype=np.float64)
            _x_units = extra_options.pop("x_units", None)
            _y_units = extra_options.pop("y_units", None)

        validate_options_coverage(
            dataclasses.asdict(options),
            pass_through=extra_options,
            known_as="PythonSRBackend.equation_search",
        )

        rng = np.random.default_rng(seed)
        run_id = f"py-{seed}-{int(time.time())}"

        binary_ids = resolve_operator_tokens(options.binary_operators)
        unary_ids = resolve_operator_tokens(options.unary_operators)

        n_features = X.shape[1]
        maxsize = options.maxsize
        maxdepth = options.maxdepth

        # Preprocessing: feature selection + denoising
        X, y = _preprocess_data(
            X, y,
            denoise=options.denoise,
            select_k_features=options.select_k_features,
            rng=rng,
        )
        n_features = X.shape[1]

        # Parse seed expressions (guesses) into Node trees
        seed_members: list[Node] = []
        if options.guesses:
            variable_names = [f"x{i}" for i in range(n_features)]
            for i, expr_str in enumerate(options.guesses):
                try:
                    node = parse_expression(expr_str, variable_names)
                except ValueError as e:
                    raise BackendOptionError(
                        SR_ERR_OPT_001,
                        f"guess[{i}] {expr_str!r}: {e}",
                    )
                cplx = compute_complexity(
                    node,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    mapping=options.complexity_mapping,
                )
                if cplx <= maxsize:
                    seed_members.append(node)

        loss_fn: LossFn | None = (
            resolve_loss(options.elementwise_loss) if options.elementwise_loss else None
        )
        constraints: dict[str, int | tuple[int, ...]] | None = options.constraints
        nested_constraints: dict[str, dict[str, int]] | None = options.nested_constraints

        _validate_options(options, binary_ids, unary_ids, n_features)

        # ── Multi-output dispatch ────────────────────────────────────────
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] > 1:
            return self._run_multi_output(
                X, y, options, weights, rng, run_id, seed,
                binary_ids, unary_ids, n_features,
                seed_members, loss_fn, constraints, nested_constraints,
                _x_units, _y_units, saved_state,
            )

        y = y.ravel()
        return self._run_search(
            X, y, options, weights, rng, run_id, seed,
            binary_ids, unary_ids, n_features,
            seed_members, loss_fn, constraints, nested_constraints,
            _x_units, _y_units, saved_state, output_idx=0,
        )


    def _run_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        options: BackendOptions,
        weights: np.ndarray | None,
        rng: np.random.Generator,
        run_id: str,
        seed: int,
        binary_ids: list[str],
        unary_ids: list[str],
        n_features: int,
        seed_members: list[Node],
        loss_fn: LossFn | None,
        constraints: dict | None,
        nested_constraints: dict | None,
        _x_units: list[str] | None,
        _y_units: str | None,
        saved_state: dict[str, Any] | None,
        output_idx: int = 0,
        _eval_count: list[int] | None = None,
    ) -> dict:
        pop_size = options.population_size
        maxsize = options.maxsize
        maxdepth = options.maxdepth

        hof = HallOfFame(max_size=options.topn)
        hof.set_parsimony(options.parsimony)
        step = 0
        trace_records: list[dict[str, Any]] = []

        # ── Warm-start restoration ───────────────────────────────────────
        _saved_population: list[Any] | None = None
        if saved_state is not None and options.warm_start:
            _saved_population = saved_state.get("population")
            if _saved_population is not None:
                for entry in saved_state.get("hof_entries", []):
                    node = parse_canonical(entry["canonical_expression"])
                    hof.consider(node, entry["loss"], entry["complexity"], entry["hash"])
                step = saved_state.get("step", 0)
                best_ever_loss = saved_state.get("best_ever_loss")
                best_ever_hash = saved_state.get("best_ever_hash")
                best_ever_complexity = saved_state.get("best_ever_complexity")
                if _eval_count is None:
                    _eval_count = [saved_state.get("eval_count", 0)]
                # Restore PRNG state
                prng_state = saved_state.get("prng_state")
                if prng_state is not None:
                    try:
                        bb = rng.bit_generator
                        bb.state = prng_state  # type: ignore[assignment]
                    except Exception:
                        pass

        if _eval_count is None:
            _eval_count = [0]
        eval_count = _eval_count

        policy = _make_policy(options)
        pol_digest = policy_digest(policy)
        om_digest = operator_manifest_digest(operator_manifest_bytes())
        ds_digest = _sha256(
            canonical_json(
                {
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "dtype": str(X.dtype),
                    "seed": str(seed),
                }
            ).encode("utf-8")
        )

        # ── Baseline loss (cost normalization) ──────────────────────────
        _baseline_loss: float = 0.01
        if loss_fn is not None:
            baseline_pred = np.full_like(y, np.mean(y))
            bl_loss, bl_ok, _ = compute_loss(y, baseline_pred, loss_fn=loss_fn, weights=weights)
            if bl_ok and np.isfinite(bl_loss):
                _baseline_loss = max(bl_loss, 0.01)
        else:
            pred = np.full_like(y, np.mean(y))
            bl_loss = float(np.average((pred - y) ** 2, weights=weights) if weights is not None else np.mean((pred - y) ** 2))
            if np.isfinite(bl_loss):
                _baseline_loss = max(bl_loss, 0.01)

        start_time = time.monotonic()
        termination_reason: str | None = None

        trace_records.append(
            run_start_record(
                run_id=run_id,
                seed=str(seed),
                operator_manifest_digest=om_digest,
                dataset_digest=ds_digest,
                numeric_policy_digest=pol_digest,
                evaluation_backend="numpy_vectorized",
                compatibility_level=CAPABILITY_LEVEL,
                start_time_policy="recorded",
                completion_status="in_progress",
                termination_reason=None,
            )
        )

        gen_depth = min(3, maxdepth)

        best_ever_loss: float | None = None
        best_ever_hash: str | None = None
        best_ever_complexity: int | None = None
        _early_stop = options.early_stop_condition

        if options.search_algorithm == "regularized_evolution" and options.populations > 1:
            return self._run_multi_pop_search(
                X, y, options, weights, rng, run_id, seed,
                binary_ids, unary_ids, n_features,
                seed_members, loss_fn, constraints, nested_constraints,
                _x_units, _y_units, saved_state, output_idx=output_idx,
                _eval_count=eval_count,
                hof=hof, trace_records=trace_records,
                policy=policy, pol_digest=pol_digest, om_digest=om_digest,
                ds_digest=ds_digest,
            )

        if options.search_algorithm == "regularized_evolution":
            total_cycles = options.niterations * options.ncycles_per_iteration
            _completed_cycles = saved_state.get("step", 0) if saved_state and options.warm_start else 0
            _remaining_cycles = max(total_cycles - _completed_cycles, 1)

            _running_stats = (
                RunningSearchStatistics(maxsize=maxsize)
                if options.use_frequency or options.use_frequency_in_tournament
                else None
            )

            # ── Mutation weights ────────────────────────────────────────
            _mutation_weights = MutationWeights()

            cfg = RegularizedEvolutionConfig(
                population_size=pop_size,
                tournament_size=options.tournament_selection_n,
                cycles=_remaining_cycles,
                maxsize=maxsize,
                maxdepth=maxdepth,
                init_maxdepth=gen_depth,
                parsimony=options.parsimony,
                fraction_replaced_guesses=options.fraction_replaced_guesses,
                use_frequency=options.use_frequency,
                use_frequency_in_tournament=options.use_frequency_in_tournament,
                adaptive_parsimony_scaling=options.adaptive_parsimony_scaling,
                baseline_loss=_baseline_loss,
                tournament_selection_p=options.tournament_selection_p,
                running_stats=_running_stats,
                ncycles_per_iteration=options.ncycles_per_iteration,
                annealing=True,
                alpha_annealing=options.alpha,
                skip_mutation_failures=options.skip_mutation_failures,
                mutation_weights=_mutation_weights,
                crossover_prob=options.crossover_probability,
                should_simplify=options.should_simplify,
                bumper=options.bumper,
                warmup_maxsize_by=options.warmup_maxsize_by,
                total_cycles=options.niterations * options.ncycles_per_iteration,
                hof_migration=options.hof_migration,
                fraction_replaced_hof=options.fraction_replaced_hof,
                early_stop_condition=_early_stop,
                probability_negate_constant=options.probability_negate_constant,
                perturbation_factor=options.perturbation_factor,
                optimizer_probability=options.optimize_probability,
                backsolve_context=(
                    {
                        "X": X,
                        "y": y,
                        "max_library_size": options.backsolve.max_library_size,
                        "lambda_": options.backsolve.lambda_,
                        "max_iter": options.backsolve.max_iter,
                    }
                    if options.backsolve is not None
                    else None
                ),
            )

            # ── Batching setup ──────────────────────────────────────────
            _use_batching = options.batching and options.batch_size > 0
            _batch_X = X
            _batch_y = y
            _batch_weights = weights

            if _use_batching:
                n_total = X.shape[0]
                bs = min(options.batch_size, n_total)
                # Single batch per run (matching Julia's one batch per s_r_cycle)
                idx = rng.integers(0, n_total, bs)
                _batch_X = X[idx]
                _batch_y = y[idx]
                if weights is not None:
                    _batch_weights = weights[idx]
                else:
                    _batch_weights = None
            else:
                _batch_X = X
                _batch_y = y
                _batch_weights = weights

            def _eval(expr: Node) -> tuple[float, int, bool, str]:
                eval_X, eval_y, eval_w = (
                    (_batch_X, _batch_y, _batch_weights)
                    if _use_batching
                    else (X, y, weights)
                )
                return _full_evaluate(
                    expr, eval_X, eval_y, maxsize, maxdepth,
                    loss_fn=loss_fn,
                    loss_function_expression=options.loss_function_expression,
                    loss_function=options.loss_function,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=eval_w,
                    loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=eval_count,
                    x_units=_x_units,
                    y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )

            # For HOF, always re-evaluate on full dataset
            def _full_eval(expr: Node) -> tuple[float, int, bool, str]:
                return _full_evaluate(
                    expr, X, y, maxsize, maxdepth,
                    loss_fn=loss_fn,
                    loss_function_expression=options.loss_function_expression,
                    loss_function=options.loss_function,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=weights,
                    loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=eval_count,
                    x_units=_x_units,
                    y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )

            from python_backend.expr import simplify_expression

            _simplify_fn: _OptimizeTreeFn | None = None
            if options.should_simplify:
                _simplify_fn = simplify_expression

            _optimize_fn: _OptimizeTreeFn | None = None
            _should_opt = (
                options.optimize_constants
                and (options.optimize_probability >= 1.0 or rng.random() < options.optimize_probability)
            )
            if _should_opt:
                def _opt_fn(expr: Node) -> Node:
                    return optimize_constants(
                        expr, X, y, maxsize, maxdepth,
                        n_iterations=options.optimizer_iterations,
                        loss_fn=loss_fn,
                        constraints=constraints, nested_constraints=nested_constraints,
                        weights=weights,
                        nrestarts=options.optimizer_nrestarts,
                        f_calls_limit=options.optimizer_f_calls_limit,
                        algorithm=options.optimizer_algorithm,
                        rng=rng,
                        autodiff_backend=options.autodiff_backend,
                    )
                _optimize_fn = _opt_fn

            def _trace(rec: dict[str, object]) -> None:
                nonlocal step, best_ever_loss, best_ever_hash, best_ever_complexity
                cand_valid = rec.get("validity_status") == "valid"
                cand_loss = rec.get("loss")
                cand_hash = rec.get("candidate_hash_after")

                if cand_valid and cand_loss is not None:
                    loss_val = float(cand_loss) if isinstance(cand_loss, str) else cand_loss
                    if best_ever_loss is None or loss_val < best_ever_loss:
                        best_ever_loss = loss_val
                        best_ever_complexity = rec.get("complexity")
                        best_ever_hash = cand_hash

                best = hof.best()
                trace_records.append(
                    search_step_record(
                        t=step,
                        rng_fingerprint=_rng_fingerprint(rng),
                        population_id=output_idx,
                        selected_parent_hashes=list(rec.get("parent_hashes", [])),
                        proposal_operator=rec.get("proposal_op", "sr.search.regularized_evolution_v1"),
                        mutation_or_crossover_type=rec.get("mutation_type", "mutation"),
                        candidate_hash_before=rec.get("cand_before"),
                        candidate_hash_after=cand_hash,
                        validity_status="valid" if cand_valid else "invalid",
                        loss=str(cand_loss) if cand_valid else None,
                        complexity=rec.get("complexity"),
                        archive_update_status=rec.get("archive_update_status", "unchanged"),
                        accepted_or_inserted=rec.get("accepted_or_inserted", False),
                        best_hash_after_step=(
                            best["hash"] if best else best_ever_hash
                        ),
                        best_loss_after_step=(
                            str(best["loss"])
                            if best
                            else (
                                str(best_ever_loss)
                                if best_ever_loss is not None
                                else None
                            )
                        ),
                        best_complexity_after_step=(
                            best["complexity"]
                            if best
                            else best_ever_complexity
                        ),
                        invalid_reason_code=rec.get("invalid_reason_code"),
                    )
                )
                step += 1

            # Wrap HOF to re-evaluate on full dataset when batching
            _hof = hof
            if _use_batching:
                _original_consider = _hof.consider

                def _hof_consider(
                    expr: Node, loss: float, complexity: int, h: str,
                ) -> str:
                    full_loss, _, full_valid, _ = _full_eval(expr)
                    if not full_valid:
                        return "unchanged"
                    return _original_consider(expr, full_loss, complexity, h)

                _hof.consider = _hof_consider  # type: ignore[method-assign]

            _seed_pop: list[Any] | None = None
            if _saved_population:
                _seed_pop = []
                for entry in _saved_population:
                    node = parse_canonical(entry["canonical"])
                    loss, complexity, valid, reason = entry["loss"], entry["complexity"], entry["valid"], entry.get("reason", "")
                    birth_val = entry.get("birth", 0)
                    freq_val = entry.get("frequency", 0)
                    _seed_pop.append((node, loss, complexity, valid, reason, birth_val, freq_val))

            result_re = run_regularized_evolution(
                rng=rng,
                binary_ids=binary_ids,
                unary_ids=unary_ids,
                n_features=n_features,
                config=cfg,
                evaluate=_eval,
                optimize_tree=_optimize_fn,
                simplify_tree=_simplify_fn,
                trace_cb=_trace,
                hall_of_fame=_hof,
                seed_members=seed_members,
                seed_population=_seed_pop,
            )

        else:
            population: list[Node] = [
                generate_expression(
                    rng, binary_ids, unary_ids, n_features, gen_depth,
                )
                for _ in range(pop_size)
            ]

            seeded_exprs = generate_seeded_for_safe_log(
                binary_ids, unary_ids, n_features, maxsize,
            )
            for i, expr in enumerate(seeded_exprs):
                if i < len(population):
                    population[i] = expr

            pop_data: list[tuple[float, int, bool, str]] = []
            frequency_baseline: list[int] = [0] * len(population)
            for expr in population:
                loss, complexity, valid, reason = _full_evaluate(
                    expr, X, y, maxsize, maxdepth, loss_fn=loss_fn,
                    loss_function_expression=options.loss_function_expression,
                    loss_function=options.loss_function,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=weights, loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    x_units=_x_units, y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )
                pop_data.append((loss, complexity, valid, reason))
                h = expr.structural_hash()
                if valid:
                    status = hof.consider(expr, loss, complexity, h)
                else:
                    status = "unchanged"
                best = hof.best()
                trace_records.append(
                    search_step_record(
                        t=step,
                        rng_fingerprint=_rng_fingerprint(rng),
                        population_id=output_idx,
                        selected_parent_hashes=[],
                        proposal_operator="initial",
                        mutation_or_crossover_type="initial",
                        candidate_hash_before=None,
                        candidate_hash_after=h,
                        validity_status="valid" if valid else "invalid",
                        loss=str(loss) if valid else None,
                        complexity=complexity,
                        archive_update_status=status,
                        accepted_or_inserted=valid,
                        best_hash_after_step=(
                            best["hash"] if best else None
                        ),
                        best_loss_after_step=(
                            str(best["loss"]) if best else None
                        ),
                        best_complexity_after_step=(
                            best["complexity"] if best else None
                        ),
                        invalid_reason_code=(
                            None if valid else reason
                        ),
                    )
                )
                step += 1

        if options.search_algorithm == "regularized_evolution":
            # The RE loop already executed the requested cycles; skip baseline loop.
            pass
        else:
            _should_opt_baseline = (
                options.optimize_constants
                and (options.optimize_probability >= 1.0 or rng.random() < options.optimize_probability)
            )
            _max_evals = options.max_evals
            _timeout = options.timeout_in_seconds
            early_stop_threshold: float | None = (
                float(_early_stop) if _early_stop else None
            )
            _use_fast = options.fast_cycle
            _hof_migrate = options.hof_migration
            _turbo = options.turbo
            seed_idx = 0

            for iteration in range(options.niterations):
                if _max_evals and eval_count[0] >= _max_evals:
                    termination_reason = "max_evals"
                    break
                if _timeout and time.monotonic() - start_time > _timeout:
                    termination_reason = "timeout"
                    break
                if early_stop_threshold is not None and best_ever_loss is not None and best_ever_loss <= early_stop_threshold:
                    termination_reason = "early_stop"
                    break

                _maybe_refresh_population(
                    population, pop_data, rng, binary_ids, unary_ids,
                    n_features, gen_depth, X, y, maxsize, maxdepth,
                    loss_fn=loss_fn,
                    loss_function=options.loss_function,
                    loss_function_expression=options.loss_function_expression,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=weights, loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping_val=options.complexity_mapping,
                    fraction_replaced=options.fraction_replaced,
                    parsimony=options.parsimony,
                    eval_count=eval_count,
                )

                # HOF migration: inject best HOF entries into population
                if _hof_migrate or _turbo:
                    best_hof = hof.entries()
                    if best_hof and options.fraction_replaced_hof > 0.0:
                        migrate_n = max(1, int(len(population) * options.fraction_replaced_hof))
                        if _turbo:
                            migrate_n = min(migrate_n * 3, len(population))
                        from_idx = 0
                        n_replaced = 0
                        for entry in best_hof:
                            if n_replaced >= migrate_n:
                                break
                            cand_expr = entry.get("expression")
                            if cand_expr is None:
                                continue
                            def _hof_cost(i: int) -> float:
                                return pop_data[i][0] + pop_data[i][1] * options.parsimony
                            # Find worst slot and replace
                            worst_idx = max(
                                range(len(population)),
                                key=lambda i: (_hof_cost(i), pop_data[i][1]),
                            )
                            # Don't replace if the HOF entry is worse than current
                            entry_cost = entry["loss"] + entry["complexity"] * options.parsimony
                            if _hof_cost(worst_idx) < entry_cost:
                                continue
                            population[worst_idx] = cand_expr
                            pop_data[worst_idx] = (entry["loss"], entry["complexity"], True, "")
                            if options.use_frequency:
                                frequency_baseline[worst_idx] = 0
                            n_replaced += 1

                # Seed expression injection (guesses)
                if seed_members and options.fraction_replaced_guesses > 0.0:
                    seed_n = max(1, int(len(population) * options.fraction_replaced_guesses))
                    for _ in range(min(seed_n, len(seed_members))):
                        loss, complexity, valid, reason = _full_evaluate(
                            seed_members[seed_idx % len(seed_members)], X, y,
                            maxsize, maxdepth, loss_fn=loss_fn,
                            loss_function_expression=options.loss_function_expression,
                            loss_function=options.loss_function,
                            constraints=constraints, nested_constraints=nested_constraints,
                            weights=weights, loss_scale=options.loss_scale,
                            const_weight=options.complexity_of_constants,
                            op_weight=options.complexity_of_operators,
                            var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=eval_count,
                    x_units=_x_units, y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )
                        if valid:
                            def _seed_cost(i: int) -> float:
                                return pop_data[i][0] + pop_data[i][1] * options.parsimony
                            worst_idx = max(
                                range(len(population)),
                                key=lambda i: (_seed_cost(i), pop_data[i][1]),
                            )
                            population[worst_idx] = seed_members[seed_idx % len(seed_members)]
                            pop_data[worst_idx] = (loss, complexity, valid, reason)
                            if options.use_frequency:
                                frequency_baseline[worst_idx] = 0
                        seed_idx += 1

                for _ in range(options.ncycles_per_iteration):
                    if _max_evals and eval_count[0] >= _max_evals:
                        break

                    if rng.random() < 0.3:
                        child = generate_expression(
                            rng, binary_ids, unary_ids, n_features, gen_depth,
                        )
                        if _should_opt_baseline:
                            child = optimize_constants(
                                child, X, y, maxsize, maxdepth,
                                n_iterations=options.optimizer_iterations,
                                loss_fn=loss_fn,
                                constraints=constraints, nested_constraints=nested_constraints,
                                weights=weights,
                                nrestarts=options.optimizer_nrestarts,
                                f_calls_limit=options.optimizer_f_calls_limit,
                                algorithm=options.optimizer_algorithm,
                                rng=rng,
                                autodiff_backend=options.autodiff_backend,
                            )
                        parent_hashes: list[str] = []
                        proposal_op = "initial"
                        mutation_type = "initial"
                        cand_before: str | None = None
                    else:
                        if rng.random() < 0.2:
                            parent_idx = int(rng.integers(0, len(population)))
                        else:
                            parent_idx = _tournament_select(
                                population, pop_data, rng,
                                options.tournament_selection_n,
                                parsimony=options.parsimony,
                                frequency=frequency_baseline,
                                use_frequency=options.use_frequency,
                                use_frequency_in_tournament=options.use_frequency_in_tournament,
                                adaptive_parsimony_scaling=options.adaptive_parsimony_scaling,
                            )
                        parent = population[parent_idx]
                        parent_hash = parent.structural_hash()
                        _mt = rng.choice(_MUTATION_TYPES)
                        child, _ = mutate(
                            parent, rng, binary_ids, unary_ids,
                            n_features, maxsize, mutation_type=_mt,
                        )
                        if _should_opt_baseline:
                            child = optimize_constants(
                                child, X, y, maxsize, maxdepth,
                                n_iterations=options.optimizer_iterations,
                                loss_fn=loss_fn,
                                constraints=constraints, nested_constraints=nested_constraints,
                                weights=weights,
                                nrestarts=options.optimizer_nrestarts,
                                f_calls_limit=options.optimizer_f_calls_limit,
                                algorithm=options.optimizer_algorithm,
                                rng=rng,
                                autodiff_backend=options.autodiff_backend,
                            )
                        parent_hashes = [parent_hash]
                        proposal_op = "sr.mutation.replace_subtree_v1"
                        mutation_type = "mutation"
                        cand_before = parent_hash

                    child_loss, child_complexity, child_valid, child_reason = (
                        _full_evaluate(
                            child, X, y, maxsize, maxdepth, loss_fn=loss_fn,
                            loss_function_expression=options.loss_function_expression,
                            loss_function=options.loss_function,
                            constraints=constraints, nested_constraints=nested_constraints,
                            weights=weights, loss_scale=options.loss_scale,
                            const_weight=options.complexity_of_constants,
                            op_weight=options.complexity_of_operators,
                            var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=eval_count,
                    x_units=_x_units, y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )
                    )
                    child_hash = child.structural_hash()

                    replace_idx = _find_replace_slot(
                        pop_data, child_loss, child_complexity, child_valid,
                        parsimony=options.parsimony,
                    )
                    if replace_idx is not None:
                        population[replace_idx] = child
                        pop_data[replace_idx] = (
                            child_loss, child_complexity, child_valid, child_reason,
                        )
                        if options.use_frequency:
                            frequency_baseline[replace_idx] = 0

                    if child_valid:
                        status = hof.consider(
                            child, child_loss, child_complexity, child_hash,
                        )
                        if best_ever_loss is None or child_loss < best_ever_loss:
                            best_ever_loss = child_loss
                            best_ever_complexity = child_complexity
                            best_ever_hash = child_hash
                    else:
                        status = "unchanged"

                    best = hof.best()
                    trace_records.append(
                        search_step_record(
                            t=step,
                            rng_fingerprint=_rng_fingerprint(rng),
                            population_id=output_idx,
                            selected_parent_hashes=parent_hashes,
                            proposal_operator=proposal_op,
                            mutation_or_crossover_type=mutation_type,
                            candidate_hash_before=cand_before,
                            candidate_hash_after=child_hash,
                            validity_status="valid" if child_valid else "invalid",
                            loss=str(child_loss) if child_valid else None,
                            complexity=child_complexity,
                            archive_update_status=status,
                            accepted_or_inserted=(
                                replace_idx is not None and child_valid
                            ),
                            best_hash_after_step=(
                                best["hash"] if best else best_ever_hash
                            ),
                            best_loss_after_step=(
                                str(best["loss"])
                                if best
                                else (
                                    str(best_ever_loss)
                                    if best_ever_loss is not None
                                    else None
                                )
                            ),
                            best_complexity_after_step=(
                                best["complexity"]
                                if best
                                else best_ever_complexity
                            ),
                            invalid_reason_code=(
                                None if child_valid else child_reason
                            ),
                        )
                    )
                    step += 1

        best_entry = hof.best()
        st_digest = step_trace_digest(
            [r for r in trace_records if r.get("record_type") == "search_step"]
        )
        archive_digest_str = _sha256(
            canonical_json(
                [{"hash": e["hash"], "loss": str(e["loss"]), "complexity": e["complexity"]}
                 for e in hof.entries()]
            ).encode("utf-8")
        )
        final_result_digest_str = _sha256(
            canonical_json(best_entry).encode("utf-8")
            if best_entry
            else b"null"
        )

        trace_records.append(
            run_end_record(
                run_id=run_id,
                termination_reason=termination_reason or "max_iterations",
                final_result_digest=final_result_digest_str,
                archive_digest=archive_digest_str,
                step_trace_digest=st_digest,
            )
        )

        dataset_manifest = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "dtype": str(X.dtype),
            "seed": str(seed),
        }

        # ── Build saved_state for warm start ────────────────────────────
        _saved_state: dict[str, Any] | None = None
        if options.warm_start:
            try:
                bb = rng.bit_generator
                prng_state = bb.state
            except Exception:
                prng_state = {}
            _pop_state: list[dict[str, Any]] = []
            if result_re is not None:
                _pop_raw = result_re.get("population", [])
                _pop_raw_data = result_re.get("population_data", [])
                _birth = result_re.get("birth", [])
                _freq = result_re.get("frequency", [])
                for i, node in enumerate(_pop_raw):
                    pd = _pop_raw_data[i] if i < len(_pop_raw_data) else (float("inf"), 0, False, "")
                    _pop_state.append({
                        "canonical": node.canonical(),
                        "loss": pd[0],
                        "complexity": pd[1],
                        "valid": pd[2],
                        "reason": pd[3],
                        "birth": _birth[i] if i < len(_birth) else 0,
                        "frequency": _freq[i] if i < len(_freq) else 0,
                    })
            _saved_state = {
                "population": _pop_state,
                "hof_entries": [
                    {"canonical_expression": e["canonical_expression"], "loss": e["loss"], "complexity": e["complexity"], "hash": e["hash"]}
                    for e in hof.entries()
                ],
                "step": step,
                "best_ever_loss": best_ever_loss,
                "best_ever_hash": best_ever_hash,
                "best_ever_complexity": best_ever_complexity,
                "eval_count": eval_count[0],
                "seed": seed,
                "prng_state": prng_state,
            }

        return {
            "run_id": run_id,
            "best": best_entry,
            "hall_of_fame": hof.entries(),
            "trace_records": trace_records,
            "policy_dict": policy,
            "dataset_manifest": dataset_manifest,
            "termination_reason": termination_reason or "max_iterations",
            "digests": {
                "operator_manifest_digest": om_digest,
                "policy_digest": pol_digest,
                "step_trace_digest": st_digest,
                "dataset_digest": ds_digest,
                "archive_digest": archive_digest_str,
                "final_result_digest": final_result_digest_str,
            },
            "saved_state": _saved_state,
        }

    def _run_multi_output(
        self,
        X: np.ndarray,
        y: np.ndarray,
        options: BackendOptions,
        weights: np.ndarray | None,
        rng: np.random.Generator,
        run_id: str,
        seed: int,
        binary_ids: list[str],
        unary_ids: list[str],
        n_features: int,
        seed_members: list[Node],
        loss_fn: LossFn | None,
        constraints: dict | None,
        nested_constraints: dict | None,
        _x_units: list[str] | None,
        _y_units: str | None,
        saved_state: dict[str, Any] | None,
    ) -> dict:
        nout = y.shape[1]
        per_output: list[dict] = []

        shared_eval_count: list[int] = [0]

        for j in range(nout):
            y_j = y[:, j]
            w_j = weights[:, j] if weights is not None and weights.ndim > 1 else weights
            out_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            out_id = f"{run_id}_out{j}"
            out_seed = int(rng.integers(0, 2**31))

            out_saved = None
            if saved_state is not None and options.warm_start:
                out_saved = saved_state.get(f"output_{j}", saved_state)

            per_output.append(
                self._run_search(
                    X, y_j, options, w_j, out_rng, out_id, out_seed,
                    binary_ids, unary_ids, n_features,
                    seed_members, loss_fn, constraints, nested_constraints,
                    _x_units, _y_units, out_saved, output_idx=j,
                    _eval_count=shared_eval_count,
                )
            )

        # ── Combine per-output results ──────────────────────────────────
        all_hofs: list[list[dict]] = [o["hall_of_fame"] for o in per_output]
        all_traces: list[dict] = []
        all_bests: list = []
        combined_hof_entries: list[dict] = []
        combined_digests: dict[str, str] = {}
        combined_saved_state: dict[str, Any] = {}
        for j, out in enumerate(per_output):
            all_bests.append(out["best"])
            if out["trace_records"]:
                all_traces.extend(out["trace_records"])
            for he in out.get("hall_of_fame", []):
                he_with_output = dict(he)
                he_with_output["output_index"] = j
                combined_hof_entries.append(he_with_output)
            combined_saved_state[f"output_{j}"] = out.get("saved_state")
            for dk, dv in out.get("digests", {}).items():
                combined_digests[f"{dk}_out{j}"] = dv

        st_digest = step_trace_digest(
            [r for r in all_traces if r.get("record_type") == "search_step"]
        )

        return {
            "run_id": run_id,
            "best": all_bests,
            "hall_of_fame": all_hofs,
            "n_outputs": nout,
            "trace_records": all_traces,
            "policy_dict": per_output[0]["policy_dict"] if per_output else {},
            "dataset_manifest": per_output[0]["dataset_manifest"] if per_output else {},
            "digests": {
                "operator_manifest_digest": per_output[0]["digests"]["operator_manifest_digest"] if per_output else "",
                "policy_digest": per_output[0]["digests"]["policy_digest"] if per_output else "",
                **combined_digests,
                "step_trace_digest": st_digest,
            },
            "saved_state": combined_saved_state,
        }

    def _run_multi_pop_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        options: BackendOptions,
        weights: np.ndarray | None,
        rng: np.random.Generator,
        run_id: str,
        seed: int,
        binary_ids: list[str],
        unary_ids: list[str],
        n_features: int,
        seed_members: list[Node],
        loss_fn: LossFn | None,
        constraints: dict | None,
        nested_constraints: dict | None,
        _x_units: list[str] | None,
        _y_units: str | None,
        saved_state: dict[str, Any] | None,
        output_idx: int = 0,
        _eval_count: list[int] | None = None,
        hof: HallOfFame | None = None,
        trace_records: list[dict[str, Any]] | None = None,
        policy: dict[str, object] | None = None,
        pol_digest: str = "",
        om_digest: str = "",
        ds_digest: str = "",
    ) -> dict:
        n_pop = options.populations
        if n_pop < 1:
            n_pop = 1

        maxsize = options.maxsize
        maxdepth = options.maxdepth
        pop_size = options.population_size
        gen_depth = min(3, maxdepth)
        _early_stop = options.early_stop_condition
        start_time = time.monotonic()
        termination_reason: str | None = None

        if _eval_count is None:
            _eval_count = [0]

        # ── Baseline loss ────────────────────────────────────────────────
        _baseline_loss: float = 0.01
        if loss_fn is not None:
            bl_pred = np.full_like(y, np.mean(y))
            bl_loss, bl_ok, _ = compute_loss(y, bl_pred, loss_fn=loss_fn, weights=weights)
            if bl_ok and np.isfinite(bl_loss):
                _baseline_loss = max(bl_loss, 0.01)
        else:
            bl_pred = np.full_like(y, np.mean(y))
            bl_loss = float(np.average((bl_pred - y) ** 2, weights=weights) if weights is not None else np.mean((bl_pred - y) ** 2))
            if np.isfinite(bl_loss):
                _baseline_loss = max(bl_loss, 0.01)

        if trace_records is None:
            trace_records = []
            trace_records.append(
                run_start_record(
                    run_id=run_id,
                    seed=str(seed),
                    operator_manifest_digest=om_digest,
                    dataset_digest=ds_digest,
                    numeric_policy_digest=pol_digest,
                    evaluation_backend="numpy_vectorized",
                    compatibility_level=CAPABILITY_LEVEL,
                    start_time_policy="recorded",
                    completion_status="in_progress",
                    termination_reason=None,
                )
            )

        if hof is None:
            hof = HallOfFame(max_size=options.topn)
            hof.set_parsimony(options.parsimony)

        # ── For each population, run the RE loop ─────────────────────────
        pop_hofs: list[HallOfFame] = []
        pop_trace_records: list[list[dict[str, Any]]] = []
        pop_result_dicts: list[dict] = []
        pop_rngs: list[np.random.Generator] = []

        total_cycles = options.niterations * options.ncycles_per_iteration
        ds_digest_str = ds_digest

        for pop_idx in range(n_pop):
            sub_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            pop_rngs.append(sub_rng)
            sub_hof = HallOfFame(max_size=options.topn)
            sub_hof.set_parsimony(options.parsimony)
            pop_hofs.append(sub_hof)
            pop_trace_records.append([])

            _running_stats = (
                RunningSearchStatistics(maxsize=maxsize)
                if options.use_frequency or options.use_frequency_in_tournament
                else None
            )

            _mutation_weights = MutationWeights()

            cfg = RegularizedEvolutionConfig(
                population_size=pop_size,
                tournament_size=options.tournament_selection_n,
                cycles=total_cycles,
                maxsize=maxsize,
                maxdepth=maxdepth,
                init_maxdepth=gen_depth,
                parsimony=options.parsimony,
                fraction_replaced_guesses=options.fraction_replaced_guesses,
                use_frequency=options.use_frequency,
                use_frequency_in_tournament=options.use_frequency_in_tournament,
                adaptive_parsimony_scaling=options.adaptive_parsimony_scaling,
                baseline_loss=_baseline_loss,
                tournament_selection_p=options.tournament_selection_p,
                running_stats=_running_stats,
                ncycles_per_iteration=options.ncycles_per_iteration,
                annealing=True,
                alpha_annealing=options.alpha,
                skip_mutation_failures=options.skip_mutation_failures,
                mutation_weights=_mutation_weights,
                crossover_prob=options.crossover_probability,
                should_simplify=options.should_simplify,
                bumper=options.bumper,
                warmup_maxsize_by=options.warmup_maxsize_by,
                total_cycles=total_cycles,
                hof_migration=options.hof_migration,
                fraction_replaced_hof=options.fraction_replaced_hof,
                early_stop_condition=_early_stop,
                probability_negate_constant=options.probability_negate_constant,
                perturbation_factor=options.perturbation_factor,
                optimizer_probability=options.optimize_probability,
                backsolve_context=(
                    {
                        "X": X,
                        "y": y,
                        "max_library_size": options.backsolve.max_library_size,
                        "lambda_": options.backsolve.lambda_,
                        "max_iter": options.backsolve.max_iter,
                    }
                    if options.backsolve is not None
                    else None
                ),
            )

            _use_batching = options.batching and options.batch_size > 0
            if _use_batching:
                n_total = X.shape[0]
                bs = min(options.batch_size, n_total)
                idx = sub_rng.integers(0, n_total, bs)
                _batch_X = X[idx]
                _batch_y = y[idx]
                _batch_w = weights[idx] if weights is not None else None
            else:
                _batch_X = X
                _batch_y = y
                _batch_w = weights

            def _eval(expr: Node, _bx=_batch_X, _by=_batch_y, _bw=_batch_w) -> tuple[float, int, bool, str]:
                return _full_evaluate(
                    expr, _bx, _by, maxsize, maxdepth,
                    loss_fn=loss_fn,
                    loss_function_expression=options.loss_function_expression,
                    loss_function=options.loss_function,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=_bw,
                    loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=_eval_count,
                    x_units=_x_units, y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )

            def _full_eval(expr: Node) -> tuple[float, int, bool, str]:
                return _full_evaluate(
                    expr, X, y, maxsize, maxdepth,
                    loss_fn=loss_fn,
                    loss_function_expression=options.loss_function_expression,
                    loss_function=options.loss_function,
                    constraints=constraints, nested_constraints=nested_constraints,
                    weights=weights,
                    loss_scale=options.loss_scale,
                    const_weight=options.complexity_of_constants,
                    op_weight=options.complexity_of_operators,
                    var_weight=options.complexity_of_variables,
                    complexity_mapping=options.complexity_mapping,
                    eval_count=_eval_count,
                    x_units=_x_units, y_units=_y_units,
                    dimensional_constraint_penalty=options.dimensional_constraint_penalty,
                    dimensionless_constants_only=options.dimensionless_constants_only,
                )

            from python_backend.expr import simplify_expression

            _simplify_fn: _OptimizeTreeFn | None = None
            if options.should_simplify:
                _simplify_fn = simplify_expression

            _optimize_fn: _OptimizeTreeFn | None = None
            _should_opt = (
                options.optimize_constants
                and (options.optimize_probability >= 1.0 or sub_rng.random() < options.optimize_probability)
            )
            if _should_opt:
                def _opt_fn(expr: Node) -> Node:
                    return optimize_constants(
                        expr, X, y, maxsize, maxdepth,
                        n_iterations=options.optimizer_iterations,
                        loss_fn=loss_fn,
                        constraints=constraints, nested_constraints=nested_constraints,
                        weights=weights,
                        nrestarts=options.optimizer_nrestarts,
                        f_calls_limit=options.optimizer_f_calls_limit,
                        algorithm=options.optimizer_algorithm,
                        rng=sub_rng,
                        autodiff_backend=options.autodiff_backend,
                    )
                _optimize_fn = _opt_fn

            step_counter: list[int] = [0]
            pop_step_offset = pop_idx * total_cycles

            def _trace(rec: dict[str, object], _idx=pop_idx, _offset=pop_step_offset) -> None:
                record = search_step_record(
                    t=_offset + step_counter[0],
                    rng_fingerprint=_rng_fingerprint(sub_rng),
                    population_id=_idx,
                    selected_parent_hashes=list(rec.get("parent_hashes", [])),
                    proposal_operator=rec.get("proposal_op", "sr.search.regularized_evolution_v1"),
                    mutation_or_crossover_type=rec.get("mutation_type", "mutation"),
                    candidate_hash_before=rec.get("cand_before"),
                    candidate_hash_after=rec.get("candidate_hash_after"),
                    validity_status=rec.get("validity_status", "invalid"),
                    loss=str(rec.get("loss")) if rec.get("validity_status") == "valid" else None,
                    complexity=rec.get("complexity"),
                    archive_update_status=rec.get("archive_update_status", "unchanged"),
                    accepted_or_inserted=rec.get("accepted_or_inserted", False),
                    best_hash_after_step=rec.get("best_hash_after_step", ""),
                    best_loss_after_step=rec.get("best_loss_after_step"),
                    best_complexity_after_step=rec.get("best_complexity_after_step"),
                    invalid_reason_code=rec.get("invalid_reason_code"),
                )
                pop_trace_records[_idx].append(record)
                step_counter[0] += 1

            _seed_pop: list[Any] | None = None
            if saved_state is not None and options.warm_start:
                _sp = saved_state.get("population")
                if _sp:
                    _seed_pop = []
                    for entry in _sp:
                        node = parse_canonical(entry["canonical"])
                        loss = entry["loss"]
                        complexity = entry["complexity"]
                        valid = entry["valid"]
                        reason = entry.get("reason", "")
                        birth_val = entry.get("birth", 0)
                        freq_val = entry.get("frequency", 0)
                        _seed_pop.append((node, loss, complexity, valid, reason, birth_val, freq_val))

            result_re = run_regularized_evolution(
                rng=sub_rng,
                binary_ids=binary_ids,
                unary_ids=unary_ids,
                n_features=n_features,
                config=cfg,
                evaluate=_eval,
                optimize_tree=_optimize_fn,
                simplify_tree=_simplify_fn,
                trace_cb=_trace,
                hall_of_fame=sub_hof,
                seed_members=seed_members,
                seed_population=_seed_pop,
            )
            pop_result_dicts.append(result_re)

        # ── Migration loop: cross-pollinate between populations ──────────
        if options.migration and n_pop > 1:
            for iteration in range(options.niterations):
                # Gather best from all populations
                all_best_candidates: list[tuple[Node, float, int, bool, str]] = []
                for pi in range(n_pop):
                    pop_hof_entries = pop_hofs[pi].entries()
                    for entry in pop_hof_entries:
                        cand_expr = entry.get("expression")
                        if cand_expr is not None:
                            all_best_candidates.append(
                                (cand_expr, entry["loss"], entry["complexity"], True, "")
                            )

                if not all_best_candidates:
                    continue

                # Migrate into each population
                for pi in range(n_pop):
                    pd = pop_result_dicts[pi]
                    raw_pop = pd.get("population", [])
                    raw_data = pd.get("population_data", [])
                    raw_birth = pd.get("birth", [])
                    raw_freq = pd.get("frequency", [])

                    # Reconstruct pop state for migration
                    from python_backend.regularized_evolution import _migrate_into_population

                    _migrate_into_population(
                        pop_rngs[pi],
                        raw_pop, raw_data, raw_birth, raw_freq,
                        0, all_best_candidates,
                        options.fraction_replaced,
                        RegularizedEvolutionConfig(
                            population_size=pop_size,
                            tournament_size=options.tournament_selection_n,
                            cycles=1,
                            maxsize=maxsize,
                            maxdepth=maxdepth,
                            use_frequency=options.use_frequency,
                        ),
                    )

        # ── Combine per-population results ──────────────────────────────
        combined_hof_entry_map: dict[str, dict] = {}
        for pi in range(n_pop):
            for entry in pop_hofs[pi].entries():
                h = entry.get("hash", "")
                if h and (h not in combined_hof_entry_map or entry["loss"] < combined_hof_entry_map[h]["loss"]):
                    combined_hof_entry_map[h] = dict(entry)

        all_trace_records = list(trace_records) if trace_records else []
        for pr in pop_trace_records:
            all_trace_records.extend(pr)

        st_digest = step_trace_digest(
            [r for r in all_trace_records if r.get("record_type") == "search_step"]
        )

        all_best_entries = [h.best() for h in pop_hofs]
        best_entry = None
        for be in all_best_entries:
            if be is not None and (best_entry is None or be["loss"] < best_entry["loss"]):
                best_entry = be

        combined_hof_entries = sorted(
            list(combined_hof_entry_map.values()),
            key=lambda e: (e.get("loss", float("inf")), e.get("complexity", 0)),
        )[:options.topn]

        archive_digest_str = _sha256(
            canonical_json(
                [{"hash": e["hash"], "loss": str(e["loss"]), "complexity": e["complexity"]}
                 for e in combined_hof_entries]
            ).encode("utf-8")
        )
        final_result_digest_str = _sha256(
            canonical_json(best_entry).encode("utf-8")
            if best_entry
            else b"null"
        )

        all_trace_records.append(
            run_end_record(
                run_id=run_id,
                final_result_digest=final_result_digest_str,
                archive_digest=archive_digest_str,
                step_trace_digest=st_digest,
            )
        )

        dataset_manifest = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "dtype": str(X.dtype),
            "seed": str(seed),
        }

        _saved_state: dict[str, Any] | None = None
        if options.warm_start:
            try:
                bb = rng.bit_generator
                prng_state = bb.state
            except Exception:
                prng_state = {}
            _pop_state: list[dict[str, Any]] = []
            for pi in range(n_pop):
                pd = pop_result_dicts[pi]
                _pop_raw = pd.get("population", [])
                _pop_raw_data = pd.get("population_data", [])
                _birth = pd.get("birth", [])
                _freq = pd.get("frequency", [])
                for i, node in enumerate(_pop_raw):
                    pdi = _pop_raw_data[i] if i < len(_pop_raw_data) else (float("inf"), 0, False, "")
                    _pop_state.append({
                        "canonical": node.canonical() if hasattr(node, "canonical") else str(node),
                        "loss": pdi[0],
                        "complexity": pdi[1],
                        "valid": pdi[2],
                        "reason": pdi[3],
                        "birth": _birth[i] if i < len(_birth) else 0,
                        "frequency": _freq[i] if i < len(_freq) else 0,
                    })
            _saved_state = {
                "population": _pop_state,
                "hof_entries": [
                    {"canonical_expression": e["canonical_expression"], "loss": e["loss"],
                     "complexity": e["complexity"], "hash": e["hash"]}
                    for e in combined_hof_entries
                ],
                "step": options.niterations + (saved_state.get("step", 0) if saved_state and options.warm_start else 0),
                "best_ever_loss": best_entry["loss"] if best_entry else None,
                "best_ever_hash": best_entry["hash"] if best_entry else None,
                "best_ever_complexity": best_entry["complexity"] if best_entry else None,
                "eval_count": sum(pd.get("eval_count", [0])[0] for pd in pop_result_dicts if pd.get("eval_count")),
                "seed": seed,
                "prng_state": prng_state,
            }

        return {
            "run_id": run_id,
            "best": best_entry,
            "hall_of_fame": combined_hof_entries,
            "trace_records": all_trace_records,
            "policy_dict": policy if policy else {},
            "dataset_manifest": dataset_manifest,
            "digests": {
                "operator_manifest_digest": om_digest if om_digest else "",
                "policy_digest": pol_digest if pol_digest else "",
                "step_trace_digest": st_digest,
                "dataset_digest": ds_digest if ds_digest else "",
                "archive_digest": archive_digest_str,
                "final_result_digest": final_result_digest_str,
            },
            "saved_state": _saved_state,
        }


def _rng_fingerprint(rng: np.random.Generator) -> str:
    state = json.dumps(rng.bit_generator.state, sort_keys=True)
    h = hashlib.sha256(state.encode("utf-8")).hexdigest()
    return f"u64:{h[:16]}"


def _validate_options(
    options: BackendOptions,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
) -> None:
    if not binary_ids and not unary_ids:
        raise BackendOptionError(
            SR_ERR_OPT_001,
            "at least one operator required",
        )
    for oid in binary_ids + unary_ids:
        if oid not in OP_ID_TO_ARITY:
            raise BackendOptionError(
                SR_ERR_OPT_001,
                f"unknown operator id {oid!r}",
            )
    assert_operators_supported(binary_ids, unary_ids)


def _make_policy(options: BackendOptions) -> dict[str, object]:
    return {
        "seed_space": "uint64",
        "prng": "PCG64",
        "dtype": "float64",
        "fast_math": "forbidden",
        "stable_sort": True,
        "deterministic_ties": True,
        "invalid_policy": "SR-INV-NONFINITE-001",
        "objective": "(loss, complexity, structural_hash)",
        "eps_denom": str(EPS_DENOM),
        "fraction_replaced": str(options.fraction_replaced),
        "fraction_replaced_hof": str(options.fraction_replaced_hof),
        "optimize_probability": str(options.optimize_probability),
        "optimizer_iterations": str(options.optimizer_iterations),
        "optimizer_nrestarts": str(options.optimizer_nrestarts),
        "max_evals": str(options.max_evals),
        "timeout_in_seconds": str(options.timeout_in_seconds),
        "fast_cycle": str(options.fast_cycle),
        "turbo": str(options.turbo),
        "early_stop_condition": str(options.early_stop_condition),
        "precision": str(options.precision),
        "hof_migration": str(options.hof_migration),
        "parsimony": str(options.parsimony),
        "complexity_of_constants": str(options.complexity_of_constants),
        "complexity_of_operators": str(options.complexity_of_operators),
        "complexity_of_variables": str(options.complexity_of_variables),
        "adaptive_parsimony_scaling": str(options.adaptive_parsimony_scaling),
        "guesses": str(len(options.guesses) if options.guesses else 0),
        "fraction_replaced_guesses": str(options.fraction_replaced_guesses),
    }


def _full_evaluate(
    expr: Node,
    X: np.ndarray,
    y: np.ndarray,
    maxsize: int,
    maxdepth: int,
    loss_fn: LossFn | None = None,
    loss_function_expression: Callable | None = None,
    loss_function: Callable | None = None,
    constraints: dict[str, int | tuple[int, ...]] | None = None,
    nested_constraints: dict[str, dict[str, int]] | None = None,
    weights: np.ndarray | None = None,
    loss_scale: float = 1.0,
    const_weight: int = 1,
    op_weight: int = 1,
    var_weight: int = 1,
    complexity_mapping: Callable | None = None,
    eval_count: list[int] | None = None,
    # Dimensional constraints
    x_units: list[str] | None = None,
    y_units: str | None = None,
    dimensional_constraint_penalty: float | None = None,
    dimensionless_constants_only: bool = False,
) -> tuple[float, int, bool, str]:
    if eval_count is not None:
        eval_count[0] += 1
    complexity = compute_complexity(
        expr,
        const_weight=const_weight,
        op_weight=op_weight,
        var_weight=var_weight,
        mapping=complexity_mapping,
    )
    if complexity > maxsize:
        return (float("inf"), complexity, False, "SR-INV-CONSTR-001")

    ok, reason = check_constraints(
        expr, maxsize, maxdepth,
        constraints=constraints, nested_constraints=nested_constraints,
    )
    if not ok:
        return (float("inf"), complexity, False, reason)

    if loss_function_expression is not None:
        loss = loss_function_expression(expr, y, X)
        if not np.isfinite(loss):
            return (float("inf"), complexity, False, "SR-INV-OBJ-001")
        return (loss, complexity, True, "")

    try:
        y_pred = evaluate(expr, X)
    except Exception:
        return (float("inf"), complexity, False, "SR-INV-EVAL-001")

    if loss_function is not None:
        loss = loss_function(y, y_pred)
        if not np.isfinite(loss):
            return (float("inf"), complexity, False, "SR-INV-OBJ-001")
    else:
        loss, valid, reason = compute_loss(y, y_pred, loss_fn=loss_fn, weights=weights)
        if not valid:
            return (float("inf"), complexity, False, reason)

    loss = loss * loss_scale if loss_scale != 1.0 else loss

    # ── Dimensional constraints penalty ──────────────────────────────────
    if dimensional_constraint_penalty is not None and x_units is not None and y_units is not None:
        from python_backend.dimensional import check_dimensions
        if check_dimensions(
            expr, x_units, y_units,
            allow_wildcards=not dimensionless_constants_only,
        ):
            loss += dimensional_constraint_penalty

    return (loss, complexity, True, "")


def _tournament_select(
    population: list[Node],
    pop_data: list[tuple[float, int, bool, str]],
    rng: np.random.Generator,
    tournament_n: int,
    parsimony: float = 0.0,
    frequency: list[int] | None = None,
    use_frequency: bool = False,
    use_frequency_in_tournament: bool = False,
    adaptive_parsimony_scaling: float = 0.0,
) -> int:
    n = len(population)
    indices = list(rng.integers(0, n, tournament_n))

    if use_frequency and frequency is not None:
        for idx in indices:
            frequency[idx] += 1

    def _effective_p(i: int) -> float:
        if (
            adaptive_parsimony_scaling > 0.0
            and use_frequency
            and frequency is not None
        ):
            max_freq = max(frequency) if frequency else 0
            if max_freq > 0:
                freq_ratio = frequency[i] / max_freq
                return parsimony * max(
                    0.0, 1.0 - adaptive_parsimony_scaling * freq_ratio,
                )
        return parsimony

    def key(i: int) -> tuple[float, int, str]:
        loss, complexity, valid, _ = pop_data[i]
        if not valid:
            return (float("inf"), complexity, population[i].structural_hash())
        score = loss + complexity * _effective_p(i)
        if use_frequency_in_tournament and use_frequency and frequency is not None:
            score += np.log(1.0 + frequency[i]) * 0.01
        return (score, complexity, population[i].structural_hash())

    return min(indices, key=key)


def _find_replace_slot(
    pop_data: list[tuple[float, int, bool, str]],
    child_loss: float,
    child_complexity: int,
    child_valid: bool,
    parsimony: float = 0.0,
) -> int | None:
    if not child_valid:
        return None

    n = len(pop_data)

    for i in range(n):
        if not pop_data[i][2]:
            return i

    def _cost(i: int) -> float:
        return pop_data[i][0] + pop_data[i][1] * parsimony

    worst_i = max(
        range(n),
        key=lambda i: (_cost(i), pop_data[i][1]),
    )

    return worst_i


def _maybe_refresh_population(
    population: list[Node],
    pop_data: list[tuple[float, int, bool, str]],
    rng: np.random.Generator,
    binary_ids: list[str],
    unary_ids: list[str],
    n_features: int,
    gen_depth: int,
    X: np.ndarray,
    y: np.ndarray,
    maxsize: int,
    maxdepth: int,
    loss_fn: LossFn | None = None,
    loss_function: Callable | None = None,
    loss_function_expression: Callable | None = None,
    constraints: dict[str, int | tuple[int, ...]] | None = None,
    nested_constraints: dict[str, dict[str, int]] | None = None,
    weights: np.ndarray | None = None,
    loss_scale: float = 1.0,
    const_weight: int = 1,
    op_weight: int = 1,
    var_weight: int = 1,
    complexity_mapping_val: Callable | None = None,
    fraction_replaced: float = 0.1,
    parsimony: float = 0.0,
    eval_count: list[int] | None = None,
) -> None:
    n = len(population)
    replace_count = max(1, int(n * fraction_replaced))

    def _cost(i: int) -> float:
        return pop_data[i][0] + pop_data[i][1] * parsimony

    worst_indices = sorted(
        range(n),
        key=lambda i: (_cost(i), pop_data[i][1]),
        reverse=True,
    )[:replace_count]
    for idx in worst_indices:
        expr = generate_expression(
            rng, binary_ids, unary_ids, n_features, gen_depth,
        )
        loss, complexity, valid, reason = _full_evaluate(
            expr, X, y, maxsize, maxdepth, loss_fn=loss_fn,
            loss_function_expression=loss_function_expression,
            loss_function=loss_function,
            constraints=constraints, nested_constraints=nested_constraints,
            weights=weights, loss_scale=loss_scale,
            const_weight=const_weight, op_weight=op_weight, var_weight=var_weight,
            complexity_mapping=complexity_mapping_val,
            eval_count=eval_count,
        )
        population[idx] = expr
        pop_data[idx] = (loss, complexity, valid, reason)


def _preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    denoise: bool = False,
    select_k_features: int | None = None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if select_k_features:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel
        except ImportError:
            return X, y
        # Use a non-negative integer seed from the Generator
        seed_val = int(rng.integers(0, 2**31))
        clf = RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=seed_val,
        )
        clf.fit(X, y.ravel() if y.ndim > 1 else y)
        selector = SelectFromModel(
            clf, threshold=-np.inf, max_features=select_k_features, prefit=True,
        )
        mask = selector.get_support(indices=False)
        X = X[:, mask]

    if denoise:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        except ImportError:
            return X, y
        seed_val = int(rng.integers(0, 2**31))
        gp_kernel = RBF(np.ones(X.shape[1])) + WhiteKernel(1e-1) + ConstantKernel()
        y_flat = y.ravel() if y.ndim > 1 else y
        gpr = GaussianProcessRegressor(
            kernel=gp_kernel, n_restarts_optimizer=50, random_state=seed_val,
        )
        gpr.fit(X, y_flat)
        y = gpr.predict(X)
        if y.ndim == 1 and y_flat.ndim == 2:
            y = y.reshape(-1, 1)

    return X, y
