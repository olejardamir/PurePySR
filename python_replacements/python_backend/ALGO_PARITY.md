# Algorithm Parity Checklist

Status of Julia (`SymbolicRegression.jl/src/`) components and their Python equivalents.

| Julia Component | Python Equivalent | Status |
|---|---|---|
| **RegularizedEvolution.jl** | `regularized_evolution.py` | Done |
| — `reg_evol_cycle` | `run_regularized_evolution()` | Done |
| — Two-child crossover | Two children per crossover cycle | Done |
| — Crossover probability | `crossover_prob` from `BackendOptions.crossover_probability` | Done |
| — Geometric tournament | `_geometric_tournament_winner()` | Done |
| — Simulated annealing | Temperature schedule + `exp(-delta/(T*alpha))` | Done |
| — Frequency-based acceptance | `prob_change *= old_freq / new_freq` | Done |
| — Early stopping (convergence) | `t - best_cycle > ncycles_per_iteration` → break | Done |
| — `condition_mutation_weights!` per cycle | Deep-copied MW conditioned every cycle | Done |
| **Mutate.jl** | `search.py` | Done |
| — `mutate!` (single-type dispatch) | `mutate()` taking `mutation_type` param, 10-attempt retry loop | Done |
| — `crossover_trees` | `crossover_trees()` with mutual subtree swap, 10-attempt retry | Done |
| — `sample_mutation` | `sample_mutation()` — Gumbel-max sampling from raw weights | Done |
| — `condition_mutation_weights!` | `condition_mutation_weights()` — zeros inapplicable ops by type | Done |
| **MutationFunctions.jl** | `search.py` | Done |
| — `mutate_constant` | `mutate_constant()` — multiplicative `maxChange^rand() * T` factor | Done |
| — `mutate_operator` (point) | `mutate_operator()` — picks random OpNode, replaces with any operator | Done |
| — `mutate_feature` | `mutate_feature()` — picks its own VarNode internally | Done |
| — `append_random_op` | `add_node()` — 50/50 leaf-expansion vs root-wrap | Done |
| — `prepend_random_op` | `insert_node()` — inserts parent above random node | Done |
| — `insert_random_op` | `insert_node()` — same function, supports any arity | Done |
| — `delete_random_op` | `delete_node()` — picks random child, not `children[0]` | Done |
| — `crossover_trees` | `crossover_trees()` — deep-copies both parents before swap | Done |
| — `randomly_rotate_tree` | `rotate_tree()` | Done |
| — `randomize_tree` | `randomize_tree()` | Done |
| — `do_nothing` | RE loop intercepts before `mutate()` call | Done |
| — `simplify` (as weighted mutation) | RE loop intercepts → `simplify_tree(parent)` | Done |
| — `optimize` (as weighted mutation) | RE loop intercepts → `optimize_tree(parent)` | Done |
| — `backsolve` | `backsolve_rewrite_random_node()` in `search.py` — STLSQ, basis library from population, weighted-sum tree | Done |
| — `form_connection` | `_form_connection()` in `search.py` — copies a non-descendant node as a child | Done |
| — `break_connection` | `_break_connection()` in `search.py` — replaces child with random terminal | Done |
| **MutationWeights.jl** | `mutation_weights.py` | Done |
| — 15 mutation weight fields | `MutationWeights` dataclass matching Julia exactly | Done |
| — Julia's tuned defaults | Fields initialized with Julia's tuned values (e.g. `mutate_operator=3.63`) | Done |
| — Zeroing conditioning | `condition_mutation_weights()` — zeros inapplicable ops (Julia parity) | Done |
| — `sample_mutation` | `sample_mutation()` — raw weights, no normalization, Gumbel-max | Done |
| — No reward/penalize/normalize | Removed — Julia doesn't do dynamic weight adaptation | Done |
| — Crossover not in weights | `crossover_probability` controls crossover, not mutation weights | Done |
| **HallOfFame.jl** | `hof.py` | Done |
| — Per-complexity storage | `hof.py` with `.best()` / `.consider()` | Done |
| — `maybe_insert_into_hof!` | `hall_of_fame.consider()` | Done |
| — `calculate_pareto_frontier` | `HallOfFame.calculate_pareto_frontier()` | Done |
| — `format_hall_of_fame` | `HallOfFame.compute_scores()` — direct + zero-centered | Done |
| — `string_dominating_pareto_curve` | `HallOfFame.string_dominating_pareto_curve()` | Done |
| **Population.jl** | (inline in RE loop) | Done |
| — `best_of_sample` | `_geometric_tournament_winner()` | Done |
| — Aging-based replacement | `birth` array, replace oldest | Done |
| — Restart / migration | Bumper restart + HOF/seed migration between iterations | Done |
| **PopMember.jl** | Tuple `(node, loss, complexity, valid, reason, birth, frequency)` | Done |
| **SingleIteration.jl** | Outer loop in `backend.py` | Done |
| — `s_r_cycle` | `equation_search()` / `run_regularized_evolution()` | Done |
| — `optimize_and_simplify_population` | `_optimize_and_simplify_population` — per-iteration and final | Done |
| — `should_simplify` | `simplify_expression()` in `expr.py` — algebraic simplification | Done |
| — `bumper` | Python-only: restarts stale members when stagnant (Julia's `bumper` is a Bumper.jl memory allocator flag, not a restart mechanism) | Python addition |
| — `warmup_maxsize_by` | `_get_cur_maxsize()` + dynamic `_cur_maxsize` in RE loop | Done |
| **Migration.jl** | `_migrate_into_population()` in `regularized_evolution.py` | Done |
| — HOF migration | `hof_migration` injects dominating entries into pop between iterations | Done |
| — Seed member migration | `fraction_replaced_guesses` via migration | Done |
| **AdaptiveParsimony.jl** | `running_statistics.py` | Done |
| **ConstantOptimization.jl** | `constant_optimization.py` | Done |
| — L-BFGS-B default | Default algorithm changed from Nelder-Mead | Done |
| — Normal restart perturbation | `value *= 1 + 0.5 * randn()` | Done |
| — `autodiff_backend` | `gradients.py` — forward-mode AD, exact gradients to scipy | Done |
| **Batching** | One batch per RE run | Done |
| **Options** | `options.py` / `option_gate.py` | All supported |
| — `crossover_probability` | Wired via `BackendOptions.crossover_probability` → `RegularizedEvolutionConfig` | Done |
| — `autodiff_backend` | Forward-mode AD via `gradients.py` — exact gradients to scipy | Done |
| — `should_simplify` | Algebraic simplification via `simplify_expression()` | Done |
| — `bumper` | Python-only: restarts stale members when stagnant (Julia's `bumper` is a Bumper.jl memory allocator flag, not a restart mechanism) | Python addition |
| — `warmup_maxsize_by` | Gradually increase maxsize from 3→maxsize | Done |
| — `hof_migration` | Inject HOF members into population between iterations | Done |
| — `mutation_weight_factor` | No-op; mutation weights always active (system uses Julia's zeroing, not scaling) | Python simplification |
| **Multi-Output Support** | `_run_multi_output()` in `backend.py` — separate RE loop per output column | Done |
| **Multi-Population (Island Model)** | `_run_multi_pop_search()` in `backend.py` — independent populations per iteration, cross-pollination via `_migrate_into_population()` | Done |
| **Backend Dispatch** | `backend` parameter in `PySRRegressor.__init__` + `PYSR_BACKEND` env var | Done |
| **`_run_python_backend()`** | Converts PySR params → `BackendOptions` → `PythonSRBackend.equation_search()` | Done |
| **Backsolve.jl** | `search.py` — `_solve_library`, `stlsq`, `build_basis_library`, `combine_trees_weighted_sum`, `fit_sparse_expression`, `_eval_inverse_tree_array`, `backsolve_rewrite_random_node` | Done |
