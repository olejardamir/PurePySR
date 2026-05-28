# Technical Parity Matrix — Python-Only Backend

_Generated from `python_backend/option_gate.py` coverage table._

## Status Legend
| Label | Meaning |
|---|---|
| **Supported** | Option is read and acted on by the Python backend |
| **Rejected** | Option is blocked by PySR_custom with a clear error before search |
| **Ignored** | Option is accepted but has no effect; a warning is emitted |
| **Pass-through** | Option passes through to search without backend interpretation |

## Full Matrix

| Status | Option |
|---|---|
| Supported | `adaptive_parsimony_scaling` |
| Supported | `alpha` |
| Supported | `annealing` |
| Rejected | `autodiff_backend` |
| Supported | `backsolve` |
| Supported | `batch_size` |
| Supported | `batching` |
| Supported | `binary_operators` |
| Ignored | `bumper` |
| Rejected | `cluster_manager` |
| Supported | `complexity_mapping` |
| Supported | `complexity_of_constants` |
| Supported | `complexity_of_operators` |
| Supported | `complexity_of_variables` |
| Supported | `constraints` |
| Supported | `crossover_probability` |
| Pass-through | `delete_tempfiles` |
| Supported | `denoise` |
| Supported | `deterministic` |
| Supported | `dimensional_constraint_penalty` |
| Supported | `dimensionless_constants_only` |
| Supported | `early_stop_condition` |
| Supported | `elementwise_loss` |
| Rejected | `expression_spec` |
| Pass-through | `extra_jax_mappings` |
| Pass-through | `extra_sympy_mappings` |
| Pass-through | `extra_torch_mappings` |
| Ignored | `fast_cycle` |
| Supported | `fraction_replaced` |
| Supported | `fraction_replaced_guesses` |
| Supported | `fraction_replaced_hof` |
| Supported | `guesses` |
| Rejected | `heap_size_hint_in_bytes` |
| Supported | `hof_migration` |
| Pass-through | `input_stream` |
| Pass-through | `logger_spec` |
| Supported | `loss_function` |
| Supported | `loss_function_expression` |
| Supported | `loss_scale` |
| Supported | `max_evals` |
| Supported | `maxdepth` |
| Supported | `maxsize` |
| Supported | `migration` |
| Pass-through | `model_selection` |
| Supported | `mutation_weight_factor` |
| Supported | `ncycles_per_iteration` |
| Supported | `nested_constraints` |
| Supported | `niterations` |
| Supported | `operators` |
| Supported | `optimize_constants` |
| Supported | `optimize_probability` |
| Supported | `optimizer_algorithm` |
| Supported | `optimizer_f_calls_limit` |
| Supported | `optimizer_iterations` |
| Supported | `optimizer_nrestarts` |
| Pass-through | `output_directory` |
| Pass-through | `output_jax_format` |
| Pass-through | `output_torch_format` |
| Ignored | `parallelism` |
| Supported | `parsimony` |
| Supported | `perturbation_factor` |
| Supported | `population_size` |
| Supported | `populations` |
| Ignored | `precision` |
| Pass-through | `print_precision` |
| Supported | `probability_negate_constant` |
| Ignored | `procs` |
| Pass-through | `progress` |
| Pass-through | `random_state` |
| Pass-through | `run_id` |
| Supported | `search_algorithm` |
| Supported | `select_k_features` |
| Supported | `should_optimize_constants` |
| Supported | `should_simplify` |
| Supported | `skip_mutation_failures` |
| Pass-through | `temp_equation_file` |
| Pass-through | `tempdir` |
| Supported | `timeout_in_seconds` |
| Supported | `topn` |
| Supported | `tournament_selection_n` |
| Supported | `tournament_selection_p` |
| Ignored | `turbo` |
| Supported | `unary_operators` |
| Supported | `update` |
| Pass-through | `update_verbosity` |
| Supported | `use_frequency` |
| Supported | `use_frequency_in_tournament` |
| Pass-through | `verbosity` |
| Supported | `warm_start` |
| Supported | `warmup_maxsize_by` |
| Supported | `weight_add_node` |
| Supported | `weight_delete_node` |
| Supported | `weight_do_nothing` |
| Supported | `weight_insert_node` |
| Supported | `weight_mutate_constant` |
| Supported | `weight_mutate_feature` |
| Supported | `weight_mutate_operator` |
| Supported | `weight_optimize` |
| Supported | `weight_randomize` |
| Supported | `weight_rotate_tree` |
| Supported | `weight_simplify` |
| Supported | `weight_swap_operands` |
| Supported | `weights` |
| Rejected | `worker_imports` |
| Rejected | `worker_timeout` |
