from __future__ import annotations

import warnings
from typing import Any

from python_backend.errors import SR_ERR_OPT_001, SR_ERR_OPT_004, SR_WARN_OPT_001

SUPPORTED = "supported"
REJECTED = "rejected_with_clear_error"
IGNORED = "accepted_but_ignored_with_warning"
PASS_THROUGH = "pass_through"

# ---------------------------------------------------------------------------
# COVERAGE_TABLE – exhaustive inventory of PySR_custom options
# ---------------------------------------------------------------------------
# Every parameter that PySRRegressor.__init__ (sr.py:948‑1061) can pass
# to the backend MUST appear exactly once.  Statuses correspond to EQC‑SR
# §3.3 vocabulary.
#
# Convention: alphabetically sorted for easy scanning.
# ---------------------------------------------------------------------------

_COVERAGE_TABLE_RAW: list[dict[str, str]] = [
    # ── BackendOptions fields (fully supported) ──────────────────────
    dict(option="binary_operators",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="deterministic",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="maxdepth",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="maxsize",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="ncycles_per_iteration",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="niterations",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="optimize_constants",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="population_size",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="search_algorithm",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="topn",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="tournament_selection_n",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="unary_operators",
         status=SUPPORTED, level="SR-L0", code=""),

    # ── Mutation / crossover weights (supported via BackendOptions) ──
    dict(option="crossover_probability",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="skip_mutation_failures",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_add_node",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_delete_node",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_do_nothing",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_insert_node",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_mutate_constant",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_mutate_feature",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_mutate_operator",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_optimize",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_randomize",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_rotate_tree",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_simplify",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="weight_swap_operands",
         status=SUPPORTED, level="SR-L0", code=""),

    # ── Backsolve ─────────────────────────────────────────────────────
    dict(option="backsolve",
         status=SUPPORTED, level="SR-L3", code=""),

    # ── Accepted-but-ignored with warning ───────────────────────────
    # These are not implemented in the Python backend. Options with
    # code="" are fully wired; those with SR_WARN_OPT_001 emit a
    # runtime warning when used (now that check_option_coverage() checks
    # the code field).
    dict(option="adaptive_parsimony_scaling",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="alpha",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="annealing",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="autodiff_backend",
         status=REJECTED, level="SR-L2", code=SR_ERR_OPT_001,
         error_message="autodiff_backend is disabled by PySR_custom for the Python backend"),
    dict(option="batch_size",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="batching",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="complexity_mapping",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="complexity_of_constants",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="complexity_of_operators",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="complexity_of_variables",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="constraints",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="dimensional_constraint_penalty",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="dimensionless_constants_only",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="early_stop_condition",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="elementwise_loss",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="expression_spec",
         status=REJECTED, level="SR-L2", code=SR_ERR_OPT_001,
         error_message="expression_spec templates require Julia; not supported in Python backend"),
    dict(option="fraction_replaced",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="fraction_replaced_hof",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="fraction_replaced_guesses",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="guesses",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="hof_migration",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="loss_function",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="loss_function_expression",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="loss_scale",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="max_evals",
         status=SUPPORTED, level="SR-L2", code="",
         warning_message=""),
    dict(option="migration",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="model_selection",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="mutation_weight_factor",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="nested_constraints",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="operators",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="optimize_probability",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="optimizer_algorithm",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="optimizer_f_calls_limit",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="optimizer_iterations",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="optimizer_nrestarts",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="parsimony",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="perturbation_factor",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="precision",
         status=IGNORED, level="SR-L3", code=SR_WARN_OPT_001,
         warning_message="precision is not configurable in this backend"),
    dict(option="probability_negate_constant",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="should_optimize_constants",
         status=SUPPORTED, level="SR-L0", code=""),
    dict(option="should_simplify",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="timeout_in_seconds",
         status=SUPPORTED, level="SR-L2", code="",
         warning_message=""),
    dict(option="tournament_selection_p",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="update",
         status=SUPPORTED, level="SR-L2", code=SR_WARN_OPT_001,
         warning_message="update parameter is not applicable"),
    dict(option="use_frequency",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="use_frequency_in_tournament",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="verbosity",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="warm_start",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="warmup_maxsize_by",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="weights",
         status=SUPPORTED, level="SR-L2", code=""),
    dict(option="denoise",
         status=SUPPORTED, level="SR-L3", code=""),
    dict(option="select_k_features",
         status=SUPPORTED, level="SR-L3", code=""),

    # ── Julia-specific (ignored with warning) ─────────────────────────
    dict(option="bumper",
         status=IGNORED, level="SR-L2", code=SR_WARN_OPT_001,
         warning_message="bumper is a Julia Bumper.jl flag with no Python equivalent"),
    dict(option="fast_cycle",
         status=IGNORED, level="SR-L3", code=SR_WARN_OPT_001,
         warning_message="fast_cycle is a Julia compilation optimisation with no Python equivalent"),
    dict(option="turbo",
         status=IGNORED, level="SR-L3", code=SR_WARN_OPT_001,
         warning_message="turbo is a Julia LoopVectorization.jl flag with no Python equivalent"),

    # ── Multi-population support ──────────────────────────────────
    dict(option="populations",
         status=SUPPORTED, level="SR-L2", code=""),

    # ── Explicitly rejected (backend cannot implement) ──────────────
    dict(option="cluster_manager",
         status=REJECTED, level="SR-L5", code=SR_ERR_OPT_001),
    dict(option="procs",
         status=IGNORED, level="SR-L4", code=SR_WARN_OPT_001),
    dict(option="parallelism",
         status=IGNORED, level="SR-L4", code=SR_WARN_OPT_001),
    dict(option="heap_size_hint_in_bytes",
         status=REJECTED, level="SR-L5", code=SR_ERR_OPT_001),
    dict(option="worker_timeout",
         status=REJECTED, level="SR-L5", code=SR_ERR_OPT_001),
    dict(option="worker_imports",
         status=REJECTED, level="SR-L5", code=SR_ERR_OPT_001),

    # ── Pass-through (handled by PySR_custom Python-side entirely)  ─
    dict(option="delete_tempfiles",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="extra_jax_mappings",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="extra_sympy_mappings",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="extra_torch_mappings",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="input_stream",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="logger_spec",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="output_directory",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="output_jax_format",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="output_torch_format",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="print_precision",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="progress",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="random_state",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="run_id",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="temp_equation_file",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="tempdir",
         status=PASS_THROUGH, level="SR-L0", code=""),
    dict(option="update_verbosity",
         status=PASS_THROUGH, level="SR-L0", code=""),
]

COVERAGE_TABLE: dict[str, dict[str, str]] = {
    entry["option"]: {
        "status": entry["status"],
        "required_capability_level": entry["level"],
        "error_or_warning_code": entry["code"],
        "warning_message": entry.get("warning_message", ""),
        "error_message": entry.get("error_message", ""),
    }
    for entry in _COVERAGE_TABLE_RAW
}

UNKNOWN_CODE = SR_ERR_OPT_004


def check_option_coverage(
    options_dict: dict[str, Any],
    *,
    pass_through: dict[str, Any] | None = None,
    known_as: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for key, value in options_dict.items():
        entry = COVERAGE_TABLE.get(key)
        if entry is None:
            results.append({
                "option": key,
                "status": "unknown",
                "code": UNKNOWN_CODE,
                "message": f"unrecognized option {key!r}",
            })
            continue

        if entry["status"] == SUPPORTED:
            code = entry["error_or_warning_code"]
            msg = entry.get("warning_message", "")
            if code:
                warnings.warn(
                    f"[{code}] option {key!r}: {msg or 'accepted but may be ignored in this backend'}",
                    stacklevel=2,
                )
            results.append({
                "option": key,
                "status": SUPPORTED,
                "code": code,
                "message": msg,
            })

        elif entry["status"] == REJECTED:
            code = entry["error_or_warning_code"]
            msg = entry.get("error_message", "")
            warnings.warn(
                f"[{code}] option {key!r}: {msg}",
                stacklevel=2,
            )
            results.append({
                "option": key,
                "status": REJECTED,
                "code": code,
                "message": (
                    f"option {key!r} requires capability level "
                    f"{entry['required_capability_level']} which is not available"
                ),
            })

        elif entry["status"] == IGNORED:
            code = entry["error_or_warning_code"]
            msg = entry.get("warning_message", "")
            if code:
                warnings.warn(
                    f"[{code}] option {key!r}: {msg or 'accepted but ignored'}",
                    stacklevel=2,
                )
            results.append({
                "option": key,
                "status": IGNORED,
                "code": code,
                "message": (
                    f"option {key!r} accepted but ignored "
                    f"(requires {entry['required_capability_level']})"
                ),
            })

    if pass_through is not None:
        for key in pass_through:
            entry = COVERAGE_TABLE.get(key)
            if entry is not None and entry["status"] == PASS_THROUGH:
                results.append({
                    "option": key,
                    "status": PASS_THROUGH,
                    "code": "",
                    "message": f"option {key!r} passed through to consumer",
                })
            else:
                results.append({
                    "option": key,
                    "status": "unknown",
                    "code": UNKNOWN_CODE,
                    "message": f"unrecognized pass-through option {key!r}",
                })

    return results
