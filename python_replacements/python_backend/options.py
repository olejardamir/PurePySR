from __future__ import annotations

import dataclasses
from typing import Any, Callable

from python_backend.errors import BackendOptionError, SR_ERR_OPT_002


@dataclasses.dataclass(frozen=True)
class BacksolveOptions:
    max_library_size: int = 500
    lambda_: float = 0.01
    max_iter: int = 10


@dataclasses.dataclass(frozen=True)
class BackendOptions:
    binary_operators: list[str] | None = None
    unary_operators: list[str] | None = None
    search_algorithm: str = "baseline"
    niterations: int = 10
    population_size: int = 33
    maxsize: int = 20
    maxdepth: int = 10
    tournament_selection_n: int = 10
    deterministic: bool = True
    ncycles_per_iteration: int = 50
    topn: int = 20
    optimize_constants: bool = False
    elementwise_loss: str | None = None
    constraints: dict[str, int | tuple[int, ...]] | None = None
    nested_constraints: dict[str, dict[str, int]] | None = None
    alpha: float = 3.17
    annealing: bool = False
    loss_scale: float = 1.0
    model_selection: str = "accuracy"
    parsimony: float = 0.0
    perturbation_factor: float = 1.0
    probability_negate_constant: float = 0.0
    mutation_weight_factor: float = 0.0  # no-op in Python backend (weights always active)
    crossover_probability: float = 0.1
    should_simplify: bool = False
    tournament_selection_p: float = 1.0
    use_frequency: bool = False
    use_frequency_in_tournament: bool = False
    verbosity: int = 0
    warmup_maxsize_by: int = 0
    # ── Population / mutation rates ──────────────────────────
    fraction_replaced: float = 0.1
    fraction_replaced_hof: float = 0.1
    fraction_replaced_guesses: float = 0.0
    optimize_probability: float = 1.0
    # ── Optimizer configuration ──────────────────────────────
    optimizer_algorithm: str = "L-BFGS-B"
    optimizer_iterations: int = 8
    optimizer_nrestarts: int = 2
    optimizer_f_calls_limit: int = 0
    autodiff_backend: bool = False
    # ── Budget / speed ───────────────────────────────────────
    max_evals: int = 0
    timeout_in_seconds: int = 0
    fast_cycle: bool = False
    turbo: bool = False
    early_stop_condition: str = ""
    # ── Output / reporting ───────────────────────────────────
    precision: int = 16
    expression_spec: str | None = None
    update: bool = True
    # ── Hall of Fame / migration ─────────────────────────────
    hof_migration: bool = True
    # ── Operators convenience ────────────────────────────────
    operators: dict | None = None
    skip_mutation_failures: bool = True
    # ── Seed expressions ─────────────────────────────────────
    guesses: list[str] | None = None
    # ── Callable losses ──────────────────────────────────────
    loss_function: Callable | None = None
    loss_function_expression: Callable | None = None
    # ── Complexity customization ─────────────────────────────
    complexity_of_constants: int = 1
    complexity_of_operators: int = 1
    complexity_of_variables: int = 1
    complexity_mapping: Callable | None = None
    # ── Batching ────────────────────────────────────────────
    batching: bool = False
    batch_size: int = 0  # 0 = use full dataset (auto)
    # ── Multi-population / island model ────────────────────
    populations: int = 1
    migration: bool = True
    # ── Warm start ──────────────────────────────────────────
    warm_start: bool = False
    # ── Performance (Julia-specific, silently ignored) ──────
    bumper: bool = False
    turbo: bool = False
    # ── Dimensional constraints ─────────────────────────────
    dimensional_constraint_penalty: float | None = None
    dimensionless_constants_only: bool = False
    # ── Data preprocessing ───────────────────────────────────
    denoise: bool = False
    select_k_features: int | None = None
    # ── Frequency-based parsimony ────────────────────────────
    adaptive_parsimony_scaling: float = 0.0
    # ── Backsolve ─────────────────────────────────────────────
    backsolve: BacksolveOptions | None = None

    def __post_init__(self) -> None:
        if self.search_algorithm not in ("baseline", "regularized_evolution"):
            raise BackendOptionError(
                SR_ERR_OPT_002,
                "search_algorithm must be 'baseline' or 'regularized_evolution'",
            )
        # Normalize operators dict into binary_operators / unary_operators
        if self.operators is not None:
            ops = self.operators
            binary: list[str] | None = None
            unary: list[str] | None = None
            if "binary" in ops:
                binary = list(ops["binary"])
            if "unary" in ops:
                unary = list(ops["unary"])
            if binary is None and unary is None:
                for k, v in ops.items():
                    key = int(k) if not isinstance(k, int) else k
                    vals = list(v) if isinstance(v, (list, tuple)) else [v]
                    if key == 1:
                        unary = vals
                    elif key == 2:
                        binary = vals
            if self.binary_operators is None and binary is not None:
                object.__setattr__(self, "binary_operators", binary)
            if self.unary_operators is None and unary is not None:
                object.__setattr__(self, "unary_operators", unary)
        if self.niterations < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "niterations must be >= 1",
            )
        if self.population_size < 2:
            raise BackendOptionError(
                SR_ERR_OPT_002, "population_size must be >= 2",
            )
        if self.maxsize < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "maxsize must be >= 1",
            )
        if self.maxdepth < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "maxdepth must be >= 1",
            )
        if self.tournament_selection_n < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "tournament_selection_n must be >= 1",
            )
        if self.ncycles_per_iteration < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "ncycles_per_iteration must be >= 1",
            )
        if self.topn < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "topn must be >= 1",
            )
        if not 0.0 <= self.fraction_replaced <= 1.0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "fraction_replaced must be in [0.0, 1.0]",
            )
        if not 0.0 <= self.fraction_replaced_hof <= 1.0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "fraction_replaced_hof must be in [0.0, 1.0]",
            )
        if not 0.0 <= self.fraction_replaced_guesses <= 1.0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "fraction_replaced_guesses must be in [0.0, 1.0]",
            )
        if not 0.0 <= self.optimize_probability <= 1.0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "optimize_probability must be in [0.0, 1.0]",
            )
        if self.optimizer_iterations < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "optimizer_iterations must be >= 0",
            )
        if self.optimizer_nrestarts < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "optimizer_nrestarts must be >= 0",
            )
        if self.optimizer_f_calls_limit < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "optimizer_f_calls_limit must be >= 0",
            )
        if self.max_evals < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "max_evals must be >= 0",
            )
        if self.timeout_in_seconds < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "timeout_in_seconds must be >= 0",
            )
        if self.precision < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "precision must be >= 1",
            )
        if self.complexity_of_constants < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "complexity_of_constants must be >= 1",
            )
        if self.complexity_of_operators < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "complexity_of_operators must be >= 1",
            )
        if self.complexity_of_variables < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "complexity_of_variables must be >= 1",
            )
        if self.select_k_features is not None and self.select_k_features < 1:
            raise BackendOptionError(
                SR_ERR_OPT_002, "select_k_features must be >= 1",
            )
        if self.adaptive_parsimony_scaling < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "adaptive_parsimony_scaling must be >= 0",
            )
        if self.batch_size < 0:
            raise BackendOptionError(
                SR_ERR_OPT_002, "batch_size must be >= 0",
            )
