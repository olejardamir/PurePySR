from __future__ import annotations

import io
import json
import os
import pathlib
import types
from typing import Any

import numpy as np
import pandas as pd

from python_backend.backend import PythonSRBackend
from python_backend.option_gate import check_option_coverage, REJECTED
from python_backend.options import BackendOptions
from python_backend.trace import canonical_json, dump_jsonl


class JuliaError(Exception):
    """Exception raised for Julia errors in the stub."""
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"JuliaError: {message}")


JuliaError.__name__ = "JuliaError"


class AnyValue:
    """Stub for juliacall.AnyValue - opaque wrapper for Julia values."""
    def __init__(self, value: Any = None):
        self._value = value


class VectorValue:
    """Stub for juliacall.VectorValue - wrapper for Julia Vectors."""
    def __init__(self, value: Any = None):
        self._value = value


def convert(type_spec: Any, value: Any) -> Any:
    """Stub for juliacall.convert - identity function."""
    return value


class _Version:
    major = 1
    minor = 10
    patch = 0


class _JuliaFunc:
    """A callable that captures the original seval code and raises on use."""
    def __init__(self, code: str):
        self._code = code
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise JuliaError(
            f"Stub cannot evaluate Julia code: {self._code!r}"
        )
    def __repr__(self) -> str:
        return f"JuliaFunc({self._code[:60]})"


class _TypeRef:
    """A Julia type reference like Array, Dict, Float64."""
    def __init__(self, name: str = "TypeRef"):
        self._name = name
    def __getitem__(self, key: Any) -> _TypeRef:
        return _TypeRef(f"{self._name}[{key}]")
    def __repr__(self) -> str:
        return self._name


import pickle as _pickle


class _Serialization:
    @staticmethod
    def serialize(buf: Any, obj: Any) -> None:
        try:
            pickled = _pickle.dumps(obj)
        except Exception:
            pickled = _pickle.dumps({"stub": True, "type": type(obj).__name__})
        buf.write(pickled)
    @staticmethod
    def deserialize(buf: Any) -> Any:
        try:
            data = buf.getvalue() if hasattr(buf, "getvalue") else bytes(buf)
            return _pickle.loads(data)
        except Exception:
            return None


class _BaseModule:
    class UUID:
        @staticmethod
        def __call__(uuid_str: str) -> str:
            return uuid_str


class _Pkg:
    @staticmethod
    def add(**kwargs: Any) -> None:
        pass
    @staticmethod
    def dependencies() -> dict:
        return {}
    @staticmethod
    def resolve() -> None:
        pass


class _DummyModule(types.ModuleType):
    def __init__(self, name: str = "Dummy"):
        super().__init__(name)
        self._name = name


class _Options:
    """Stub for SymbolicRegression.Options - stores all keyword args as attrs."""
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
    def __repr__(self) -> str:
        return f"SymbolicRegression.Options({len(self.__dict__)} fields)"


class _OpRef:
    """A callable that knows its original operator token."""
    def __init__(self, code: str):
        self._code = code
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None
    def __repr__(self) -> str:
        return f"OpRef({self._code!r})"


class _DummyType:
    """A no-op type that can be instantiated or called with any args."""
    def __new__(cls, *args: Any, **kwargs: Any) -> _DummyType:
        return object.__new__(cls)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __call__(self, *args: Any, **kwargs: Any) -> _DummyType:
        return _DummyType()
    def __getattr__(self, name: str) -> Any:
        return _make_noop_callable()
    def __repr__(self) -> str:
        return "DummyValue"
    def __bool__(self) -> bool:
        return True
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _DummyType)


def _make_noop_callable() -> Any:
    """Return a callable that accepts anything and returns None."""
    def _noop(*args: Any, **kwargs: Any) -> None:
        return None
    return _noop


class _SymbolicRegression:
    """Stub for the SymbolicRegression Julia module - bridges to PythonSRBackend."""

    class ExpressionSpec:
        """Stub for SymbolicRegression.ExpressionSpec."""
        pass

    class SearchUtilsModule:
        @staticmethod
        def generate_run_id() -> str:
            return "stub-run-id"

    class MutationWeights:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    Options = _Options

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return _DummyType()

    @staticmethod
    def get_logger(logger: Any) -> Any:
        return logger

    @staticmethod
    def equation_search(
        X: Any,
        y: Any,
        *,
        options: Any = None,
        niterations: int = 10,
        run_id: str = "stub-run",
        output_directory: str = "",
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Transpose back from Julia convention (n_features, n_samples)
        # to Python convention (n_samples, n_features)
        if X.shape[0] < X.shape[1]:
            X = X.T
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 0:
            y = y.reshape(-1)

        if options is None:
            options = _Options()
        out_dir = getattr(options, "output_directory", output_directory)
        if not out_dir:
            out_dir = pathlib.Path.cwd() / "pysr_output"
        out_path = pathlib.Path(out_dir) / str(run_id)
        out_path.mkdir(parents=True, exist_ok=True)

        be_opts = _to_backend_options(options, niterations, verbosity)
        backend = PythonSRBackend()
        weights = kwargs.get("weights", None)
        saved_state = kwargs.pop("saved_state", None)
        if saved_state is not None:
            if hasattr(saved_state, "_result"):
                saved_state = saved_state._result.get("saved_state")
            elif isinstance(saved_state, dict):
                saved_state = saved_state.get("saved_state")
        # Data-dependent pass-through: X_units, y_units
        extra_options: dict[str, Any] = {}
        for key in ("x_units", "y_units", "X_units", "Y_units"):
            val = kwargs.pop(key, None)
            if val is not None:
                extra_options[key.lower()] = val
        result = backend.equation_search(
            X, y, options=be_opts, weights=weights,
            saved_state=saved_state, extra_options=extra_options,
        )

        _write_hall_of_fame_csv(out_path, result)
        _write_checkpoint(out_path, result)
        _write_artifacts(out_path, result)

        return _ResultWrapper(result)


def _token_to_op_id(token: str) -> str:
    try:
        from python_backend.ops import TOKEN_TO_OP_ID
        return TOKEN_TO_OP_ID.get(token, token)
    except ImportError:
        return token


def _normalize_constraint(
    val: object,
) -> int | tuple[int, ...]:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (tuple, list)):
        # PySR may wrap constraints in nested tuples, e.g. ((2, 3),)
        # or pass a Julia VectorValue that looks like a tuple of tuples.
        parts = []
        for v in val:
            if isinstance(v, (tuple, list)):
                inner = tuple(int(x) for x in v)
                parts.append(inner[0] if len(inner) == 1 else inner)
            else:
                parts.append(int(v))
        if len(parts) == 1 and isinstance(parts[0], int):
            return parts[0]
        return tuple(parts)
    if hasattr(val, "_jl_tuple") or type(val).__name__ == "VectorValue":
        return _normalize_constraint(list(val))
    return int(val)


def _convert_elementwise_loss(val: object) -> str:
    s = str(val)
    if s.startswith("OpRef("):
        s = s[6:-1].strip("'\"")
    return s


def _to_backend_options(
    options: Any, niterations: int, verbosity: int,
) -> BackendOptions:
    """Convert SymbolicRegression Options + kwargs -> BackendOptions.

    Fallback defaults match PySR_custom/pysr/sr.py defaults.
    """
    pop_size = getattr(options, "npop", 27)
    maxsize = getattr(options, "maxsize", 30)
    maxdepth = getattr(options, "maxdepth", 10)
    tsel_n = getattr(options, "tournament_selection_n", 15)
    ncycles = getattr(options, "ncycles_per_iteration", 380)
    topn = getattr(options, "topn", 12)
    deterministic = getattr(options, "deterministic", True)

    def _resolve_op(val: Any) -> str:
        if isinstance(val, _OpRef):
            return val._code
        if hasattr(val, "_code"):
            return val._code
        s = str(val)
        if s.startswith("OpRef("):
            return s[6:-1].strip("'\"")
        return s

    binary: list[str] = []
    raw_binary = getattr(options, "binary_operators", None)
    if raw_binary is not None:
        binary = [_resolve_op(o) for o in raw_binary]

    unary: list[str] = []
    raw_unary = getattr(options, "unary_operators", None)
    if raw_unary is not None:
        unary = [_resolve_op(o) for o in raw_unary]

    if not binary and not unary:
        ops = getattr(options, "operators", None)
        if isinstance(ops, dict):
            for k, v in ops.items():
                key = int(k) if not isinstance(k, int) else k
                if key == 1:
                    unary = [_resolve_op(o) for o in (v if isinstance(v, (list, tuple)) else [v])]
                elif key == 2:
                    binary = [_resolve_op(o) for o in (v if isinstance(v, (list, tuple)) else [v])]

    if not binary:
        binary = ["+"]
    if not unary:
        unary = []

    elementwise_loss = getattr(options, "elementwise_loss", None)
    if elementwise_loss is not None:
        elementwise_loss = _convert_elementwise_loss(elementwise_loss)

    # ── constraints & nested_constraints ──────────────────────────────
    constraints: dict[str, int | tuple[int, ...]] | None = None
    raw_constraints = getattr(options, "constraints", None)
    if isinstance(raw_constraints, dict):
        constraints = {}
        for k, v in raw_constraints.items():
            op_id = _token_to_op_id(str(k))
            constraints[op_id] = _normalize_constraint(v)

    nested_constraints: dict[str, dict[str, int]] | None = None
    raw_nested = getattr(options, "nested_constraints", None)
    if isinstance(raw_nested, dict):
        nested_constraints = {}
        for outer_k, inner_dict in raw_nested.items():
            outer_id = _token_to_op_id(str(outer_k))
            inner_map: dict[str, int] = {}
            for inner_k, v in inner_dict.items():
                inner_id = _token_to_op_id(str(inner_k))
                inner_map[inner_id] = int(v)
            nested_constraints[outer_id] = inner_map

    # ── Scalar passthrough options ─────────────────────────────────────
    scalar_keys = (
        "alpha", "annealing", "loss_scale", "model_selection",
        "parsimony", "perturbation_factor", "probability_negate_constant",
        "should_simplify", "tournament_selection_p", "use_frequency",
        "use_frequency_in_tournament", "verbosity", "warmup_maxsize_by",
        # Population / mutation rates:
        "fraction_replaced", "fraction_replaced_hof",
        "fraction_replaced_guesses", "optimize_probability",
        # Optimizer configuration:
        "optimizer_algorithm", "optimizer_iterations",
        "optimizer_nrestarts", "optimizer_f_calls_limit",
        # Budget / speed:
        "max_evals", "timeout_in_seconds", "fast_cycle", "turbo",
        "early_stop_condition", "precision", "expression_spec", "update",
        "hof_migration", "optimize_constants",
        # Operators convenience:
        "operators",
        # Complexity customization:
        "complexity_of_constants", "complexity_of_operators",
        "complexity_of_variables", "complexity_mapping",
        # Frequency-based parsimony:
        "adaptive_parsimony_scaling",
        # Batching:
        "batching", "batch_size",
        # Warm start:
        "warm_start",
        # Julia-specific performance (silently ignored):
        "bumper", "turbo",
        # Dimensional constraints:
        "dimensional_constraint_penalty", "dimensionless_constants_only",
    )
    scalar_defaults = {
        "alpha": 3.17,
        "annealing": False,
        "loss_scale": 1.0,
        "model_selection": "accuracy",
        "parsimony": 0.0,
        "perturbation_factor": 1.0,
        "probability_negate_constant": 0.0,
        "should_simplify": False,
        "tournament_selection_p": 1.0,
        "use_frequency": False,
        "use_frequency_in_tournament": False,
        "verbosity": 0,
        "warmup_maxsize_by": 0,
        "fraction_replaced": 0.1,
        "fraction_replaced_hof": 0.1,
        "fraction_replaced_guesses": 0.0,
        "optimize_probability": 1.0,
        "optimizer_algorithm": "NelderMead",
        "optimizer_iterations": 8,
        "optimizer_nrestarts": 2,
        "optimizer_f_calls_limit": 0,
        "max_evals": 0,
        "timeout_in_seconds": 0,
        "fast_cycle": False,
        "turbo": False,
        "early_stop_condition": "",
        "precision": 16,
        "expression_spec": None,
        "update": True,
        "hof_migration": True,
        "optimize_constants": False,
        "operators": None,
        "complexity_of_constants": 1,
        "complexity_of_operators": 1,
        "complexity_of_variables": 1,
        "complexity_mapping": None,
        "adaptive_parsimony_scaling": 0.0,
        "batching": False,
        "batch_size": 0,
        "warm_start": False,
        "bumper": False,
        "turbo": False,
        "dimensional_constraint_penalty": None,
        "dimensionless_constants_only": False,
    }
    scalar = {}
    for k in scalar_keys:
        raw = getattr(options, k, None)
        if raw is None:
            scalar[k] = scalar_defaults[k]
        else:
            scalar[k] = raw
    # Sanitize loss_scale: PySR accepts string strategies like "linear"/"log"
    # which are not simple float multipliers. Default to 1.0 for those.
    if not isinstance(scalar.get("loss_scale"), (int, float)):
        scalar["loss_scale"] = 1.0
    # Sanitize early_stop_condition: PySR accepts a callable or numeric string.
    # The backend only supports a numeric threshold string.
    if not isinstance(scalar.get("early_stop_condition"), (str, int, float)):
        scalar["early_stop_condition"] = ""
    else:
        scalar["early_stop_condition"] = str(scalar["early_stop_condition"])

    # ── Callable loss functions ──────────────────────────────────────
    loss_function = getattr(options, "loss_function", None)
    loss_function_expression = getattr(options, "loss_function_expression", None)

    # ── Guesses / seed expressions ──────────────────────────────────
    guesses: list[str] | None = None
    raw_guesses = getattr(options, "guesses", None)
    if raw_guesses is not None:
        # Julia may pass guesses as a Vector{Vector{String}} or Vector{String}
        if isinstance(raw_guesses, (list, tuple)):
            if raw_guesses and isinstance(raw_guesses[0], (list, tuple)):
                guesses = [str(g) for sub in raw_guesses for g in sub]
            else:
                guesses = [str(g) for g in raw_guesses if isinstance(g, str)]

    # ── Alias handling ────────────────────────────────────────────────
    # should_optimize_constants is a deprecated alias for optimize_constants
    raw_should_opt = getattr(options, "should_optimize_constants", None)
    raw_opt_const = getattr(options, "optimize_constants", None)
    if raw_should_opt is not None and raw_opt_const is None:
        scalar["optimize_constants"] = bool(raw_should_opt)

    # migration=False disables HOF migration in single-population mode
    raw_migration = getattr(options, "migration", None)
    if raw_migration is not None and not raw_migration:
        scalar["hof_migration"] = False

    # ── weights passthrough ───────────────────────────────────────────
    # weights is extracted from extra_options in equation_search, not passed here.

    return BackendOptions(
        binary_operators=binary,
        unary_operators=unary,
        niterations=int(niterations),
        population_size=int(pop_size),
        maxsize=int(maxsize),
        maxdepth=int(maxdepth),
        tournament_selection_n=int(tsel_n),
        deterministic=bool(deterministic),
        ncycles_per_iteration=int(ncycles),
        topn=int(topn),
        elementwise_loss=elementwise_loss,
        constraints=constraints,
        nested_constraints=nested_constraints,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
        guesses=guesses,
        **scalar,
    )


def _canonical_to_equation(expr: str) -> str:
    """Convert a backend canonical expression to a sympy-parseable string.

    Handles:
      ``var[N]``          → ``xN``
      ``const[float64:hex]`` → decimal literal
      ``op_id(a,b)``       → 'op(a,b)' with op mapped via *OP_ID_MAP*
    """
    _OP_ID_MAP: dict[str, str] = {
        "sr.arith.add_v1": "({}+{})",
        "sr.arith.sub_v1": "({}-{})",
        "sr.arith.mul_v1": "({}*{})",
        "sr.arith.protected_div_v1": "({}/{})",
        "sr.math.pow_v1": "({}**{})",
        "sr.math.sin_v1": "sin({})",
        "sr.math.cos_v1": "cos({})",
        "sr.math.abs_v1": "abs({})",
        "sr.math.safe_log_v1": "log(abs({}))",
        "sr.arith.less_v1": "({}<{})",
    }

    expr = expr.strip()

    # var[N]
    if expr.startswith("var[") and expr.endswith("]"):
        idx = expr[len("var["):-1]
        try:
            return f"x{int(idx)}"
        except ValueError:
            return "x0"

    # const[float64:hex]
    if expr.startswith("const[float64:") and expr.endswith("]"):
        hex_val = expr[len("const[float64:"):-1]
        try:
            return str(float.fromhex(hex_val))
        except (ValueError, TypeError):
            return "0.0"

    # op_id(arg1,arg2,...)
    paren_idx = expr.find("(")
    if paren_idx != -1 and expr.endswith(")"):
        op_id = expr[:paren_idx]
        args_raw = expr[paren_idx + 1:-1]
        args = _split_args(args_raw)
        converted = [_canonical_to_equation(a) for a in args]
        template = _OP_ID_MAP.get(op_id)
        if template is not None:
            return template.format(*converted)
        # fallback: just join with space
        return f"{op_id.split('.')[-1]}({','.join(converted)})"

    return "x0"


def _split_args(raw: str) -> list[str]:
    """Split comma-separated arguments respecting nested parens."""
    depth = 0
    parts: list[str] = []
    cur: list[str] = []
    for ch in raw:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _write_hall_of_fame_csv(out_path: pathlib.Path, result: dict) -> None:
    """Write hall_of_fame.csv in the format PySR_custom expects.

    Keeps only the best entry per complexity level (lowest loss),
    sorted by ascending complexity.  Clamps zero-loss entries to
    ``1e-16`` so that PySR_custom's ``calculate_scores()`` does not
    divide by zero.

    Includes a ``sympy_format`` column computed from the canonical
    expression via :func:`node_to_sympy`.
    """
    from python_backend.expr import parse_canonical, node_to_sympy

    _LOSS_EPS = 1e-16
    hof = result.get("hall_of_fame", [])
    n_features = result.get("dataset_manifest", {}).get("n_features", 1)
    variable_names = [f"x{i}" for i in range(n_features)]
    rows = []
    for i, entry in enumerate(hof):
        loss = entry.get("loss", 0.0)
        loss = max(float(loss), _LOSS_EPS)
        complexity = entry.get("complexity", 1)
        score = -float(loss)
        canonical = str(entry.get("canonical_expression", ""))
        equation = _canonical_to_equation(canonical)
        try:
            node = parse_canonical(canonical)
            sympy_expr = node_to_sympy(node, variable_names=variable_names)
            sympy_format = str(sympy_expr)
        except Exception:
            sympy_format = ""
        rows.append({
            "Complexity": complexity,
            "Loss": loss,
            "Score": score,
            "Equation": equation,
            "sympy_format": sympy_format,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["Complexity", "Loss"], ascending=[True, True])
    df = df.drop_duplicates(subset=["Complexity"], keep="first")
    csv_path = out_path / "hall_of_fame.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _write_checkpoint(out_path: pathlib.Path, result: dict) -> None:
    """Write a checkpoint.pkl with saved_state for warm-start."""
    import pickle
    saved_state = result.get("saved_state")
    ckpt_path = out_path / "checkpoint.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump(saved_state if saved_state is not None else {"stub": True}, f)


def _write_artifacts(out_path: pathlib.Path, result: dict) -> None:
    """Write governed artifacts into *out_path*.

    Written files match the ``run_and_write_artifacts`` layout so they
    can be validated with ``cli_validate_trace.py``.
    """
    trace_path = out_path / "trace.jsonl"
    dump_jsonl(result.get("trace_records", []), str(trace_path))

    digests_path = out_path / "digests.json"
    with open(digests_path, "w") as f:
        json.dump(result.get("digests", {}), f, sort_keys=True)

    policy_path = out_path / "policy.json"
    with open(policy_path, "w") as f:
        json.dump(result.get("policy_dict", {}), f, sort_keys=True)

    dataset_manifest = result.get("dataset_manifest", {})
    digests = result.get("digests", {})
    dataset_digest = digests.get("dataset_digest", "")
    dataset_path = out_path / "dataset.json"
    dataset_out = dict(dataset_manifest)
    dataset_out["dataset_digest"] = dataset_digest
    with open(dataset_path, "w") as f:
        json.dump(dataset_out, f, sort_keys=True)

    hof = result.get("hall_of_fame", [])
    archive_entries = [
        {"hash": e["hash"], "loss": str(e["loss"]), "complexity": e["complexity"]}
        for e in hof
    ]
    archive_path = out_path / "archive.json"
    archive_path.write_text(canonical_json(archive_entries))


class _ResultWrapper:
    """Minimal wrapper around the python_backend result dict."""
    def __init__(self, result: dict):
        self._result = result
    def get(self, key: str, default: Any = None) -> Any:
        return self._result.get(key, default)


_LOADED_MODULES: dict[str, Any] = {}


class _Main:
    _MAIN_TYPE = "stub"
    VERSION = _Version()

    def seval(self, code: str) -> Any:
        code = code.strip()

        if code.startswith("using "):
            rest = code[5:].strip()
            mod_name = rest.split(":")[0].split()[0].strip()
            _LOADED_MODULES[mod_name] = _DummyModule(mod_name)
            return lambda: None

        if "Base.active_project()" in code:
            return "/tmp/py-stub-project"

        if "Val{" in code and "where x" in code:
            return lambda x: x

        if "isa Function" in code:
            return lambda op: callable(op)

        if "NamedTuple{(:class,)}" in code:
            return lambda tup: {"class": tup[0]}

        if "OperatorEnum" in code:
            return lambda ops_dict: ops_dict

        if "create_operator_enum" in code:
            return lambda ops_dict: {
                1: tuple(ops_dict.get(1, ())),
                2: tuple(ops_dict.get(2, ())),
            }

        if "get_logger" in code:
            return lambda logger: logger

        if "string(typeof" in code:
            return lambda x: str(type(x).__name__)

        if code == "nothing":
            return None

        if code in ("+", "-", "*", "/", "^", "sin", "cos", "abs", "log", "D", "<"):
            return _OpRef(code)

        if code == "plus":
            return _OpRef("+")
        if code == "sub":
            return _OpRef("-")
        if code == "mult":
            return _OpRef("*")
        if code == "div":
            return _OpRef("/")
        if code == "pow":
            return _OpRef("pow")

        if code == "Serialization":
            return _Serialization()

        return _OpRef(code) if code.isidentifier() or code in ("+", "-", "*", "/", "<") else _make_noop_callable()

    def __getattr__(self, name: str) -> Any:
        if name == "SymbolicRegression":
            return _SymbolicRegression()
        if name == "Serialization":
            return _Serialization()
        if name == "PythonCall":
            return _DummyModule("PythonCall")
        if name == "Pkg":
            return _Pkg()
        if name == "Base":
            return _BaseModule()
        if name == "Dict":
            return lambda pairs=None: dict(pairs) if pairs is not None else {}
        if name in ("Array", "Float64", "Float32", "Vector", "Matrix"):
            return _TypeRef(name)
        if name == "Symbol":
            return lambda x: x
        if name == "NamedTuple":
            return lambda d=None: d if d is not None else {}
        if name == "Pair":
            return lambda k, v: (k, v)
        if name == "IOBuffer":
            return io.BytesIO
        if name == "take_b":
            return lambda buf: buf.getvalue()
        if name == "write":
            def _w(buf: Any, data: Any) -> None:
                if not hasattr(buf, "write"):
                    return
                if isinstance(data, np.ndarray):
                    buf.write(data.tobytes())
                else:
                    buf.write(data)
            return _w
        if name == "seekstart":
            return lambda buf: buf.seek(0) if hasattr(buf, "seek") else None
        if name in ("applicable", "haskey"):
            return lambda *a, **kw: True
        if name == "collect":
            return lambda iterable: list(iterable)
        if name == "methods":
            return lambda func: []
        if name == "typeof":
            return lambda x: _TypeRef(type(x).__name__)
        if name == "first":
            return lambda x: x[0] if x else None
        if name == "last":
            return lambda x: x[-1] if x else None
        if name == "close":
            return lambda x: None
        return _make_noop_callable()


Main = _Main()
jl = Main
