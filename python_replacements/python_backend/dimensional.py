from __future__ import annotations

from typing import Any

import numpy as np

from python_backend.expr import ConstNode, Node, OpNode, VarNode


# Base dimension indices
_M, _L, _T, _I, _Θ, _N, _J = range(7)
_BASE_DIM_NAMES = ["M", "L", "T", "I", "Θ", "N", "J"]

# Simple unit → dimension mapping (7-vector of exponents)
_SIMPLE_UNITS: dict[str, tuple[int, ...]] = {
    # SI base
    "kg":    (1, 0, 0, 0, 0, 0, 0),
    "m":     (0, 1, 0, 0, 0, 0, 0),
    "s":     (0, 0, 1, 0, 0, 0, 0),
    "A":     (0, 0, 0, 1, 0, 0, 0),
    "K":     (0, 0, 0, 0, 1, 0, 0),
    "mol":   (0, 0, 0, 0, 0, 1, 0),
    "cd":    (0, 0, 0, 0, 0, 0, 1),
    # SI derived
    "N":     (1, 1, -2, 0, 0, 0, 0),  # kg·m/s^2
    "J":     (1, 2, -2, 0, 0, 0, 0),  # kg·m^2/s^2
    "W":     (1, 2, -3, 0, 0, 0, 0),  # kg·m^2/s^3
    "Pa":    (1, -1, -2, 0, 0, 0, 0), # kg/(m·s^2)
    "Hz":    (0, 0, -1, 0, 0, 0, 0),
    # Dimensionless
    "rad":   (0, 0, 0, 0, 0, 0, 0),
    "sr":    (0, 0, 0, 0, 0, 0, 0),
}


def _parse_unit(unit_str: str) -> tuple[int, ...]:
    """Parse a simple unit string into a dimension vector.

    Supports ``*``, ``/``, and ``^N`` syntax, e.g. ``m/s``, ``kg*m/s^2``.
    """
    # Start with dimensionless
    dims = [0] * 7

    if not unit_str or unit_str.strip() == "":
        return tuple(dims)

    # Handle numerical factor at start: just skip it
    s = unit_str.strip()

    # Split by '*' and '/' to handle compound units
    # Very simple parser — splits on * and /, applies ^ exponents
    def _parse_atom(atom: str) -> tuple[int, ...]:
        atom = atom.strip()
        exp = 1
        if "^" in atom:
            atom, exp_str = atom.rsplit("^", 1)
            exp = int(exp_str)
        atom = atom.strip()
        base_dims = _SIMPLE_UNITS.get(atom)
        if base_dims is None:
            return tuple(dims)  # unknown → dimensionless
        return tuple(d * exp for d in base_dims)

    # Split on / first, then on *
    # e.g., "kg*m/s^2" → numerator: ["kg", "m"], denominator: ["s^2"]
    parts = s.split("/")
    numerators = parts[0].split("*") if parts[0] else []
    denominators = parts[1].split("*") if len(parts) > 1 else []

    for num in numerators:
        atom_dims = _parse_atom(num)
        dims = [d + a for d, a in zip(dims, atom_dims)]
    for den in denominators:
        atom_dims = _parse_atom(den)
        dims = [d - a for d, a in zip(dims, atom_dims)]

    return tuple(dims)


def _format_dim(dims: tuple[int, ...]) -> str:
    parts = []
    for i, d in enumerate(dims):
        if d != 0:
            parts.append(f"{_BASE_DIM_NAMES[i]}^{d}")
    return "*".join(parts) if parts else "1"


# For users who pass unit strings via the API — we track the raw strings too.
# The dimension vector is an immutable 7-tuple (M, L, T, I, Θ, N, J).


class DimTracker:
    """Tracks a (dimension_vector, wildcard, violates) triple through expression evaluation.

    - ``dims``: 7-tuple of exponents (M, L, T, I, Θ, N, J)
    - ``wildcard``: if True, the expression contains a free constant whose
      dimensions are not yet determined
    - ``violates``: if True, a dimensional violation has already occurred
    """

    __slots__ = ("dims", "wildcard", "violates")

    def __init__(
        self, dims: tuple[int, ...], wildcard: bool = False, violates: bool = False,
    ) -> None:
        self.dims = dims
        self.wildcard = wildcard
        self.violates = violates

    @staticmethod
    def zero(wildcard: bool = False) -> DimTracker:
        return DimTracker((0,) * 7, wildcard=wildcard)

    def copy(self) -> DimTracker:
        return DimTracker(self.dims, self.wildcard, self.violates)

    def __repr__(self) -> str:
        return f"DimTracker({_format_dim(self.dims)}, wildcard={self.wildcard}, violates={self.violates})"


def _dim_mul(a: DimTracker, b: DimTracker) -> DimTracker:
    if a.violates:
        return a
    if b.violates:
        return b
    new_dims = tuple(d1 + d2 for d1, d2 in zip(a.dims, b.dims))
    return DimTracker(new_dims, a.wildcard or b.wildcard, False)


def _dim_div(a: DimTracker, b: DimTracker) -> DimTracker:
    if a.violates:
        return a
    if b.violates:
        return b
    new_dims = tuple(d1 - d2 for d1, d2 in zip(a.dims, b.dims))
    return DimTracker(new_dims, a.wildcard or b.wildcard, False)


def _dim_add_sub(a: DimTracker, b: DimTracker) -> DimTracker:
    if a.violates:
        return a
    if b.violates:
        return b
    if a.dims == b.dims:
        return DimTracker(a.dims, a.wildcard and b.wildcard, False)
    # Wildcard blending
    if a.wildcard and b.wildcard:
        return DimTracker(a.dims, True, False)
    if a.wildcard:
        return DimTracker(b.dims, False, False)
    if b.wildcard:
        return DimTracker(a.dims, False, False)
    # Mismatch — violation
    return DimTracker(a.dims, False, True)


def _dim_pow(base: DimTracker, exp: DimTracker) -> DimTracker:
    if base.violates or exp.violates:
        return DimTracker(base.dims, False, True)
    # Both base and exponent must be dimensionless for valid power
    # exponent must be dimensionless
    if exp.dims != (0,) * 7 and not exp.wildcard:
        return DimTracker(base.dims, False, True)
    if base.dims == (0,) * 7 or base.wildcard:
        # Base is dimensionless — result is dimensionless
        return DimTracker((0,) * 7, False, False)
    # Power of a dimensionful quantity: only valid if exponent is a constant
    # For simplicity, reject dimensionful ^ anything
    return DimTracker(base.dims, False, True)


def _dim_unary(a: DimTracker) -> DimTracker:
    """Unary transcendental ops (sin, cos) require dimensionless input."""
    if a.dims != (0,) * 7 and not a.wildcard:
        return DimTracker((0,) * 7, False, True)  # violation
    return DimTracker((0,) * 7, False, False)


def _dim_cmp(_a: DimTracker, _b: DimTracker) -> DimTracker:
    """Comparison operators return dimensionless regardless of inputs."""
    return DimTracker((0,) * 7, False, False)


def _dim_bool_unary(_a: DimTracker) -> DimTracker:
    return DimTracker((0,) * 7, False, False)


def _dim_abs(a: DimTracker) -> DimTracker:
    """abs preserves dimensions."""
    return a.copy()


def _dim_safe_log(a: DimTracker) -> DimTracker:
    """log requires dimensionless input, returns dimensionless."""
    if a.dims != (0,) * 7 and not a.wildcard:
        return DimTracker((0,) * 7, False, True)  # violation
    return DimTracker((0,) * 7, False, False)


def _dim_min_max(a: DimTracker, b: DimTracker) -> DimTracker:
    return _dim_add_sub(a, b)


# Map op_id to dim-check function
_DIM_OP_MAP: dict[str, Any] = {
    "sr.arith.add_v1": _dim_add_sub,
    "sr.arith.sub_v1": _dim_add_sub,
    "sr.arith.mul_v1": _dim_mul,
    "sr.math.protected_div_v1": _dim_div,
    "sr.math.pow_v1": _dim_pow,
    "sr.math.sin_v1": _dim_unary,
    "sr.math.cos_v1": _dim_unary,
    "sr.math.tan_v1": _dim_unary,
    "sr.math.exp_v1": _dim_unary,
    "sr.math.factorial_v1": _dim_unary,
    "sr.math.sqrt_v1": _dim_unary,
    "sr.math.logistic_v1": _dim_unary,
    "sr.math.step_v1": _dim_bool_unary,
    "sr.math.sign_v1": _dim_bool_unary,
    "sr.math.gauss_v1": _dim_unary,
    "sr.math.tanh_v1": _dim_unary,
    "sr.math.erf_v1": _dim_unary,
    "sr.math.erfc_v1": _dim_unary,
    "sr.math.abs_v1": _dim_abs,
    "sr.math.safe_log_v1": _dim_safe_log,
    "sr.arith.less_v1": _dim_cmp,  # comparison returns dimensionless
    "sr.bool.equal_v1": _dim_cmp,
    "sr.bool.less_equal_v1": _dim_cmp,
    "sr.bool.greater_v1": _dim_cmp,
    "sr.bool.greater_equal_v1": _dim_cmp,
    "sr.bool.and_v1": _dim_cmp,
    "sr.bool.or_v1": _dim_cmp,
    "sr.bool.xor_v1": _dim_cmp,
    "sr.bool.not_v1": _dim_bool_unary,
    "sr.math.min_v1": _dim_min_max,
    "sr.math.max_v1": _dim_min_max,
    "sr.math.mod_v1": _dim_min_max,
    "sr.math.floor_v1": _dim_unary,
    "sr.math.ceil_v1": _dim_unary,
    "sr.math.round_v1": _dim_unary,
    "sr.math.asin_v1": _dim_unary,
    "sr.math.acos_v1": _dim_unary,
    "sr.math.atan_v1": _dim_unary,
    "sr.math.atan2_v1": _dim_cmp,
    "sr.arith.neg_v1": _dim_abs,
    "sr.ts.delay_v1": _dim_abs,
    "sr.ts.sma_v1": _dim_abs,
    "sr.ts.wma_v1": _dim_abs,
    "sr.ts.mma_v1": _dim_abs,
    "sr.ts.median_v1": _dim_abs,
}


def check_dimensions(
    expr: Node,
    x_units: list[str] | None,
    y_units: str | None,
    allow_wildcards: bool,
) -> bool:
    """Check whether *expr* violates dimensional constraints.

    Returns ``True`` if the expression **violates** dimensional constraints
    (i.e., it should be penalized).  Returns ``False`` if it is dimensionally
    consistent (or if units are not provided).

    Parameters
    ----------
    expr:
        The expression tree to check.
    x_units:
        Unit strings for each input variable, e.g. ``["m", "s"]``.
    y_units:
        Unit string for the target, e.g. ``"m/s"``.
    allow_wildcards:
        If ``True``, free constants can have arbitrary dimensions
        (``dimensionless_constants_only=False``).
    """
    if x_units is None or y_units is None:
        return False  # no constraint

    parsed_x = [_parse_unit(u) for u in x_units]
    parsed_y = _parse_unit(y_units)

    def _walk(node: Node) -> DimTracker:
        if isinstance(node, ConstNode):
            return DimTracker.zero(wildcard=allow_wildcards)
        if isinstance(node, VarNode):
            if node.index < len(parsed_x):
                return DimTracker(parsed_x[node.index], False, False)
            return DimTracker.zero(False, True)
        if isinstance(node, OpNode):
            fn = _DIM_OP_MAP.get(node.op_id)
            if fn is None:
                return DimTracker.zero(False, True)
            child_trackers = [_walk(c) for c in node.children]
            if not child_trackers:
                return DimTracker.zero(False, False)
            if len(child_trackers) == 1:
                return fn(child_trackers[0])
            return fn(*child_trackers[:2])
        return DimTracker.zero(False, True)

    result = _walk(expr)
    if result.violates:
        return True
    # Check output dimensions match y_units
    if not result.wildcard and result.dims != parsed_y:
        return True
    return False
