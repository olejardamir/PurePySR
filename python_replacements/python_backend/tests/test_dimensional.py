from __future__ import annotations

from python_backend.dimensional import _parse_unit, check_dimensions
from python_backend.expr import ConstNode, OpNode, VarNode


def test_parse_unit_dimensionless():
    dims = _parse_unit("")
    assert dims == (0, 0, 0, 0, 0, 0, 0)


def test_parse_unit_meter():
    dims = _parse_unit("m")
    assert dims[1] == 1  # L
    assert all(dims[i] == 0 for i in range(7) if i != 1)


def test_parse_unit_compound():
    dims = _parse_unit("m/s")
    assert dims[1] == 1   # L
    assert dims[2] == -1  # T^-1


def test_parse_unit_newton():
    dims = _parse_unit("N")
    assert dims[0] == 1   # M
    assert dims[1] == 1   # L
    assert dims[2] == -2  # T^-2


def test_check_dimensions_no_units():
    expr = OpNode("sr.arith.add_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, None, None, True) is False
    assert check_dimensions(expr, ["m"], None, True) is False
    assert check_dimensions(expr, None, "m", True) is False


def test_check_dimensions_add_same():
    # x0 (m) + x1 (m) → m  ✓ (if y_units is m)
    expr = OpNode("sr.arith.add_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, ["m", "m"], "m", True) is False


def test_check_dimensions_add_mismatch():
    # x0 (m) + x1 (s) → violation
    expr = OpNode("sr.arith.add_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, ["m", "s"], "m", True) is True


def test_check_dimensions_mul():
    # x0 (m) * x1 (s) → m·s  ✓ (if y_units is m*s is not directly handled)
    expr = OpNode("sr.arith.mul_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, ["m", "s"], "m*s", True) is False
    # Wrong output
    assert check_dimensions(expr, ["m", "s"], "m", True) is True


def test_check_dimensions_sin():
    # sin(x0) requires dimensionless input
    expr = OpNode("sr.math.sin_v1", [VarNode(0)])
    # sin(m) → violation (needs dimensionless)
    assert check_dimensions(expr, ["m"], "1", True) is True
    # sin(1) → ok
    expr2 = OpNode("sr.math.sin_v1", [ConstNode(1.0)])
    assert check_dimensions(expr2, ["m"], "1", True) is False


def test_check_dimensions_pow_dimensionless():
    # dimensionless^dimensionless → ok
    expr = OpNode("sr.math.pow_v1", [ConstNode(2.0), ConstNode(3.0)])
    assert check_dimensions(expr, ["m"], "1", True) is False


def test_check_dimensions_div():
    # m / s → m/s
    expr = OpNode("sr.math.protected_div_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, ["m", "s"], "m/s", True) is False
    assert check_dimensions(expr, ["m", "s"], "m", True) is True


def test_check_dimensions_less():
    # comparison is dimensionless
    expr = OpNode("sr.arith.less_v1", [VarNode(0), VarNode(1)])
    assert check_dimensions(expr, ["m", "m"], "1", True) is False
