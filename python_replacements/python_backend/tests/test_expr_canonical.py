from python_backend.expr import VarNode, ConstNode, OpNode


def test_canonical_is_stable():
    n1 = ConstNode(3.14)
    n2 = ConstNode(3.14)
    assert n1.canonical() == n2.canonical()
    assert n1.structural_hash() == n2.structural_hash()


def test_var_canonical():
    v = VarNode(0)
    assert v.canonical() == "var[0]"


def test_const_canonical():
    c = ConstNode(1.5)
    expected = f"const[float64:{1.5.hex()}]"
    assert c.canonical() == expected


def test_opnode_canonical():
    x = VarNode(0)
    y = VarNode(1)
    add = OpNode("sr.arith.add_v1", [x, y])
    assert add.canonical() == "sr.arith.add_v1(var[0],var[1])"


def test_commutative_sorts_children():
    x = VarNode(1)
    y = VarNode(0)
    add = OpNode("sr.arith.add_v1", [x, y])
    assert add.canonical() == "sr.arith.add_v1(var[0],var[1])"


def test_non_commutative_preserves_order():
    x = VarNode(1)
    y = VarNode(0)
    sub = OpNode("sr.arith.sub_v1", [x, y])
    assert sub.canonical() == "sr.arith.sub_v1(var[1],var[0])"


def test_structural_hash_changes_on_structure():
    a = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(1.0)])
    b = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(2.0)])
    assert a.structural_hash() != b.structural_hash()


def test_structural_hash_same_for_equal_trees():
    a = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(1.0)])
    b = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(1.0)])
    assert a.structural_hash() == b.structural_hash()


def test_canonical_uses_full_op_id():
    """Canonical form must use full registry IDs, not short tokens."""
    x = VarNode(0)
    add = OpNode("sr.arith.add_v1", [x, ConstNode(1.0)])
    c = add.canonical()
    assert c.startswith("sr.arith.add_v1("), f"expected full op_id, got {c!r}"
    assert "var[" in c
    assert "const[float64:" in c


def test_deeply_nested_canonical():
    expr = OpNode(
        "sr.arith.add_v1",
        [
            OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
            VarNode(1),
        ],
    )
    c = expr.canonical()
    assert c.startswith("sr.arith.add_v1(")
    assert "sr.arith.mul_v1(" in c
    assert "var[0]" in c
    assert "var[1]" in c
    assert "const[float64:" in c


# ── parse_canonical ──────────────────────────────────────────────────────────

from python_backend.expr import parse_canonical, node_to_sympy


def test_parse_canonical_var():
    node = parse_canonical("var[0]")
    assert isinstance(node, VarNode)
    assert node.index == 0


def test_parse_canonical_const():
    node = parse_canonical("const[float64:0x1.921fb54442d18p+1]")
    assert isinstance(node, ConstNode)
    assert abs(node.value - 3.14159) < 1e-5


def test_parse_canonical_binary_op():
    node = parse_canonical("sr.arith.add_v1(var[0],var[1])")
    assert isinstance(node, OpNode)
    assert node.op_id == "sr.arith.add_v1"
    assert isinstance(node.children[0], VarNode)
    assert isinstance(node.children[1], VarNode)
    assert node.children[0].index == 0
    assert node.children[1].index == 1


def test_parse_canonical_unary_op():
    node = parse_canonical("sr.math.sin_v1(var[2])")
    assert isinstance(node, OpNode)
    assert node.op_id == "sr.math.sin_v1"
    assert node.children[0].index == 2


def test_parse_canonical_nested():
    node = parse_canonical(
        "sr.arith.add_v1(sr.arith.mul_v1(var[0],const[float64:0x1.0000000000000p+1]),var[1])"
    )
    assert isinstance(node, OpNode)
    assert node.op_id == "sr.arith.add_v1"
    inner = node.children[0]
    assert isinstance(inner, OpNode)
    assert inner.op_id == "sr.arith.mul_v1"
    assert isinstance(inner.children[0], VarNode)
    assert isinstance(inner.children[1], ConstNode)
    assert inner.children[1].value == 2.0


def test_parse_canonical_roundtrip():
    orig = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
        VarNode(1),
    ])
    c = orig.canonical()
    parsed = parse_canonical(c)
    assert parsed.canonical() == c


# ── node_to_sympy ────────────────────────────────────────────────────────────

import sympy


def test_node_to_sympy_const():
    node = ConstNode(3.14)
    sp = node_to_sympy(node)
    assert sp == sympy.Number(3.14)


def test_node_to_sympy_var():
    node = VarNode(0)
    sp = node_to_sympy(node, variable_names=["x", "y"])
    assert sp == sympy.Symbol("x")


def test_node_to_sympy_add():
    node = OpNode("sr.arith.add_v1", [VarNode(0), VarNode(1)])
    sp = node_to_sympy(node, variable_names=["x0", "x1"])
    assert sp == sympy.Symbol("x0") + sympy.Symbol("x1")


def test_node_to_sympy_sin():
    node = OpNode("sr.math.sin_v1", [VarNode(0)])
    sp = node_to_sympy(node, variable_names=["x"])
    assert sp == sympy.sin(sympy.Symbol("x"))


def test_node_to_sympy_exp():
    node = OpNode("sr.math.exp_v1", [VarNode(0)])
    sp = node_to_sympy(node, variable_names=["x"])
    assert sp == sympy.exp(sympy.Symbol("x"))


def test_node_to_sympy_less():
    node = OpNode("sr.arith.less_v1", [VarNode(0), ConstNode(0.5)])
    sp = node_to_sympy(node, variable_names=["x"])
    assert sp == sympy.Heaviside(
        sympy.Number(0.5) - sympy.Symbol("x")
    )


def test_node_to_sympy_parse_roundtrip():
    orig = OpNode("sr.arith.add_v1", [
        OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
        OpNode("sr.math.sin_v1", [VarNode(1)]),
    ])
    c = orig.canonical()
    parsed = parse_canonical(c)
    sp = node_to_sympy(parsed, variable_names=["x0", "x1"])
    assert sp == sympy.Symbol("x0") * 2.0 + sympy.sin(sympy.Symbol("x1"))
