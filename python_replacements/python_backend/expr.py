"""
Canonical expression serialization.

The canonical form uses full operator IDs from `sr-operator-registry.yaml`
(e.g. ``sr.arith.add_v1``, ``sr.math.sin_v1``) rather than short tokens.
This ensures unambiguity across namespaces and versions.

Format examples::

  Variable:  var[<index>]
  Constant:  const[float64:<hex>]
  Unary op:  sr.math.sin_v1(var[0])
  Binary op: sr.arith.add_v1(var[0],var[1])
"""

from __future__ import annotations

import hashlib


_COMMUTATIVE_OPS: frozenset[str] = frozenset([
    "sr.arith.add_v1",
    "sr.arith.mul_v1",
])


class Node:
    def canonical(self) -> str:
        raise NotImplementedError

    def structural_hash(self) -> str:
        raw = self.canonical().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class VarNode(Node):
    __slots__ = ("index", "_eval_cache")

    def __init__(self, index: int) -> None:
        self.index = index
        self._eval_cache = None

    def canonical(self) -> str:
        return f"var[{self.index}]"

    def __repr__(self) -> str:
        return f"VarNode({self.index})"


class ConstNode(Node):
    __slots__ = ("value", "_eval_cache")

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self._eval_cache = None

    def canonical(self) -> str:
        return f"const[float64:{self.value.hex()}]"

    def __repr__(self) -> str:
        return f"ConstNode({self.value})"


class OpNode(Node):
    __slots__ = ("op_id", "children", "_eval_cache")

    def __init__(self, op_id: str, children: list[Node]) -> None:
        self.op_id = op_id
        self.children = list(children)
        self._eval_cache = None

    def canonical(self) -> str:
        parts = [c.canonical() for c in self.children]
        if self.op_id in _COMMUTATIVE_OPS:
            parts = sorted(parts)
        return f"{self.op_id}({','.join(parts)})"

    def __repr__(self) -> str:
        args = ", ".join(repr(c) for c in self.children)
        return f"OpNode({self.op_id!r}, [{args}])"


def count_operator_occurrences(node: Node) -> dict[str, int]:
    counts: dict[str, int] = {}
    _walk_count(node, counts)
    return counts


def _walk_count(node: Node, counts: dict[str, int]) -> None:
    if isinstance(node, OpNode):
        counts[node.op_id] = counts.get(node.op_id, 0) + 1
        for c in node.children:
            _walk_count(c, counts)


def check_nested_constraints(
    node: Node,
    nested_constraints: dict[str, dict[str, int]],
) -> tuple[bool, str]:
    if isinstance(node, OpNode):
        outer = node.op_id
        limits = nested_constraints.get(outer, {})
        for child in node.children:
            if isinstance(child, OpNode):
                inner = child.op_id
                limit = limits.get(inner)
                if limit is not None:
                    depth = _nested_depth(child, inner)
                    if depth > limit:
                        return (False, "SR-INV-NESTING-001")
        for child in node.children:
            ok, reason = check_nested_constraints(child, nested_constraints)
            if not ok:
                return (False, reason)
    return (True, "")


def _nested_depth(node: Node, target_op: str) -> int:
    if isinstance(node, OpNode) and node.op_id == target_op:
        best = 0
        for c in node.children:
            sub = _nested_depth(c, target_op)
            if sub > best:
                best = sub
        return 1 + best
    return 0


# ── Expression parser (string → Node) ──────────────────────────────────────

import re

from python_backend.ops import TOKEN_TO_OP_ID
from python_backend.ops import OP_ID_TO_ARITY

_TOKEN_PATTERN = re.compile(r"""
    (?P<NUMBER>\d+\.?\d*(?:[eE][+-]?\d+)?)
    |(?P<IDENT>[a-zA-Z_][a-zA-Z0-9_]*)
    |(?P<OP>[+\-*/^])
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<COMMA>,)
    |(?P<SKIP>\s+)
    |(?P<ERR>.)
""", re.VERBOSE)


def _tokenize(s: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_PATTERN.finditer(s):
        kind = m.lastgroup
        val = m.group()
        if kind == "SKIP":
            continue
        if kind == "ERR":
            raise ValueError(f"unexpected character {val!r} at position {m.start()}")
        tokens.append((kind, val))
    return tokens


class _Parser:
    def __init__(
        self,
        tokens: list[tuple[str, str]],
        variable_names: list[str],
    ) -> None:
        self._tokens = tokens
        self._pos = 0
        self._variable_names = variable_names

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str, val: str | None = None) -> tuple[str, str]:
        tok = self._peek()
        if tok is None or tok[0] != kind or (val is not None and tok[1] != val):
            expected = repr(val) if val else kind
            got = repr(tok[1]) if tok else "end of input"
            raise ValueError(f"expected {expected}, got {got}")
        return self._consume()

    def parse(self) -> Node:
        result = self._expression()
        return result

    def _expression(self) -> Node:
        left = self._term()
        while True:
            op = self._peek()
            if op is None or op[0] != "OP" or op[1] not in ("+", "-"):
                break
            self._consume()
            right = self._term()
            op_id = TOKEN_TO_OP_ID[op[1]]
            left = OpNode(op_id, [left, right])
        return left

    def _term(self) -> Node:
        left = self._factor()
        while True:
            op = self._peek()
            if op is None or op[0] != "OP" or op[1] not in ("*", "/"):
                break
            self._consume()
            right = self._factor()
            op_id = TOKEN_TO_OP_ID[op[1]]
            left = OpNode(op_id, [left, right])
        return left

    def _factor(self) -> Node:
        left = self._primary()
        op = self._peek()
        if op is not None and op[0] == "OP" and op[1] == "^":
            self._consume()
            right = self._primary()
            op_id = TOKEN_TO_OP_ID["^"]
            left = OpNode(op_id, [left, right])
        return left

    def _primary(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of input")

        # Unary minus: "-" primary
        if tok[0] == "OP" and tok[1] == "-":
            self._consume()
            operand = self._primary()
            # Represent as (0 - operand) = sub(zero, operand)
            zero = ConstNode(0.0)
            return OpNode(TOKEN_TO_OP_ID["-"], [zero, operand])

        if tok[0] == "NUMBER":
            self._consume()
            return ConstNode(float(tok[1]))

        if tok[0] == "IDENT":
            self._consume()
            name = tok[1]
            # Function call or variable
            if self._peek() is not None and self._peek()[0] == "LPAREN":
                # Function call: f(args)
                self._consume()  # "("
                args = []
                if self._peek() is not None and self._peek()[0] != "RPAREN":
                    while True:
                        args.append(self._expression())
                        if self._peek() is None or self._peek()[0] != "COMMA":
                            break
                        self._consume()
                self._expect("RPAREN")
                op_id = TOKEN_TO_OP_ID.get(name)
                if op_id is None:
                    raise ValueError(f"unknown function {name!r}")
                arity = OP_ID_TO_ARITY.get(op_id)
                if arity is None:
                    raise ValueError(f"unknown function {name!r}")
                if len(args) != arity:
                    raise ValueError(
                        f"function {name!r} expects {arity} args, got {len(args)}"
                    )
                return OpNode(op_id, args)
            else:
                # Variable
                try:
                    idx = self._variable_names.index(name)
                except ValueError:
                    try:
                        idx = int(name[1:]) if name.startswith("x") else self._variable_names.index(name)
                    except (ValueError, IndexError):
                        raise ValueError(f"unknown variable {name!r}")
                return VarNode(idx)

        if tok[0] == "LPAREN":
            self._consume()
            result = self._expression()
            self._expect("RPAREN")
            return result

        raise ValueError(f"unexpected token {tok[1]!r}")


def parse_canonical(s: str) -> Node:
    """Parse a canonical expression string (e.g. ``sr.arith.add_v1(var[0],var[1])``)
    into a :class:`Node` tree.

    The canonical form is::

        var[<index>]
        const[float64:<hex>]
        <op_id>(<canonical>,<canonical>,...)
    """
    s = s.strip()

    # var[N]
    if s.startswith("var[") and s.endswith("]"):
        idx = int(s[len("var["):-1])
        return VarNode(idx)

    # const[float64:hex]
    if s.startswith("const[float64:") and s.endswith("]"):
        hex_val = s[len("const[float64:"):-1]
        return ConstNode(float.fromhex(hex_val))

    # op_id(arg1,arg2,...)
    paren_idx = s.find("(")
    if paren_idx != -1 and s.endswith(")"):
        op_id = s[:paren_idx]
        args_raw = s[paren_idx + 1:-1]
        args = _split_canonical_args(args_raw)
        children = [parse_canonical(a) for a in args]
        return OpNode(op_id, children)

    raise ValueError(f"cannot parse canonical expression: {s!r}")


def _split_canonical_args(raw: str) -> list[str]:
    """Split comma-separated arguments in a canonical expression."""
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


def node_to_sympy(
    node: Node,
    variable_names: list[str] | None = None,
) -> object:
    """Convert a Node tree to a SymPy expression.

    Requires ``sympy`` to be installed.  The *variable_names* list maps
    ``VarNode.index`` to sympy symbols (default: ``x0``, ``x1``, …).
    """
    import sympy as _sp

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(10)]
    _symbols = {i: _sp.Symbol(n) for i, n in enumerate(variable_names)}

    _OP_TO_SYMPY: dict[str, object] = {
        "sr.arith.add_v1": lambda a, b: a + b,
        "sr.arith.sub_v1": lambda a, b: a - b,
        "sr.arith.mul_v1": lambda a, b: a * b,
        "sr.math.protected_div_v1": lambda a, b: a / b,
        "sr.math.pow_v1": lambda a, b: a ** b,
        "sr.math.sin_v1": _sp.sin,
        "sr.math.cos_v1": _sp.cos,
        "sr.math.tan_v1": _sp.tan,
        "sr.math.abs_v1": _sp.Abs,
        "sr.math.safe_log_v1": lambda a: _sp.log(_sp.Abs(a)),
        "sr.math.exp_v1": _sp.exp,
        "sr.math.factorial_v1": lambda a: _sp.gamma(a + 1),
        "sr.math.sqrt_v1": _sp.sqrt,
        "sr.math.logistic_v1": lambda a: 1 / (1 + _sp.exp(-a)),
        "sr.math.step_v1": lambda a: _sp.Piecewise((1, a > 0), (0, True)),
        "sr.math.sign_v1": _sp.sign,
        "sr.math.gauss_v1": lambda a: _sp.exp(-(a ** 2)),
        "sr.math.tanh_v1": _sp.tanh,
        "sr.math.erf_v1": _sp.erf,
        "sr.math.erfc_v1": _sp.erfc,
        "sr.bool.equal_v1": lambda a, b: _sp.Piecewise((1, _sp.Eq(a, b)), (0, True)),
        "sr.arith.less_v1": lambda a, b: _sp.Heaviside(b - a),
        "sr.bool.less_equal_v1": lambda a, b: _sp.Piecewise((1, a <= b), (0, True)),
        "sr.bool.greater_v1": lambda a, b: _sp.Piecewise((1, a > b), (0, True)),
        "sr.bool.greater_equal_v1": lambda a, b: _sp.Piecewise((1, a >= b), (0, True)),
        "sr.bool.and_v1": lambda a, b: _sp.Piecewise((1, (a > 0) & (b > 0)), (0, True)),
        "sr.bool.or_v1": lambda a, b: _sp.Piecewise((1, (a > 0) | (b > 0)), (0, True)),
        "sr.bool.xor_v1": lambda a, b: _sp.Piecewise((1, ((a > 0) & (b <= 0)) | ((a <= 0) & (b > 0))), (0, True)),
        "sr.bool.not_v1": lambda a: _sp.Piecewise((0, a > 0), (1, True)),
        "sr.math.min_v1": _sp.Min,
        "sr.math.max_v1": _sp.Max,
        "sr.math.mod_v1": _sp.Mod,
        "sr.math.floor_v1": _sp.floor,
        "sr.math.ceil_v1": _sp.ceiling,
        "sr.math.round_v1": lambda a: _sp.floor(a + _sp.Rational(1, 2)),
        "sr.math.asin_v1": _sp.asin,
        "sr.math.acos_v1": _sp.acos,
        "sr.math.atan_v1": _sp.atan,
        "sr.math.atan2_v1": _sp.atan2,
        "sr.arith.neg_v1": lambda a: -a,
        "sr.ts.delay_v1": lambda a, n: _sp.Function("delay")(a, n),
        "sr.ts.sma_v1": lambda a, n: _sp.Function("sma")(a, n),
        "sr.ts.wma_v1": lambda a, n: _sp.Function("wma")(a, n),
        "sr.ts.mma_v1": lambda a, n: _sp.Function("mma")(a, n),
        "sr.ts.median_v1": lambda a, n: _sp.Function("median")(a, n),
    }

    def _convert(n: Node) -> _sp.Basic:
        if isinstance(n, ConstNode):
            return _sp.Number(n.value)
        if isinstance(n, VarNode):
            return _symbols.get(n.index, _sp.Symbol(f"x{n.index}"))
        if isinstance(n, OpNode):
            fn = _OP_TO_SYMPY.get(n.op_id)
            if fn is None:
                raise ValueError(f"no SymPy mapping for operator {n.op_id!r}")
            args = [_convert(c) for c in n.children]
            return fn(*args)
        raise TypeError(f"unknown node type: {type(n).__name__}")

    return _convert(node)


def parse_expression(
    s: str,
    variable_names: list[str] | None = None,
) -> Node:
    """Parse a string expression into a Node tree.

    Supports: +, -, *, /, ^, sin, cos, abs, safe_log, parentheses,
    numbers, and variables (``x0``, ``x1``, ... or named via *variable_names*).
    """
    if variable_names is None:
        variable_names = []
    tokens = _tokenize(s)
    parser = _Parser(tokens, variable_names)
    result = parser.parse()
    if parser._pos < len(tokens):
        remaining = " ".join(t[1] for t in tokens[parser._pos:])
        raise ValueError(f"unexpected tokens after expression: {remaining!r}")
    return result


# ── Expression simplification ────────────────────────────────────────────────

def simplify_expression(node: Node) -> Node:
    """Algebraic simplification (bottom-up).

    Handles: constant folding, identity elimination (x+0, x*1, …),
    and double-negation.
    """
    if isinstance(node, VarNode):
        return node
    if isinstance(node, ConstNode):
        return node
    if isinstance(node, OpNode):
        children = [simplify_expression(c) for c in node.children]
        node = OpNode(node.op_id, children)
        return _simplify_op(node)
    return node


def _simplify_op(node: OpNode) -> Node:
    op = node.op_id
    ch = node.children

    # ── Constant folding ────────────────────────────────────────────
    if len(ch) == 2 and all(isinstance(c, ConstNode) for c in ch):
        a = ch[0].value  # type: ignore[union-attr]
        b = ch[1].value  # type: ignore[union-attr]
        if op == "sr.arith.add_v1":
            return ConstNode(a + b)
        if op == "sr.arith.sub_v1":
            return ConstNode(a - b)
        if op == "sr.arith.mul_v1":
            return ConstNode(a * b)
        if op == "sr.math.protected_div_v1":
            return ConstNode(a / b if abs(b) > 1e-8 else 1.0)
        if op == "sr.math.pow_v1":
            return ConstNode(a ** b if a >= 0 else 1.0)

    # ── Unary constant folding ──────────────────────────────────────
    if len(ch) == 1 and isinstance(ch[0], ConstNode):
        v = ch[0].value  # type: ignore[union-attr]
        if op == "sr.math.sin_v1":
            import numpy as np
            return ConstNode(float(np.sin(v)))
        if op == "sr.math.cos_v1":
            import numpy as np
            return ConstNode(float(np.cos(v)))
        if op == "sr.math.abs_v1":
            return ConstNode(abs(v))
        if op == "sr.math.safe_log_v1":
            import numpy as np
            return ConstNode(float(np.log(abs(v) + 1e-8)))

    # ── Identity / zero / one elimination ───────────────────────────
    if op == "sr.arith.add_v1":
        if _is_zero(ch[0]):
            return ch[1]
        if _is_zero(ch[1]):
            return ch[0]
        # x + x → 2*x
        if _same_tree(ch[0], ch[1]):
            return OpNode("sr.arith.mul_v1", [ConstNode(2.0), ch[0]])

    if op == "sr.arith.sub_v1":
        if _is_zero(ch[1]):
            return ch[0]
        # x - x → 0
        if _same_tree(ch[0], ch[1]):
            return ConstNode(0.0)

    if op == "sr.arith.mul_v1":
        if _is_zero(ch[0]) or _is_zero(ch[1]):
            return ConstNode(0.0)
        if _is_one(ch[0]):
            return ch[1]
        if _is_one(ch[1]):
            return ch[0]
        # x * x → x^2
        if _same_tree(ch[0], ch[1]):
            return OpNode("sr.math.pow_v1", [ch[0], ConstNode(2.0)])

    if op == "sr.math.protected_div_v1":
        if _is_zero(ch[0]):
            return ConstNode(0.0)
        if _is_one(ch[1]):
            return ch[0]
        # x / x → 1
        if _same_tree(ch[0], ch[1]):
            return ConstNode(1.0)

    if op == "sr.math.pow_v1":
        if _is_zero(ch[1]):
            return ConstNode(1.0)
        if _is_one(ch[1]):
            return ch[0]
        if _is_one(ch[0]):
            return ConstNode(1.0)
        if _is_zero(ch[0]):
            return ConstNode(0.0)

    return node


def _is_zero(n: Node) -> bool:
    return isinstance(n, ConstNode) and n.value == 0.0


def _is_one(n: Node) -> bool:
    return isinstance(n, ConstNode) and n.value == 1.0


def _same_tree(a: Node, b: Node) -> bool:
    """Structural equality (not hash-based)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, VarNode):
        return isinstance(b, VarNode) and a.index == b.index
    if isinstance(a, ConstNode):
        return isinstance(b, ConstNode) and a.value == b.value
    if isinstance(a, OpNode):
        if not isinstance(b, OpNode) or a.op_id != b.op_id or len(a.children) != len(b.children):
            return False
        return all(_same_tree(ca, cb) for ca, cb in zip(a.children, b.children))
    return False
