from __future__ import annotations

import copy

import pytest

from python_backend.expr import ConstNode, VarNode, OpNode
from python_backend.hof import HallOfFame


def _make_expr() -> OpNode:
    return OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(1.0)])


def _hash(e: object) -> str:
    return str(id(e))


class TestHallOfFameBasic:
    def test_empty_hof(self) -> None:
        hof = HallOfFame(max_size=5)
        assert hof.best() is None
        assert hof.entries() == []
        assert hof.calculate_pareto_frontier() == []

    def test_insert_and_best(self) -> None:
        hof = HallOfFame(max_size=5)
        e = _make_expr()
        result = hof.consider(e, loss=0.5, complexity=3, h=_hash(e))
        assert result == "inserted"
        best = hof.best()
        assert best is not None
        assert best["loss"] == 0.5
        assert best["complexity"] == 3

    def test_replace_better(self) -> None:
        hof = HallOfFame(max_size=5)
        e1 = _make_expr()
        hof.consider(e1, loss=0.5, complexity=3, h=_hash(e1))
        e2 = OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)])
        result = hof.consider(e2, loss=0.3, complexity=3, h=_hash(e2))
        assert result == "replaced"
        assert hof.best()["loss"] == 0.3

    def test_insert_worse_at_same_complexity(self) -> None:
        hof = HallOfFame(max_size=5)
        e1 = _make_expr()
        hof.consider(e1, loss=0.5, complexity=3, h=_hash(e1))
        e2 = OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)])
        result = hof.consider(e2, loss=0.7, complexity=3, h=_hash(e2))
        assert result == "unchanged"

    def test_max_size_eviction(self) -> None:
        hof = HallOfFame(max_size=3)
        for i in range(4):
            e = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(float(i))])
            hof.consider(e, loss=0.1 * i, complexity=i, h=_hash(e))
        assert len(hof._by_complexity) == 3

    def test_parsimony(self) -> None:
        hof = HallOfFame(max_size=5)
        hof.set_parsimony(0.1)
        e1 = _make_expr()
        hof.consider(e1, loss=0.5, complexity=10, h=_hash(e1))
        e2 = OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)])
        hof.consider(e2, loss=0.4, complexity=5, h=_hash(e2))
        best = hof.best()
        assert best is not None
        assert best["complexity"] == 5  # lower parsimonious loss


class TestParetoFrontier:
    def test_pareto_basic(self) -> None:
        hof = HallOfFame(max_size=10)
        for i, loss in enumerate([0.9, 0.5, 0.7, 0.3, 0.4]):
            e = OpNode("sr.arith.add_v1", [VarNode(0), ConstNode(float(i))])
            hof.consider(e, loss=loss, complexity=i + 1, h=_hash(e))
        frontier = hof.calculate_pareto_frontier()
        # Pareto: complexity 1 loss 0.9, 2 loss 0.5, 4 loss 0.3 (loss < all simpler)
        assert len(frontier) == 3
        assert frontier[0]["complexity"] == 1
        assert frontier[1]["complexity"] == 2
        assert frontier[2]["complexity"] == 4

    def test_pareto_all_dominated(self) -> None:
        hof = HallOfFame(max_size=10)
        hof.consider(_make_expr(), loss=0.1, complexity=1, h="a")
        hof.consider(_make_expr(), loss=0.2, complexity=2, h="b")
        hof.consider(_make_expr(), loss=0.3, complexity=3, h="c")
        frontier = hof.calculate_pareto_frontier()
        assert len(frontier) == 1
        assert frontier[0]["complexity"] == 1

    def test_pareto_tie_loss(self) -> None:
        hof = HallOfFame(max_size=10)
        hof.consider(_make_expr(), loss=0.5, complexity=1, h="a")
        hof.consider(_make_expr(), loss=0.5, complexity=2, h="b")
        frontier = hof.calculate_pareto_frontier()
        assert len(frontier) == 1


class TestScoreComputation:
    def test_direct_score_linear(self) -> None:
        hof = HallOfFame(max_size=10)
        hof.consider(_make_expr(), loss=0.8, complexity=1, h="a")
        hof.consider(_make_expr(), loss=0.5, complexity=2, h="b")
        hof.consider(_make_expr(), loss=0.3, complexity=3, h="c")
        scored = hof.compute_scores(loss_scale="linear")
        assert len(scored) == 3
        assert scored[0]["score"] == pytest.approx(0.0, abs=1e-10)
        assert scored[1]["score"] == pytest.approx(0.375, abs=1e-4)
        assert scored[2]["score"] == pytest.approx(0.4, abs=1e-4)

    def test_zero_centered_score_log(self) -> None:
        hof = HallOfFame(max_size=10)
        hof.consider(_make_expr(), loss=0.8, complexity=1, h="a")
        hof.consider(_make_expr(), loss=0.5, complexity=2, h="b")
        scored = hof.compute_scores(loss_scale="log")
        assert len(scored) == 2
        assert scored[1]["score"] == pytest.approx(scored[0]["score"] + 0.375, abs=1e-4)


class TestStringFormatting:
    def test_string_pareto_curve(self) -> None:
        hof = HallOfFame(max_size=10)
        hof.consider(_make_expr(), loss=0.5, complexity=2, h="a")
        hof.consider(_make_expr(), loss=0.3, complexity=3, h="b")
        s = hof.string_dominating_pareto_curve()
        assert "Dominating Pareto Curve" in s
        assert "Complexity" in s
        assert "Loss" in s

    def test_empty_string_curve(self) -> None:
        hof = HallOfFame(max_size=5)
        assert hof.string_dominating_pareto_curve() == ""


class TestDunders:
    def test_repr_empty(self) -> None:
        hof = HallOfFame(max_size=5)
        assert "empty" in repr(hof)

    def test_repr_with_entries(self) -> None:
        hof = HallOfFame(max_size=5)
        hof.consider(_make_expr(), loss=0.5, complexity=3, h="a")
        assert "best_loss=0.5" in repr(hof)

    def test_str(self) -> None:
        hof = HallOfFame(max_size=5)
        hof.consider(_make_expr(), loss=0.5, complexity=3, h="a")
        s = str(hof)
        assert "Dominating Pareto Curve" in s

    def test_copy(self) -> None:
        hof = HallOfFame(max_size=5)
        hof.consider(_make_expr(), loss=0.5, complexity=3, h="a")
        copied = copy.copy(hof)
        assert copied.best()["loss"] == 0.5
        assert copied.max_size == 5

    def test_deepcopy(self) -> None:
        hof = HallOfFame(max_size=5)
        e = _make_expr()
        hof.consider(e, loss=0.5, complexity=3, h=_hash(e))
        copied = copy.deepcopy(hof)
        assert copied.best()["loss"] == 0.5
        assert copied.max_size == 5


class TestNegativeLoss:
    def test_negative_loss_raises(self) -> None:
        hof = HallOfFame(max_size=5)
        with pytest.raises(ValueError, match="negative loss"):
            hof.consider(_make_expr(), loss=-0.1, complexity=3, h="a")

    def test_negative_loss_log_score_raises(self) -> None:
        hof = HallOfFame(max_size=5)
        hof._by_complexity[1] = dict(
            canonical_expression="test", loss=-0.1, complexity=1, hash="a",
        )
        with pytest.raises(ValueError, match="negative loss"):
            hof.compute_scores(loss_scale="log")


class TestConstraintCheck:
    def test_constraint_check_rejects(self) -> None:
        hof = HallOfFame(max_size=5)
        def _reject(_: object) -> bool:
            return False
        result = hof.consider(
            _make_expr(), loss=0.5, complexity=3, h="a",
            constraints_check=_reject,
        )
        assert result == "unchanged"
        assert hof.best() is None

    def test_constraint_check_accepts(self) -> None:
        hof = HallOfFame(max_size=5)
        def _accept(_: object) -> bool:
            return True
        result = hof.consider(
            _make_expr(), loss=0.5, complexity=3, h="a",
            constraints_check=_accept,
        )
        assert result == "inserted"
        assert hof.best() is not None
