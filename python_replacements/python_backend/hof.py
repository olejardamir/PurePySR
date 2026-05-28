from __future__ import annotations

import copy
from typing import Any, Callable

from python_backend.expr import Node


def _parsimonious_loss(loss: float, complexity: int, parsimony: float) -> float:
    return loss + complexity * parsimony


def _drop_worst(
    by_complexity: dict[int, dict[str, Any]],
    parsimony: float = 0.0,
) -> None:
    worst_key = max(
        by_complexity,
        key=lambda k: (
            _parsimonious_loss(by_complexity[k]["loss"], by_complexity[k]["complexity"], parsimony),
            by_complexity[k]["complexity"],
        ),
    )
    del by_complexity[worst_key]


class HallOfFame:
    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._by_complexity: dict[int, dict[str, Any]] = {}
        self._parsimony: float = 0.0

    def set_parsimony(self, parsimony: float) -> None:
        self._parsimony = parsimony

    def consider(
        self,
        expr: Node,
        loss: float,
        complexity: int,
        h: str,
        constraints_check: Callable[[Node], bool] | None = None,
    ) -> str:
        if loss < 0.0:
            raise ValueError(f"negative loss ({loss}) not allowed in HallOfFame")

        if constraints_check is not None and not constraints_check(expr):
            return "unchanged"

        existing = self._by_complexity.get(complexity)
        if existing is not None:
            existing_cost = _parsimonious_loss(existing["loss"], complexity, self._parsimony)
            candidate_cost = _parsimonious_loss(loss, complexity, self._parsimony)
            if candidate_cost < existing_cost:
                self._by_complexity[complexity] = dict(
                    canonical_expression=expr.canonical(),
                    loss=loss,
                    complexity=complexity,
                    hash=h,
                    expression=expr,
                )
                return "replaced"
            return "unchanged"

        self._by_complexity[complexity] = dict(
            canonical_expression=expr.canonical(),
            loss=loss,
            complexity=complexity,
            hash=h,
            expression=expr,
        )
        if len(self._by_complexity) > self.max_size:
            _drop_worst(self._by_complexity, self._parsimony)
        return "inserted"

    def best(self) -> dict[str, Any] | None:
        if not self._by_complexity:
            return None
        best_entry = min(
            self._by_complexity.values(),
            key=lambda e: (
                _parsimonious_loss(e["loss"], e["complexity"], self._parsimony),
                e["complexity"],
                e["hash"],
            ),
        )
        return {k: v for k, v in best_entry.items() if k != "expression"}

    def entries(self) -> list[dict[str, Any]]:
        sorted_entries = sorted(
            self._by_complexity.values(),
            key=lambda e: (
                e["complexity"],
                _parsimonious_loss(e["loss"], e["complexity"], self._parsimony),
                e["hash"],
            ),
        )
        return [
            {k: v for k, v in e.items() if k != "expression"}
            for e in sorted_entries
        ]

    # ── Pareto frontier ────────────────────────────────────────────

    def calculate_pareto_frontier(self) -> list[dict[str, Any]]:
        sorted_by_cplx = sorted(
            self._by_complexity.values(),
            key=lambda e: e["complexity"],
        )
        frontier: list[dict[str, Any]] = []
        best_loss = float("inf")
        for entry in sorted_by_cplx:
            if entry["loss"] < best_loss:
                frontier.append({k: v for k, v in entry.items() if k != "expression"})
                best_loss = entry["loss"]
        return frontier

    # ── Score computation ──────────────────────────────────────────

    def compute_scores(
        self,
        loss_scale: str = "linear",
    ) -> list[dict[str, Any]]:
        """Score each Pareto member by improvement rate over the previous point.

        Two modes:
        - ``"linear"``: direct score = (prev_loss - loss) / prev_loss
        - ``"log"``: zero-centered score = prev_score + (prev_loss - loss) / prev_loss
          (throws if any loss < 0)
        """
        frontier = self.calculate_pareto_frontier()
        if not frontier:
            return []

        if loss_scale == "log":
            for e in frontier:
                if e["loss"] < 0.0:
                    raise ValueError(
                        f"negative loss ({e['loss']}) incompatible with log-loss scaling"
                    )

        scored: list[dict[str, Any]] = []
        prev_loss: float | None = None
        prev_score = 0.0
        for entry in frontier:
            if prev_loss is None:
                score = 0.0
            elif loss_scale == "linear":
                improvement = (prev_loss - entry["loss"]) / max(prev_loss, 1e-16)
                score = improvement
            else:
                improvement = (prev_loss - entry["loss"]) / max(prev_loss, 1e-16)
                score = prev_score + improvement
            scored.append({**entry, "score": score})
            prev_loss = entry["loss"]
            prev_score = score
        return scored

    # ── String formatting ──────────────────────────────────────────

    def string_dominating_pareto_curve(
        self,
        variable_prefix: str = "y",
        loss_scale: str = "linear",
    ) -> str:
        scored = self.compute_scores(loss_scale=loss_scale)
        if not scored:
            return ""

        lines: list[str] = []
        lines.append(f"Dominating Pareto Curve ({variable_prefix} = ...)")
        lines.append(f"{'Complexity':>10} {'Loss':>12} {'Score':>12} {'Equation':>30}")
        lines.append("-" * 70)
        for entry in scored:
            cplx = entry["complexity"]
            loss_fmt = f"{entry['loss']:.6g}"
            score_fmt = f"{entry['score']:.6g}"
            eq = entry["canonical_expression"][:28]
            lines.append(f"{cplx:>10} {loss_fmt:>12} {score_fmt:>12} {eq:>30}")
        return "\n".join(lines)

    # ── Dunder methods ─────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self._by_complexity)
        if n == 0:
            return f"<HallOfFame max_size={self.max_size} empty>"
        best = self.best()
        best_loss = best["loss"] if best else "N/A"
        return (
            f"<HallOfFame max_size={self.max_size} entries={n} "
            f"best_loss={best_loss}>"
        )

    def __str__(self) -> str:
        return self.string_dominating_pareto_curve()

    def __copy__(self) -> HallOfFame:
        new = HallOfFame(max_size=self.max_size)
        new._by_complexity = copy.deepcopy(self._by_complexity)
        new._parsimony = self._parsimony
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> HallOfFame:
        new = HallOfFame(max_size=self.max_size)
        new._by_complexity = copy.deepcopy(self._by_complexity, memo)
        new._parsimony = copy.deepcopy(self._parsimony, memo)
        return new
