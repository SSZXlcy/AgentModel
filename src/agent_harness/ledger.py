"""Hard budgets. An action is checked against the ledger before it runs, never after."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import Cost


class BudgetError(RuntimeError):
    """The action would exceed a hard budget; it was not executed."""


@dataclass
class Ledger:
    max_train_seconds: float | None = None
    max_labels: int | None = None
    max_llm_tokens: int | None = None
    spent: Cost = field(default_factory=Cost)

    def _over(self, cost: Cost) -> list[str]:
        total = self.spent + cost
        reasons = []
        if self.max_train_seconds is not None and total.train_seconds > self.max_train_seconds:
            reasons.append(f"train_seconds {total.train_seconds:.2f} > {self.max_train_seconds}")
        if self.max_labels is not None and total.labels > self.max_labels:
            reasons.append(f"labels {total.labels} > {self.max_labels}")
        if self.max_llm_tokens is not None and total.llm_tokens > self.max_llm_tokens:
            reasons.append(f"llm_tokens {total.llm_tokens} > {self.max_llm_tokens}")
        return reasons

    def affordable(self, estimate: Cost) -> bool:
        return not self._over(estimate)

    def require(self, estimate: Cost) -> None:
        reasons = self._over(estimate)
        if reasons:
            raise BudgetError("; ".join(reasons))

    def charge(self, actual: Cost) -> None:
        self.spent = self.spent + actual

    def remaining(self) -> dict:
        def left(limit, used):
            return None if limit is None else max(0.0, limit - used)
        return {"train_seconds": left(self.max_train_seconds, self.spent.train_seconds),
                "labels": left(self.max_labels, self.spent.labels),
                "llm_tokens": left(self.max_llm_tokens, self.spent.llm_tokens)}

    def to_dict(self) -> dict:
        return {"limits": {"train_seconds": self.max_train_seconds, "labels": self.max_labels,
                           "llm_tokens": self.max_llm_tokens},
                "spent": self.spent.to_dict(), "remaining": self.remaining()}

    @classmethod
    def from_dict(cls, data: dict) -> "Ledger":
        limits = data["limits"]
        return cls(limits["train_seconds"], limits["labels"], limits["llm_tokens"], Cost(**data["spent"]))
