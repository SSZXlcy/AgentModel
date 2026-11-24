"""Shared value types. Everything here is plain data so it can be written to JSONL as-is."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    return value


def digest(value: Any, length: int = 12) -> str:
    """Stable short identity of any JSON-ready value."""
    raw = json.dumps(json_ready(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Cost:
    """The three quantities the objective score charges for."""

    train_seconds: float = 0.0
    labels: int = 0
    llm_tokens: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.train_seconds + other.train_seconds, self.labels + other.labels,
                    self.llm_tokens + other.llm_tokens)

    def __sub__(self, other: "Cost") -> "Cost":
        return Cost(self.train_seconds - other.train_seconds, self.labels - other.labels,
                    self.llm_tokens - other.llm_tokens)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataRecipe:
    """Which labelled data a candidate trains on. The current window's fit half is always included."""

    include_base: bool
    past_windows: tuple[int, ...]
    name: str = ""

    @property
    def recipe_id(self) -> str:
        return digest({"base": self.include_base, "windows": list(self.past_windows)})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "include_base": self.include_base,
                "past_windows": list(self.past_windows), "recipe_id": self.recipe_id}


@dataclass(frozen=True)
class TrainSpec:
    """One inner-loop candidate: exactly one dimension differs from its siblings."""

    recipe: DataRecipe
    lr_scale: float = 1.0
    max_epochs: int = 3

    @property
    def label(self) -> str:
        return f"{self.recipe.name or self.recipe.recipe_id}@lr{self.lr_scale:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"recipe": self.recipe.to_dict(), "lr_scale": self.lr_scale,
                "max_epochs": self.max_epochs, "label": self.label}


@dataclass(frozen=True)
class DriftJudgment:
    """The agent's (or the fallback table's) reading of one window's evidence."""

    drift_type: str          # covariate | concept | both | none
    recipe: str              # all | adaptive | recent:N
    confidence: float
    rationale: str
    source: str              # llm | rule
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
