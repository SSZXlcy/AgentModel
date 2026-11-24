"""Task and policy configuration loaded from configs/*.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    task: str                         # A | B | C
    package: Path                     # official package directory
    profile: str                      # official profile name
    stream: str = "validation"        # which public split simulates the stream
    stream_file: Path | None = None   # any JSONL in the official format; overrides `stream` (derived public files)
    window_size: int = 300            # rows per window (used when window_groups is unset)
    window_groups: int | None = None  # entities per window (whole entities never split across windows)
    # Meta fields naming one entity. Empty: the entity key structure.detect() finds in the training set
    # (the same key a person would write; nothing task-specific has to be configured).
    group_key: list[str] = field(default_factory=list)
    anchor_fraction: float = 0.1      # base rows held out to monitor forgetting
    base_rows_cap: int | None = None  # fixed seeded subset of the base set used in recipes (overrides the budget)
    # One training call should not cost more than this many rows of the official set: a larger set is
    # subsampled once (seeded) for every recipe. A cost rule, the same for every task.
    base_rows_budget: int = 20_000
    # Extra strings that must never appear in stored experiences (development guard). The sealed-data
    # directory is always refused, whatever is listed here.
    leak_markers: list[str] = field(default_factory=list)
    # Simulated platform protocol, not a strategy choice: when do labels exist?
    #   immediate          the label is returned in the window it is requested
    #   on_group_complete  a row's label is released only once its whole entity has left the stream
    #                      (countdown targets such as remaining life exist only after failure)
    label_protocol: str = "immediate"
    protocol_group_by: list[str] = field(default_factory=list)   # empty: detected from the data

    @property
    def train_path(self) -> Path:
        return self.package / "public" / "train.jsonl"

    @property
    def stream_path(self) -> Path:
        if self.stream_file is not None:
            return Path(self.stream_file)
        return self.package / "public" / f"{self.stream}.jsonl"

    @property
    def guarded_markers(self) -> list[str]:
        return sorted(set(self.leak_markers) | {"data/sealed", "data\\sealed"})

    @property
    def checkpoint(self) -> Path:
        return self.package / "checkpoint.npz"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["package"] = self.package.as_posix()
        data["stream_file"] = Path(self.stream_file).as_posix() if self.stream_file is not None else None
        return data


@dataclass
class PolicyConfig:
    """π: every rule parameter of the inner loop. The outer loop may evolve these."""

    policy_id: str = "default"
    # labels
    label_fraction: float = 0.5
    gate_fraction: float = 0.5
    # search
    max_candidates: int = 3
    max_epochs: int = 3
    min_samples: int = 300
    lr_scales_overfit: list[float] = field(default_factory=lambda: [0.5, 0.25])
    # statistics
    z: float = 2.0
    bootstrap_reps: int = 300
    # gates
    min_gate_units: int = 8             # independent entities a gate decision must rest on (1 entity = time blocks)
    min_roi: float = 0.0
    anchor_veto: float = 0.25
    confirm_windows: int = 2
    static_floor_windows: int = 2
    cooldown_windows: int = 1
    # drift
    drift_quantile: float = 0.95
    drift_reference_rows: int = 2000
    # outer-layer trigger
    stall_windows: int = 2
    # budgets (None = unlimited)
    budget_train_seconds: float | None = 120.0
    budget_labels: int | None = None
    budget_llm_tokens: int | None = 200_000
    # cost weights used by the ROI rule (gain per unit cost)
    cost_per_second: float = 1.0
    cost_per_label: float = 0.0
    cost_per_token: float = 1e-4
    # replay protocol for code evaluation
    replay_epochs: int = 2
    # ---- strategy fields the explorer may change (defaults are the least restrictive) ----
    split_mode: str = "random"          # random | by_group | by_order  (how labelled rows split into fit/gate)
    split_group_by: list[str] = field(default_factory=list)   # meta fields naming one entity
    serve_mode: str = "trial"           # trial | shadow | freeze
    cap_per_group: int | None = None    # training rows kept per group, evenly spread in time
    cap_group_by: list[str] = field(default_factory=list)     # meta fields, or ["answer"] for classes
    recent_windows_only: int | None = None   # train only on the last N labelled windows (no base set)
    lr_scale: float = 1.0               # every candidate's learning rate = official rate × lr_scale
    branch_from: str = "serving"        # serving | static: where candidates start training from
    # ---- rhythm: how the inner loop follows the stream (set together by the mode.* operators) ----
    mode: str = "standard"              # standard | adaptive | event | conserve | custom (a label for records)
    gate_mode: str = "in_window"        # in_window: half the fresh labels judge candidates in this window
                                        # next_window: all fresh labels train one candidate; the next window judges it
    train_trigger: str = "every_window" # every_window | on_completion (train only when an entity has finished)
    label_sampling: str = "random"      # random | spread (evenly along the window, for correlated rows)
    direct_epochs: int = 2              # epochs of the single candidate under next_window / on_completion
    # ---- supervisor: when to continue, review or update the strategy ----
    health_window: int = 3              # k: windows looked back by the health checks
    review_every: int = 8               # N: periodic light review
    probation_windows: int = 3          # M: no further updates while an update is being judged
    max_updates: int = 4                # strategy updates per run
    explore_steps: int = 12             # tool calls per exploration
    explore_tokens: int = 12_000        # LLM tokens per exploration (soft: the next turn must plan)
    run_llm_tokens: int = 30_000        # hard cap on LLM tokens per run (tokens are a scored cost)
    experiment_windows: int = 6         # past windows replayed by an in-run experiment
    experiment_folds: int = 10          # folds of an opening group k-fold (leave-one-group-out up to 10 groups)
    # Opening experiments train from initialisation and are paid for like any training: training rows per
    # fold are capped (evenly spread, so every entity keeps its share), and only the most credible
    # candidates of each layer are tried.
    opening_rows: int = 6000
    opening_candidates: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> "PolicyConfig":
        data = self.to_dict()
        data.update(changes)
        return PolicyConfig(**data)


def _read(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def load_task(path: Path | str, root: Path | str = ".") -> TaskConfig:
    data = _read(path)
    known = {f.name for f in fields(TaskConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown task fields {sorted(unknown)}")
    data["package"] = Path(root) / data["package"]
    if data.get("stream_file"):
        data["stream_file"] = Path(root) / data["stream_file"]
    key = data.get("group_key") or []
    data["group_key"] = [key] if isinstance(key, str) else list(key)
    return TaskConfig(**data)


def load_policy(path: Path | str | None) -> PolicyConfig:
    if path is None:
        return PolicyConfig()
    data = _read(path)
    known = {f.name for f in fields(PolicyConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown policy fields {sorted(unknown)}")
    return PolicyConfig(**data)
