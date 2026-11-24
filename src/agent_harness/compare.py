"""Comparison (W7): the same stream and seeds under each arm.

  static        the official checkpoint, never updated (threshold ③)
  naive         every window, continue training on all labels held so far
  inner         the inner loop alone, default strategy, no supervisor
  system        inner loop + supervisor with the deterministic explorer
  system_llm    inner loop + supervisor with the LLM explorer (only with --llm)

`arms` re-runs only some arms and merges them into an existing compare.json (e.g. after a change
that only affects the LLM arm); the others keep their stored results.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .agent import Agent
from .codebase import CodeVersion
from .config import PolicyConfig, TaskConfig
from .controller import Controller, Session
from .replay import replay_code
from .supervisor import Supervisor

ARMS = ("naive", "inner", "system", "system_llm")


def compare(task: TaskConfig, policy: PolicyConfig, code: CodeVersion, seeds: list[int], root: Path,
            code_store: Path, provider=None, log=print, arms: list[str] | None = None) -> dict:
    root = Path(root)
    previous = None
    if arms:
        previous = json.loads((root / "compare.json").read_text(encoding="utf-8"))
        if previous["seeds"] != seeds:
            raise ValueError(f"stored comparison used seeds {previous['seeds']}")
        for name in arms:
            shutil.rmtree(root / name, ignore_errors=True)
    else:
        shutil.rmtree(root, ignore_errors=True)
        arms = [a for a in ARMS if a != "system_llm" or provider is not None]
    session = Session(task, code)
    static = _static_series(session, code)
    runs: dict[str, list[dict]] = {}
    if "naive" in arms:
        naive = replay_code(task, policy, code, seeds, root / "naive")
        runs["naive"] = [{"per_window": run["per_window"], "row_epochs": run["row_epochs"], "labels": run["labels"],
                          "tokens": 0} for run in naive["runs"]]
    for name, llm in (("inner", None), ("system", False), ("system_llm", True)):
        if name not in arms:
            continue
        if llm and provider is None:
            raise ValueError("the system_llm arm needs --llm")
        runs[name] = []
        for seed in seeds:
            supervisor = None
            if llm is not None:
                supervisor = Supervisor(Agent(provider if llm else None), root / name / "memory", code_store, use_llm=llm)
            summary = Controller(task, policy, code, root / name / f"seed{seed}", session=session, label_seed=seed,
                                 supervisor=supervisor).run()
            runs[name].append({"per_window": [w["utility"] for w in summary["per_window"]],
                               "row_epochs": summary["train_row_epochs"], "labels": summary["cost"]["labels"],
                               "tokens": summary["cost"]["llm_tokens"], "run_dir": summary["run_dir"],
                               "strategy_versions": summary.get("strategy_versions", 0),
                               "modes": _modes(root / name / f"seed{seed}")})
            log(f"  {name} seed {seed}: mean {np.mean(runs[name][-1]['per_window']):.4f}")
    runs["static"] = [{"per_window": static, "row_epochs": 0, "labels": 0, "tokens": 0} for _ in seeds]
    table = dict(previous["arms"]) if previous else {}
    for name, items in runs.items():
        cells = np.asarray([r["per_window"] for r in items], dtype=float)
        diff = cells - np.asarray(static, dtype=float)[None, :]
        n = diff.size
        table[name] = {"mean": float(cells.mean()), "worst_window": float(cells.min()),
                       "vs_static": float(diff.mean()),
                       "vs_static_band": float(2 * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                       "row_epochs": float(np.mean([r["row_epochs"] for r in items])),
                       "labels": float(np.mean([r["labels"] for r in items])),
                       "tokens": float(np.mean([r["tokens"] for r in items])),
                       "strategy_versions": [r.get("strategy_versions", 0) for r in items],
                       "modes": [r.get("modes") for r in items],
                       "rhythm_switches": float(np.mean([max(len(r.get("modes") or ["standard@0"]) - 2, 0)
                                                         for r in items])),
                       "run_dirs": [r.get("run_dir") for r in items]}
    _gain_per_cost(table)
    return {"task": task.task, "label_protocol": task.label_protocol, "seeds": seeds, "windows": len(static),
            "arms": {name: table[name] for name in ("static",) + ARMS if name in table}}


TOKEN_UNIT = 10_000     # tokens counted as one unit of cost (the naive arm uses none to normalise by)


def _gain_per_cost(table: dict) -> None:
    """The objective score's shape: gain over static per unit of cost, where cost adds training work and
    labels (each relative to the naive arm, which trains every window on every label) and LLM tokens."""
    naive = table.get("naive")
    if not naive:
        return
    for name, arm in table.items():
        if name == "static":
            continue
        cost = (arm["row_epochs"] / max(naive["row_epochs"], 1) + arm["labels"] / max(naive["labels"], 1)
                + arm["tokens"] / TOKEN_UNIT)
        arm["cost_units"] = round(cost, 4)
        arm["gain_per_cost"] = arm["vs_static"] / cost if cost > 0 else None


def _static_series(session: Session, code: CodeVersion) -> list[float]:
    """Platform score of the official checkpoint on every window (what "never update" earns)."""
    model = code.modules().model.Model.load(session.task.checkpoint)
    stream = session.new_stream()
    return [stream.score(k, model.predict(stream.features(k)[1]))["utility"] for k in range(len(stream))]


def _modes(run_dir: Path) -> list[str]:
    """Rhythm of each strategy version in order, e.g. ["standard@0", "event@0", "adaptive@1"]."""
    out = ["standard@0"]
    for path in sorted((Path(run_dir) / "strategies").glob("s*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        mode = item["policy"].get("mode", "standard")
        if not out[-1].startswith(mode + "@"):
            out.append(f"{mode}@{item['window']}")
    return out
