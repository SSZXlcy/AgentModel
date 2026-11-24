"""Validation experiments (W3). Each compares the current strategy with one candidate change on data
that has already happened, and returns a paired difference with a band.

Which experiment runs is decided by the operator manual (operators.py):

  protocol "fixed"   does the change make a better MODEL? (data, training and code operators)
                     before the stream: group k-fold on the official training set, both arms trained
                       from scratch with the official configuration
                     during the run:   replay the last N windows with the labels already held; every
                       window both arms train and serve their model unconditionally, so the comparison
                       is not blocked by promotion gates
  protocol "full"    does the change make better DECISIONS? (gate / serve / split loosening)
                     during the run only: replay the full controller, gates included
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
import time

import numpy as np

from . import codebase, metrics
from . import slots as slot_rules
from .codebase import CodeVersion
from .config import PolicyConfig
from .contracts import DataRecipe
from .data import write_lines

EXPERIMENT_FIELDS = {"cap_per_group", "cap_group_by", "recent_windows_only", "split_mode", "split_group_by",
                     "serve_mode", "z", "label_fraction", "max_epochs", "max_candidates", "min_samples",
                     "confirm_windows", "gate_fraction", "lr_scales_overfit", "cooldown_windows", "min_gate_units",
                     "lr_scale", "branch_from", "mode", "gate_mode", "train_trigger", "label_sampling",
                     "direct_epochs"}


def candidate_code(base: CodeVersion, edits: list[dict] | None, store: Path) -> CodeVersion:
    if not edits:
        return base
    source = slot_rules.apply_edits(base.slots_source(), edits)
    return codebase.derive_version(base, source, edits, store)


def _train(version: CodeVersion, profile: str, train: Path, val: Path, out: Path, stage: str, **kwargs) -> dict:
    """Official trainer; one retry on a Windows rename refusal (the output does not exist then)."""
    for attempt in range(2):
        try:
            return version.modules().train.train_run(str(version.profile_path(profile)), train, val, out,
                                                     stage_id=stage, **kwargs)
        except PermissionError:
            if attempt or out.exists():
                raise
            time.sleep(0.5)


def _paired(a: list[float], b: list[float], z: float) -> dict:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    band = z * sd / np.sqrt(max(n, 1)) * metrics.t_factor(n - 1)     # few units → wider band
    return {"n": n, "mean_diff": float(d.mean()) if n else 0.0, "band": band,
            "worst_diff": float(d.min()) if n else 0.0,
            "baseline_mean": float(np.mean(a)) if n else None, "candidate_mean": float(np.mean(b)) if n else None}


CUT_POINTS_PER_ENTITY = 3
FUTILITY_FOLDS = 2          # an opening experiment stops once this many folds show no gain on average


def group_kfold(ctrl, candidate: PolicyConfig, code: CodeVersion, group_by: list[str] | None,
                folds: int, scratch: Path) -> dict:
    """Opening experiment on the official training set (the stream is never touched).

    Folds hold out whole entities (the discovered entity key unless another key is given); independent
    rows are dealt round-robin. Training rows per fold are capped at `opening_rows` (evenly spread, so every
    entity keeps its share) because these trainings are paid for. When the stream shows each entity once
    (snapshot), held-out entities are scored at a few random cut points, as such a stream would score them.
    The baseline arm of a fold is computed once per round and reused by every candidate."""
    from .controller import _cap
    from .signature import stream_format
    s = ctrl.session
    table = s.base
    rows = s.base_rows
    group_by = group_by or s.entity or None
    snapshot = bool(s.entity) and s.structure.get("countdown") and stream_format(ctrl, s.entity) == "snapshot"
    if group_by:
        keys = table.key(group_by, rows)
        unique = list(dict.fromkeys(keys.tolist()))
        k = min(folds, len(unique))
        fold_of = {g: i % k for i, g in enumerate(sorted(unique))}
        assign = np.asarray([fold_of[g] for g in keys.tolist()])
    else:
        k = folds
        assign = np.arange(len(rows)) % k
    limit = max(int(ctrl.policy.opening_rows), 1)
    cache = getattr(ctrl, "_kfold_baseline", None)
    if cache is None or cache.get("round") != ctrl.state.get("strategy_version"):
        cache = {"round": ctrl.state.get("strategy_version"), "folds": {}}
        ctrl._kfold_baseline = cache
    base_key = (tuple(group_by or ()), k, limit, snapshot, ctrl.code.version_id, ctrl.policy.cap_per_group,
                tuple(ctrl.policy.cap_group_by), ctrl.policy.lr_scale)
    baseline, trial = [], []
    work = {"baseline": 0, "candidate": 0}          # training work of each arm (cached folds included)
    spent = 0                                       # training actually run now (what this experiment costs)
    for f in range(k):
        test_rows, train_rows = rows[assign == f], rows[assign != f]
        if len(test_rows) == 0 or len(train_rows) == 0:
            continue
        if len(train_rows) > limit:
            train_rows = train_rows[np.linspace(0, len(train_rows) - 1, limit).astype(np.int64)]
        if snapshot:
            test_rows = _cut_points(table, test_rows, group_by, seed=f)
        val = write_lines(scratch / f"fold{f}" / "val.jsonl", [table.raw[i] for i in test_rows])
        for arm, policy, version in (("baseline", ctrl.policy, ctrl.code), ("candidate", candidate, code)):
            if arm == "baseline" and (base_key, f) in cache["folds"]:
                u, w = cache["folds"][(base_key, f)]
                baseline.append(u)
                work[arm] += w
                continue
            items = [(table.raw[i], table.meta[i], table.y[i]) for i in train_rows]
            if policy.cap_per_group and policy.cap_group_by:
                items = _cap(items, policy.cap_per_group, policy.cap_group_by, s.classes)
            path = write_lines(scratch / f"fold{f}" / f"{arm}.jsonl", [line for line, _, _ in items])
            out = scratch / f"fold{f}" / f"{arm}_out"
            report = _train(version, ctrl.task.profile, path, val, out, f"exp-{arm}-{f}",
                            learning_rate=s.base_lr * policy.lr_scale)
            model = version.modules().model.Model.load(out / "checkpoint.npz")
            u = metrics.utility(s.task_type, table.y[test_rows], model.predict(table.x[test_rows]))
            w = len(items) * report["final_epochs"]
            if arm == "baseline":
                baseline.append(u)
                cache["folds"][(base_key, f)] = (u, w)
            else:
                trial.append(u)
            work[arm] += w
            spent += w
        if len(trial) >= FUTILITY_FOLDS and len(trial) < k and np.mean(np.subtract(trial, baseline)) <= 0:
            stopped = True          # no sign of a gain after a few folds: stop paying for the rest
            break
    else:
        stopped = False
    result = _paired(baseline, trial, ctrl.policy.z)
    result.update(mode="group_kfold", group_by=group_by, folds=k, folds_run=len(trial), stopped_early=stopped,
                  row_epochs=work, spent_row_epochs=spent,
                  scoring="cut_points" if snapshot else "all_rows", train_rows_per_fold=limit,
                  cost_change=(work["candidate"] - work["baseline"]) / max(work["baseline"], 1))
    return result


def _cut_points(table, rows: np.ndarray, group_by: list[str] | None, seed: int) -> np.ndarray:
    """A few random rows of every held-out entity: how a stream showing each unit once scores a model."""
    keys = table.key(group_by, rows) if group_by else np.asarray([str(i) for i in rows])
    rng = np.random.default_rng([seed, 7])
    keep = []
    for key in dict.fromkeys(keys.tolist()):
        own = rows[keys == key]
        keep.extend(rng.choice(own, size=min(CUT_POINTS_PER_ENTITY, len(own)), replace=False).tolist())
    return np.asarray(sorted(keep), dtype=np.int64)


def _past(ctrl, windows: int) -> tuple[list[int], dict[int, np.ndarray]]:
    now = ctrl.state["next_window"]
    past = [w for w in ctrl.window_ids if w < now][-windows:]
    oracle = {w: np.asarray(ctrl.state["pool"].get(str(w), []), dtype=np.int64) for w in past}
    return past, oracle


def past_windows_fixed(ctrl, candidate: PolicyConfig, code: CodeVersion, windows: int, scratch: Path) -> dict:
    """In-run model-quality experiment. Both arms start from the model that served at the first
    replayed window; each window they train on their own recipe and serve the result, and are scored
    on the next window's held labels. Gates play no part, so a better model shows as a difference."""
    from .executor import Executor
    from .ledger import Ledger
    from .recorder import Recorder
    from .stream import LocalStream
    from .tree import Tree
    past, oracle = _past(ctrl, windows)
    scored = [w for w in past[1:] if len(oracle[w])]
    if len(scored) < 2:
        return {"mode": "past_windows_fixed", "n": 0, "skipped": "fewer than 3 labelled past windows"}
    s = ctrl.session
    stream = LocalStream(s.stream_table, dataclasses.replace(ctrl.task, label_protocol="immediate"), s.task_type)
    for w in past:
        stream.request_labels(w, oracle[w])
    row = next(r for r in ctrl.state["rows"] if r["window"] == past[0])
    start = ctrl.tree.checkpoint(row["served_node"])
    val_path = write_lines(scratch / "anchor.jsonl", s.anchor.raw)
    arms: dict[str, dict[int, float]] = {}
    work = {}
    for arm, policy, version in (("baseline", ctrl.policy, ctrl.code), ("candidate", candidate, code)):
        run_dir = scratch / arm
        tree = Tree(run_dir)
        Model = version.modules().model.Model
        tree.add_root(start, version.version_id, Model.load(start).meta)
        executor = Executor(version, ctrl.task.profile, tree, Ledger(), Recorder(run_dir), s.seconds_per_row_epoch)
        serving, pool, scores = "n0000", {}, {}
        for i, w in enumerate(past):
            positions = oracle[w]
            if i > 0 and len(positions):
                x = stream.features(w)[1][positions]
                scores[w] = metrics.utility(s.task_type, stream.labels(w, positions), tree.model(serving, Model).predict(x))
            if i == len(past) - 1 or not len(positions):
                continue
            history = sorted(pool)
            if policy.recent_windows_only:
                recipe = DataRecipe(False, tuple(history[-policy.recent_windows_only:]), "recent")
            else:
                recipe = DataRecipe(True, tuple(history), "all")
            path, rows, _ = s.materialize(run_dir / "data" / f"w{w:03d}.jsonl", stream, recipe, pool,
                                          [(w, int(q)) for q in positions], policy)
            parent = "n0000" if policy.branch_from == "static" else serving
            node = executor.train(tree.nodes[parent], path, val_path, rows, epochs=max(1, policy.max_epochs),
                                  learning_rate=s.base_lr * policy.lr_scale, window=w, recipe_id=recipe.recipe_id,
                                  spec=f"fixed-{arm}")
            path.unlink()
            serving, pool[w] = node.node_id, positions
        arms[arm], work[arm] = scores, executor.row_epochs
    common = [w for w in scored if w in arms["baseline"] and w in arms["candidate"]]
    result = _paired([arms["baseline"][w] for w in common], [arms["candidate"][w] for w in common], ctrl.policy.z)
    result.update(mode="past_windows_fixed", windows=past, row_epochs=work,
                  cost_change=(work["candidate"] - work["baseline"]) / max(work["baseline"], 1))
    return result


def past_windows_full(ctrl, candidate: PolicyConfig, code: CodeVersion, windows: int, seeds: list[int],
                      scratch: Path) -> dict:
    """In-run decision experiment: replay the full controller (gates included) on held labels."""
    from .controller import Controller
    past, oracle = _past(ctrl, windows)
    if len(past) < 2 or sum(len(v) for v in oracle.values()) < 8:
        return {"mode": "past_windows_full", "n": 0, "skipped": "not enough labelled past windows"}
    row = next(r for r in ctrl.state["rows"] if r["window"] == past[0])
    start = ctrl.tree.checkpoint(row["served_node"])
    arms: dict[str, list[float]] = {"baseline": [], "candidate": []}
    work = {"baseline": 0, "candidate": 0}
    for arm, policy, version in (("baseline", ctrl.policy, ctrl.code), ("candidate", candidate, code)):
        replay_policy = dataclasses.replace(policy, budget_train_seconds=None, budget_labels=None)
        for seed in seeds:
            sub = Controller(ctrl.task, replay_policy, version, scratch / arm / f"seed{seed}", session=ctrl.session,
                             label_seed=seed, label_oracle=oracle, windows=past, start_checkpoint=start,
                             score_on_labels=True)
            summary = sub.run()
            arms[arm] += [w["utility"] for w in summary["per_window"] if w["utility"] is not None]
            work[arm] += summary["train_row_epochs"]
    result = _paired(arms["baseline"], arms["candidate"], ctrl.policy.z)
    result.update(mode="past_windows_full", windows=past, seeds=seeds, row_epochs=work,
                  cost_change=(work["candidate"] - work["baseline"]) / max(work["baseline"], 1))
    return result


def run(ctrl, protocol: str, spec: dict, candidate: PolicyConfig, code: CodeVersion, scratch: Path) -> dict:
    """Execute the experiment the manual prescribes; the scratch directory is removed afterwards.
    `spec` may override folds / windows / seeds / group_by. Folds and windows can only be raised above the
    policy's values: fewer of them would weaken the evidence a change must clear."""
    spec = dict(spec or {})
    for key, floor in (("folds", ctrl.policy.experiment_folds), ("windows", ctrl.policy.experiment_windows)):
        try:
            spec[key] = max(int(spec.get(key, floor)), floor)
        except (TypeError, ValueError):
            spec[key] = floor
    started = ctrl.state["next_window"] > (ctrl.window_ids[0] if ctrl.window_ids else 0)
    try:
        if protocol == "fixed" and not started:
            group_by = spec.get("group_by") or None
            return group_kfold(ctrl, candidate, code, group_by, int(spec.get("folds", ctrl.policy.experiment_folds)),
                               scratch)
        if protocol == "fixed":
            return past_windows_fixed(ctrl, candidate, code, int(spec.get("windows", ctrl.policy.experiment_windows)),
                                      scratch)
        if protocol == "full":
            if not started:
                return {"mode": "past_windows_full", "n": 0, "skipped": "decision changes can only be replayed in-run"}
            seeds = [int(v) for v in spec.get("seeds", [0])][:3]
            return past_windows_full(ctrl, candidate, code, int(spec.get("windows", ctrl.policy.experiment_windows)),
                                     seeds, scratch)
        return {"mode": protocol, "n": 0, "skipped": f"protocol {protocol!r} cannot validate a change"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def passed(result: dict, cost_tolerance: float = 0.25) -> tuple[bool, str]:
    if not result.get("n"):
        return False, result.get("skipped", "no comparable samples")
    if result["mean_diff"] <= result["band"]:
        return False, f"gain {result['mean_diff']:+.4f} within band {result['band']:.4f}"
    relative_gain = result["mean_diff"] / max(abs(result.get("baseline_mean") or 0.0), 1e-9)
    if result.get("cost_change", 0) > cost_tolerance + 5 * relative_gain:
        return False, f"training work {result['cost_change']:+.0%} not paid for by the gain"
    return True, f"gain {result['mean_diff']:+.4f} > band {result['band']:.4f}"
