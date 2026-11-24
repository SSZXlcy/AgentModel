"""Replays used by the outer layer. Everything is deterministic, so a replay is an exact
counterfactual, not an estimate; the only randomness across seeds is which rows get labelled.

replay_code    freezes the decisions (a fixed protocol: every window, train from the serving
               model on all labelled data, serve the result) and varies only the code θ.
replay_policy  freezes θ and re-runs the full controller under a policy π.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from . import metrics
from .codebase import CodeVersion
from .config import PolicyConfig, TaskConfig
from .contracts import DataRecipe, digest
from .controller import Controller, Session
from .data import write_lines
from .executor import Executor
from .ledger import Ledger
from .recorder import Recorder, write_json
from .tree import Tree


def protocol_id(policy: PolicyConfig, task: TaskConfig | None = None) -> str:
    return digest({"label_fraction": policy.label_fraction, "epochs": policy.replay_epochs,
                   "cap": [policy.cap_per_group, policy.cap_group_by],
                   "protocol": task.label_protocol if task else "immediate"})


def _warm_up(session: Session, code: CodeVersion, root: Path) -> None:
    """One throw-away epoch so the first measured replay does not pay process warm-up; training
    seconds are a scored cost and the cost rule compares them across versions."""
    scratch = root / ".warmup" / code.version_id
    shutil.rmtree(scratch, ignore_errors=True)
    half = len(session.anchor.raw) // 2   # train and validation IDs must not overlap
    train = write_lines(scratch / "train.jsonl", session.anchor.raw[:half])
    val = write_lines(scratch / "val.jsonl", session.anchor.raw[half:])
    code.modules().train.train_run(str(code.profile_path(session.task.profile)), train, val, scratch / "out",
                                   resume=session.task.checkpoint, epochs=1, stage_id="warmup")
    shutil.rmtree(scratch, ignore_errors=True)


def replay_code(task: TaskConfig, policy: PolicyConfig, code: CodeVersion, seeds: list[int], root: Path) -> dict:
    session = Session(task, code)
    Model = code.modules().model.Model
    runs = []
    warmed = False
    for seed in seeds:
        run_dir = root / protocol_id(policy, task) / code.version_id / f"seed_{seed}"
        done = run_dir / "replay.json"
        if done.is_file():
            runs.append(json.loads(done.read_text(encoding="utf-8")))
            continue
        if not warmed:
            _warm_up(session, code, root)
            warmed = True
        shutil.rmtree(run_dir, ignore_errors=True)
        tree, ledger, recorder = Tree(run_dir), Ledger(), Recorder(run_dir)
        tree.add_root(task.checkpoint, code.version_id, Model.load(task.checkpoint).meta)
        executor = Executor(code, task.profile, tree, ledger, recorder, session.seconds_per_row_epoch)
        stream = session.new_stream()
        val_path = write_lines(run_dir / "data" / "anchor.jsonl", session.anchor.raw)
        serving, pool, utilities, outputs, truths = "n0000", {}, [], [], []
        for k in range(len(stream)):
            _, x, _ = stream.features(k)
            out = tree.model(serving, Model).predict(x)
            utilities.append(stream.score(k, out)["utility"])
            delivered = stream.deliver(k)
            positions = session.label_positions(seed, k, len(x), policy.label_fraction)
            released, labels = stream.request_labels(k, positions)
            outputs.append(out[released])
            truths.append(labels)
            fresh = [(w, int(q)) for w in sorted(delivered) for q in delivered[w]] + [(k, int(q)) for q in released]
            if k == len(stream) - 1:
                break
            if fresh:
                recipe = DataRecipe(True, tuple(sorted(pool)), "all")
                path, rows, _ = session.materialize(run_dir / "data" / f"w{k:03d}.jsonl", stream, recipe, pool,
                                                    fresh, policy)
                node = executor.train(tree.nodes[serving], path, val_path, rows, epochs=policy.replay_epochs,
                                      learning_rate=session.base_lr, window=k, recipe_id=recipe.recipe_id, spec="replay")
                path.unlink()
                serving = node.node_id
            for w, q in fresh:
                pool.setdefault(w, [])
                pool[w] = np.asarray(sorted(set(np.asarray(pool[w]).tolist()) | {q}), dtype=np.int64)
        y = np.concatenate(truths)
        out = np.concatenate(outputs) if outputs else np.zeros((0,))
        result = {"seed": seed, "code_version": code.version_id, "per_window": utilities,
                  "train_seconds": ledger.spent.train_seconds, "row_epochs": executor.row_epochs,
                  "labels": int(sum(len(p) for p in pool.values())), "errors": _errors(session, y, out)}
        write_json(done, result)
        tree.save()
        runs.append(result)
    return _aggregate(runs)


def replay_policy(task: TaskConfig, policy: PolicyConfig, code: CodeVersion, seeds: list[int], root: Path,
                  session: Session | None = None) -> dict:
    session = session or Session(task, code)
    runs = []
    for seed in seeds:
        run_dir = root / digest(policy.to_dict()) / code.version_id / f"seed_{seed}"
        done = run_dir / "summary.json"
        if done.is_file():
            summary = json.loads(done.read_text(encoding="utf-8"))
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
            summary = Controller(task, policy, code, run_dir, label_seed=seed, session=session).run()
        runs.append({"seed": seed, "per_window": [w["utility"] for w in summary["per_window"]],
                     "train_seconds": summary["cost"]["train_seconds"], "row_epochs": summary["train_row_epochs"],
                     "labels": summary["cost"]["labels"], "errors": {}})
    return _aggregate(runs)


def _errors(session: Session, y: np.ndarray, out: np.ndarray) -> dict:
    if len(y) == 0:
        return {}
    if session.task_type == "classification":
        counts = np.bincount(session.base.y.astype(np.int64), minlength=len(session.classes))
        predicted = out.argmax(axis=1)
        recall = {session.classes[c]: float(np.mean(predicted[y == c] == c)) for c in np.unique(y)}
        return {"class_imbalance": float(counts.max() / max(counts.min(), 1)), "per_class_recall": recall,
                "macro_f1": metrics.macro_f1(y, predicted)}
    error = out - y
    return {"over_estimate_ratio": float(np.mean(error > 0)), "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def _aggregate(runs: list[dict]) -> dict:
    scored = np.array([run["per_window"][1:] for run in runs])   # window 0 is always the official model
    return {"runs": runs, "S": float(scored.mean()), "C": float(np.mean([r["train_seconds"] for r in runs])),
            "W": float(np.mean([r["row_epochs"] for r in runs])),
            "labels": float(np.mean([r["labels"] for r in runs])), "scored": scored.tolist(),
            "errors": runs[0]["errors"] if runs else {}}


def paired(candidate: dict, incumbent: dict, z: float) -> dict:
    """Paired comparison over every (seed, window) cell.

    cost_change uses deterministic training work (rows × epochs); seconds_change is kept to catch
    a code change that makes each step much slower, but is too noisy at millisecond scale to rank by.
    """
    d = np.asarray(candidate["scored"]) - np.asarray(incumbent["scored"])
    n = d.size
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {"mean_diff": float(d.mean()), "band": z * sd / np.sqrt(max(n, 1)), "n": int(n),
            "worst_seed_diff": float(d.mean(axis=1).min()),
            "cost_change": (candidate["W"] - incumbent["W"]) / max(incumbent["W"], 1e-9),
            "seconds_change": (candidate["C"] - incumbent["C"]) / max(incumbent["C"], 1e-9),
            "candidate_seconds": candidate["C"],
            "label_change": (candidate["labels"] - incumbent["labels"]) / max(incumbent["labels"], 1e-9)}
