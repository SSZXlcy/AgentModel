"""Inner layer (模型训练层): one pass over the stream, window by window.

Per window:
  1. serve      the serving model predicts every row (the platform scores this)
  2. labels     request a budgeted share of labels; the platform may release some only later
  3. settle     fresh labels (nobody trained on them) judge the previous decision, the provisional
                or shadow candidate, and the static floor
  4. evidence   drift, performance, error structure
  5. decide     rule judgment of the drift; the strategy decides how to split, serve and train
  6. search     candidates differ in one dimension; 1 epoch each, then successive halving
  7. gate       paired bootstrap on the gate rows; pass → provisional (trial) or shadow
  8. record     decision now, settlement next window, health row for the supervisor

Rhythm (policy.gate_mode / train_trigger, set together by the mode.* operators):
  in_window      steps 6–7 as above: half the fresh labels train, the other half judges
  next_window    all fresh labels train ONE candidate, which serves on trial at once; the next window's
                 fresh labels confirm it or roll it back (step 3)
  on_completion  train only when an entity has finished: the finished entities (which nobody trained on)
                 judge a candidate trained on everything else; if it wins, refit including them and serve

The strategy (self.policy, self.code) may be replaced between windows by the supervisor; every
replacement is a recorded, reversible strategy version.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path

import numpy as np

from . import diagnose, gate, metrics, structure
from . import policy as rules
from .agent import Agent
from .codebase import CodeVersion
from .config import PolicyConfig, TaskConfig
from .contracts import Cost, DataRecipe, DriftJudgment, json_ready
from .data import load_table, write_lines
from .drift import DriftMeter
from .executor import Executor
from .ledger import BudgetError, Ledger
from .recorder import Recorder, write_json
from .stream import LocalStream
from .tree import Tree

ANCHOR_SEED = 20261110


class Session:
    """Everything derived from the task config and the official files, shared by runs and replays."""

    def __init__(self, task: TaskConfig, code: CodeVersion):
        self.task = task
        self.code = code
        self.profile = code.profile(task.profile)
        self.task_type = self.profile["task"]
        self.classes = self.profile.get("classes") or []
        self._structure: dict | None = None
        self.base = load_table(task.train_path, code, self.profile, task.group_key)
        # The entity key is configured, or discovered from the training set (task-agnostic).
        self.group_key = list(task.group_key)
        if not self.group_key:
            self.group_key = list(self.structure.get("entity") or [])
            if self.group_key:
                self.base = load_table(task.train_path, code, self.profile, self.group_key)
        self.stream_table = load_table(task.stream_path, code, self.profile, self.group_key)
        rng = np.random.default_rng(ANCHOR_SEED)
        order = rng.permutation(len(self.base))
        n_anchor = int(round(task.anchor_fraction * len(self.base)))
        self.anchor = self.base.take(np.sort(order[:n_anchor]))
        train_rows = np.sort(order[n_anchor:])
        cap = task.base_rows_cap or task.base_rows_budget
        if cap and len(train_rows) > cap:
            train_rows = np.sort(rng.choice(train_rows, cap, replace=False))
        self.base_rows = train_rows
        self.base_lines = [self.base.raw[i] for i in train_rows]
        report = json.loads((task.package / "training.json").read_text(encoding="utf-8"))
        self.seconds_per_row_epoch = report["training_seconds"] / (report["train_rows"] * report["final_epochs"])
        self.base_lr = float(self.profile["learning_rate"])

    @property
    def structure(self) -> dict:
        if self._structure is None:
            self._structure = structure.detect(self.base, self.task_type)
        return self._structure

    def new_stream(self) -> LocalStream:
        keys = None
        if self.task.label_protocol == "on_group_complete":
            fields = self.task.protocol_group_by or self.structure.get("entity")
            if not fields:
                raise ValueError("on_group_complete: no entity key configured or detected")
            keys = self.stream_table.key(fields)
        return LocalStream(self.stream_table, self.task, self.task_type, keys)

    @property
    def entity(self) -> list[str]:
        return list(self.structure.get("entity") or [])

    @staticmethod
    def label_positions(seed: int, window: int, n: int, fraction: float, sampling: str = "random") -> np.ndarray:
        """random: a seeded sample; spread: every (n / count)-th row from a seeded offset, so the labels
        cover the window evenly (neighbouring rows of one entity carry almost the same information)."""
        count = min(n, int(round(fraction * n)))
        rng = np.random.default_rng([seed, window, 1])
        if sampling == "spread" and count:
            offset = rng.random()
            return np.unique(np.minimum(((np.arange(count) + offset) * n / count).astype(np.int64), n - 1))
        return np.sort(rng.choice(n, size=count, replace=False))

    @staticmethod
    def split(policy: PolicyConfig, seed: int, window: int, metas: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Indices into the labelled references: (fit, gate). References are in time order.

        random    rows drawn at random (fine when rows are independent)
        by_group  whole entities go to gate, so a candidate is never judged on its own entity
        by_order  the latest rows go to gate, so a candidate predicts forward instead of filling gaps
        """
        count = len(metas)
        n_gate = max(1, int(round(policy.gate_fraction * count)))
        mode = policy.split_mode
        if mode == "by_group" and policy.split_group_by:
            keys = ["|".join(str(m.get(f)) for f in policy.split_group_by) for m in metas]
            unique = list(dict.fromkeys(keys))
            if len(unique) >= 2:
                order = np.random.default_rng([seed, window, 3]).permutation(len(unique))
                chosen = {unique[i] for i in order[:max(1, int(round(policy.gate_fraction * len(unique))))]}
                gate_idx = np.asarray([i for i, key in enumerate(keys) if key in chosen], dtype=np.int64)
                fit_idx = np.asarray([i for i, key in enumerate(keys) if key not in chosen], dtype=np.int64)
                if len(fit_idx):
                    return fit_idx, gate_idx
            mode = "by_order"
        if mode == "by_order":
            return np.arange(count - n_gate, dtype=np.int64), np.arange(count - n_gate, count, dtype=np.int64)
        order = np.random.default_rng([seed, window, 2]).permutation(count)
        return np.sort(order[n_gate:]), np.sort(order[:n_gate])

    def materialize(self, path: Path, stream: LocalStream, recipe: DataRecipe, pool: dict[int, np.ndarray],
                    fit_refs: list[tuple[int, int]], policy: PolicyConfig) -> tuple[Path, int, str]:
        """Base rows (optional) + labelled past windows + this decision's fit rows, then the
        per-group cap of the strategy (evenly spread in time inside each group)."""
        items: list[tuple[str, dict, object]] = []          # (line, meta, label)
        if recipe.include_base:
            items += [(self.base.raw[i], self.base.meta[i], self.base.y[i]) for i in self.base_rows]
        windows = sorted(set(recipe.past_windows) | {w for w, _ in fit_refs})
        for w in windows:
            positions = list(pool.get(w, [])) if w in recipe.past_windows else []
            positions += [p for v, p in fit_refs if v == w]
            positions = np.asarray(sorted(set(positions)), dtype=np.int64)
            if len(positions):
                for line, meta, label in zip(stream.labelled_lines(w, positions), stream.meta(w, positions),
                                             stream.labels(w, positions)):
                    items.append((line, meta, label))
        if policy.cap_per_group and policy.cap_group_by:
            items = _cap(items, policy.cap_per_group, policy.cap_group_by, self.classes)
        lines = [line for line, _, _ in items]
        write_lines(path, lines)
        return path, len(lines), "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cap(items: list, n: int, fields: list[str], classes: list[str]) -> list:
    groups: dict[str, list[int]] = {}
    for i, (_, meta, label) in enumerate(items):
        key = (classes[int(label)] if classes else str(label)) if fields == ["answer"] else \
            "|".join(str(meta.get(f)) for f in fields)
        groups.setdefault(key, []).append(i)
    keep: set[int] = set()
    for rows in groups.values():
        keep.update(rows if len(rows) <= n else [rows[j] for j in np.linspace(0, len(rows) - 1, n).astype(int)])
    return [items[i] for i in sorted(keep)]


class Controller:
    def __init__(self, task: TaskConfig, policy: PolicyConfig, code: CodeVersion, run_dir: Path, *,
                 agent: Agent | None = None, label_seed: int = 0, tracker=None, keep_data: bool = False,
                 session: Session | None = None, supervisor=None,
                 label_oracle: dict[int, np.ndarray] | None = None, windows: list[int] | None = None,
                 start_checkpoint: Path | None = None, score_on_labels: bool = False):
        self.task, self.policy, self.code = task, policy, code
        self.run_dir = Path(run_dir)
        self.agent = agent or Agent()
        self.seed = label_seed
        self.keep_data = keep_data
        self.session = session or Session(task, code)
        if label_oracle is not None:
            # A replay only uses labels the original run already holds, so they are available at once.
            self.stream = LocalStream(self.session.stream_table, dataclasses.replace(task, label_protocol="immediate"),
                                      self.session.task_type)
        else:
            self.stream = self.session.new_stream()
        self.recorder = Recorder(self.run_dir, tracker)
        self.tracker = tracker
        self.Model = code.modules().model.Model
        self.supervisor = supervisor
        # Replay mode (used by experiments): fixed labels, a window range, a start model,
        # and utility measured only on labels the original run had (never on hidden answers).
        self.label_oracle = label_oracle
        self.window_ids = windows if windows is not None else list(range(len(self.stream)))
        self.start_checkpoint = Path(start_checkpoint) if start_checkpoint else task.checkpoint
        self.score_on_labels = score_on_labels

    # ================================================================ lifecycle
    def run(self, resume: bool = False) -> dict:
        self._setup(resume)
        if self.supervisor is not None and not self.state.get("opened"):
            self._guarded("opening", lambda: self.supervisor.opening(self))
            self.state["opened"] = True
            self._save()
        for k in self.window_ids:
            if k < self.state["next_window"]:
                continue
            self._window(k)
            self.state["next_window"] = k + 1
            if self.supervisor is not None:
                self._guarded("supervisor", lambda: self.supervisor.after_window(self, k))
            self._save()
        if self.supervisor is not None:
            self._guarded("closing", lambda: self.supervisor.closing(self))
        return self._finish()

    def _guarded(self, stage: str, action) -> None:
        """The supervisor may fail; the run may not (threshold ①)."""
        try:
            action()
        except Exception as error:
            self.state["errors"] += 1
            self.recorder.event("error", stage=stage, error=f"{type(error).__name__}: {error}")

    def _setup(self, resume: bool) -> None:
        s, p = self.session, self.policy
        state_path = self.run_dir / "state.json"
        if resume and state_path.is_file():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            self.tree = Tree.load(self.run_dir)
            self.ledger = Ledger.from_dict(self.state["ledger"])
            self.policy = PolicyConfig(**self.state["policy"])
            self.stream.restore(self.state["released"], self.state["pending_labels"])
            self.recorder.event("resume", next_window=self.state["next_window"])
        else:
            self.tree = Tree(self.run_dir)
            root_meta = self.Model.load(self.start_checkpoint).meta
            self.tree.add_root(self.start_checkpoint, self.code.version_id, root_meta)
            self.ledger = Ledger(p.budget_train_seconds, p.budget_labels, p.budget_llm_tokens)
            anchor_out = self.tree.model("n0000", self.Model).predict(s.anchor.x)
            self.state = {"next_window": self.window_ids[0] if self.window_ids else 0, "pool": {},
                          "confirmations": 0, "below_static": 0, "cooldown": 0, "stall": 0,
                          "last_diagnosis": diagnose.NONE, "pending": None, "rows": [], "history": [],
                          "health": [], "promotions": 0, "rollbacks": 0, "resets": 0, "errors": 0,
                          "ref_utility": metrics.utility(s.task_type, s.anchor.y, anchor_out),
                          "policy": p.to_dict(), "strategy_version": 0, "code_version": self.code.version_id,
                          "labels_requested_total": 0, "labels_released_at_once": 0,
                          "entities_seen": {}, "completed": []}
            self.recorder.manifest(self._manifest())
        root = self.tree.model(self.tree.pointers["static"], self.Model)
        self.normalize = root.normalized
        reference = s.base.x
        if len(reference) > p.drift_reference_rows:
            reference = reference[np.random.default_rng(0).choice(len(reference), p.drift_reference_rows, replace=False)]
        self.meter = DriftMeter(self.normalize(reference), p.drift_quantile)
        self.executor = Executor(self.code, self.task.profile, self.tree, self.ledger, self.recorder,
                                 s.seconds_per_row_epoch)
        self.executor.row_epochs = self.state.get("row_epochs", 0)

    def _manifest(self) -> dict:
        def sha(path: Path) -> str:
            return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        info = self.task.package / "baseline-info.json"
        return {"run_dir": self.run_dir, "task": self.task.to_dict(), "policy": self.policy.to_dict(),
                "code_version": self.code.info(), "label_seed": self.seed,
                "official": json.loads(info.read_text(encoding="utf-8")) if info.is_file() else None,
                "data": {"train": sha(self.task.train_path), "stream": sha(self.task.stream_path),
                         "stream_split": self.task.stream, "windows": [len(w) for w in self.stream.windows],
                         "label_protocol": self.task.label_protocol,
                         "anchor_rows": len(self.session.anchor), "base_rows_in_recipes": len(self.session.base_lines)},
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "cpu_threads": os.environ.get("BASELINE_CPU_THREADS")},
                "agent": self.agent.status()["provider"], "replay": self.label_oracle is not None}

    def _save(self) -> None:
        """Checkpoint the run for resume. A file-system refusal costs the resume point, never the run."""
        try:
            self._write_state()
        except OSError as error:
            self.state["errors"] += 1
            self.recorder.event("error", stage="save", error=f"{type(error).__name__}: {error}")

    def _write_state(self) -> None:
        self.state["ledger"] = self.ledger.to_dict()
        self.state["row_epochs"] = self.executor.row_epochs
        self.state["released"], self.state["pending_labels"] = self.stream.snapshot()
        self.tree.save()
        write_json(self.run_dir / "state.json", self.state)

    # ================================================================ strategy versions
    def apply_strategy(self, *, policy: PolicyConfig | None = None, code: CodeVersion | None = None,
                       reason: str, source: str, round_index: int | None = None, evidence: dict | None = None) -> int:
        """Replace the active strategy from the next window on; returns the new version number."""
        st = self.state
        before = {"policy": st["policy"], "code_version": st["code_version"]}
        same_policy = policy is None or policy.to_dict() == st["policy"]
        same_code = code is None or code.version_id == st["code_version"]
        if same_policy and same_code:
            return st["strategy_version"]           # nothing would change: no empty version
        if policy is not None:
            self.policy = policy
            st["policy"] = policy.to_dict()
        if code is not None:
            self.code = code
            self.executor.code = code
            st["code_version"] = code.version_id
        st["strategy_version"] += 1
        version = st["strategy_version"]
        write_json(self.run_dir / "strategies" / f"s{version:03d}.json",
                   {"version": version, "window": st["next_window"], "policy": st["policy"],
                    "code_version": st["code_version"], "reason": reason, "source": source, "round": round_index})
        changed = {k: v for k, v in st["policy"].items() if before["policy"].get(k) != v}
        self.recorder.strategy_update(version=version, effective_from_window=st["next_window"], source=source,
                                      round=round_index, reason=reason, policy_changes=changed,
                                      code_version_before=before["code_version"], code_version_after=st["code_version"],
                                      evidence=evidence or {})
        return version

    def refit_opening(self, reason: str) -> str | None:
        """Before the stream: serve the procedure an opening group k-fold has just validated, i.e. the
        official configuration from initialisation on the official set under the new strategy (data cap,
        training code). The k folds held out whole entities, which is stronger evidence than any in-stream
        gate; the static floor and the supervisor's loss trigger still guard it."""
        s, p = self.session, self.policy
        items = [(s.base.raw[i], s.base.meta[i], s.base.y[i]) for i in s.base_rows]
        if p.cap_per_group and p.cap_group_by:
            items = _cap(items, p.cap_per_group, p.cap_group_by, s.classes)
        path = write_lines(self.run_dir / "data" / "opening_refit.jsonl", [line for line, _, _ in items])
        try:
            node = self.executor.train_from_scratch(path, self._anchor_file(), len(items),
                                                    learning_rate=s.base_lr * p.lr_scale, window=-1,
                                                    recipe_id="opening", spec="opening-refit",
                                                    epochs=int(s.profile.get("epochs", 20)))
        except BudgetError as error:
            self.recorder.event("budget_stop", window=-1, error=str(error))
            return None
        finally:
            path.unlink(missing_ok=True)
        self.tree.pointers.update(confirmed=node.node_id, provisional=None, shadow=None)
        self.recorder.event("opening_refit", node=node.node_id, rows=len(items), reason=reason)
        return node.node_id

    # ================================================================ references to labelled rows
    def _arrays(self, refs: list[tuple[int, int]]):
        """Features, labels, metas and bootstrap groups for (window, position) references."""
        if not refs:
            empty = np.zeros((0, self.session.stream_table.x.shape[1]))
            return empty, np.array([]), [], np.array([], dtype=str)
        xs, ys, metas = [], [], []
        by_window: dict[int, list[int]] = {}
        for w, p in refs:
            by_window.setdefault(w, []).append(p)
        for w in sorted(by_window):
            positions = np.asarray(by_window[w], dtype=np.int64)
            xs.append(self.stream.features(w)[1][positions])
            ys.append(self.stream.labels(w, positions))
            metas += self.stream.meta(w, positions)
        fields = self.policy.split_group_by
        if fields:
            groups = np.asarray(["|".join(str(m.get(f)) for f in fields) for m in metas])
        else:
            groups = np.concatenate([self.stream.features(w)[2][np.asarray(by_window[w], dtype=np.int64)]
                                     for w in sorted(by_window)])
        return np.vstack(xs), np.concatenate(ys), metas, groups

    def _entity_key(self, meta: dict) -> str:
        return "|".join(str(meta.get(f)) for f in self.session.entity)

    def _completions(self, k: int, delivered: dict[int, np.ndarray]) -> list[str]:
        """Entities that finished by window k, seen from the controller's side only: an entity that was
        in an earlier window and is absent now has left the stream; an entity whose earlier labels are
        delivered only now has just finished (late labels are released when the countdown resolves)."""
        if not self.session.entity:
            return []
        st = self.state
        seen, done = st.setdefault("entities_seen", {}), set(st.setdefault("completed", []))
        now = {self._entity_key(m) for m in self.stream.meta(k)}
        finished = {e for e in seen if e not in now and e not in done}
        for w, positions in delivered.items():
            finished |= {self._entity_key(m) for m in self.stream.meta(w, positions)} - done
        for e in now:
            seen[e] = k
        st["completed"] = sorted(done | finished)
        return sorted(finished)

    # ================================================================ one window
    def _predict(self, node_id: str, x: np.ndarray) -> np.ndarray:
        return self.tree.model(node_id, self.Model).predict(x)

    def _window(self, k: int) -> None:
        s, p, st = self.session, self.policy, self.state
        _, x, _ = self.stream.features(k)
        serving = self.tree.serving
        static_id = self.tree.pointers["static"]

        # labels: earlier requests the platform releases now, then this window's request
        delivered = self.stream.deliver(k)
        if self.label_oracle is not None:
            requested = np.asarray(self.label_oracle.get(k, []), dtype=np.int64)
        else:
            count = int(round(p.label_fraction * len(x)))
            requested = s.label_positions(self.seed, k, len(x), p.label_fraction, p.label_sampling) \
                if count > 0 and self.ledger.affordable(Cost(labels=count)) else np.array([], dtype=np.int64)
        if len(requested):
            self.ledger.charge(Cost(labels=len(requested)))
        released, _ = self.stream.request_labels(k, requested)
        fresh = [(w, int(q)) for w in sorted(delivered) for q in delivered[w]] + [(k, int(q)) for q in released]
        st["labels_requested_total"] = st.get("labels_requested_total", 0) + int(len(requested))
        st["labels_released_at_once"] = st.get("labels_released_at_once", 0) + int(len(released))
        completed = self._completions(k, delivered)

        if self.score_on_labels:
            ref = [(k, int(q)) for q in released]
            fx, fy, _, _ = self._arrays(ref)
            served = metrics.report(s.task_type, fy, self._predict(serving, fx)) if len(fy) else None
            static = metrics.report(s.task_type, fy, self._predict(static_id, fx)) if len(fy) else None
        else:
            served = self.stream.score(k, self._predict(serving, x))
            static = self.stream.score(k, self._predict(static_id, x))

        health = {"window": k, "fresh": len(fresh), "delivered_late": sum(len(v) for v in delivered.values()),
                  "requested": int(len(requested)), "released_at_once": int(len(released)), "completed": completed}
        decision: dict = {"action": "observe"}
        try:
            self._settle(k, fresh, health)
            decision = self._decide(k, x, fresh, health, completed)
        except Exception as error:   # threshold ①: a window may fail, the run may not
            st["errors"] += 1
            self.recorder.event("error", window=k, error=f"{type(error).__name__}: {error}")
            decision = {"action": "observe", "error": f"{type(error).__name__}: {error}"}

        for w, q in fresh:
            st["pool"].setdefault(str(w), [])
            if q not in st["pool"][str(w)]:
                st["pool"][str(w)].append(q)
        for w in st["pool"]:
            st["pool"][w].sort()
        if not self.keep_data:
            shutil.rmtree(self.run_dir / "data" / f"w{k:03d}", ignore_errors=True)
        st["health"].append(health)
        headline = "macro_f1" if s.task_type == "classification" else "rmse"
        row = {"window": k, "rows": int(len(x)), "served_node": serving,
               "utility": served["utility"] if served else None, "static_utility": static["utility"] if static else None,
               headline: served[headline] if served else None, f"static_{headline}": static[headline] if static else None,
               "labels_requested": int(len(requested)), "labels_fresh": len(fresh),
               "action": decision["action"], "promoted": bool(decision.get("promoted")),
               "serving_after": self.tree.serving, "strategy_version": st["strategy_version"]}
        st["rows"].append(row)
        self.recorder.metrics({f"window/{key}": value for key, value in row.items() if isinstance(value, (int, float))})

    def _settle(self, k: int, fresh: list[tuple[int, int]], health: dict) -> None:
        st, p, task_type = self.state, self.policy, self.session.task_type
        st["cooldown"] = max(0, st["cooldown"] - 1)
        if len(fresh) < 4:
            return
        lx, ly, _, lg = self._arrays(fresh)
        pending = st["pending"]
        verdict = None
        shadow = self.tree.pointers.get("shadow")
        if shadow:
            diff, sd = metrics.paired_difference(task_type, ly, self._predict(shadow, lx),
                                                 self._predict(self.tree.serving, lx), lg, p.bootstrap_reps, seed=k)
            if diff > p.z * sd:          # a shadow serves only when it is clearly better on fresh rows
                self.tree.adopt_shadow()
                verdict = "shadow_adopted"
            else:
                self.tree.set_shadow(None)
                verdict = "shadow_dropped"
            health["shadow"] = {"diff": diff, "band": p.z * sd, "adopted": verdict == "shadow_adopted"}
            self.recorder.event("shadow", window=k, node=shadow, outcome=verdict, diff=diff, band=p.z * sd)
        provisional, confirmed = self.tree.pointers["provisional"], self.tree.pointers["confirmed"]
        if provisional:
            outcome, diff, band = gate.confirmation(task_type, ly, self._predict(provisional, lx),
                                                    self._predict(confirmed, lx), lg, p, seed=k)
            # small losses that no single window can show still add up: judge the running sum too
            track = st.setdefault("provisional_track", [])
            track.append([diff, band])
            if gate.cumulative_worse(track):
                outcome = "rollback"
            if outcome == "rollback":
                self.tree.rollback()
                st.update(rollbacks=st["rollbacks"] + 1, cooldown=p.cooldown_windows, confirmations=0)
                verdict = "rolled_back"
                health["rollback"] = True
            else:
                st["confirmations"] += 1
                verdict = "holding"
                if st["confirmations"] >= p.confirm_windows:
                    self.tree.confirm()
                    st["confirmations"] = 0
                    verdict = "confirmed"
            self.recorder.event("confirmation", window=k, node=provisional, outcome=verdict, diff=diff, band=band)
        if pending:
            before = self._predict(pending["before"], lx)
            after = self._predict(pending["after"], lx)
            gain = metrics.utility(task_type, ly, after) - metrics.utility(task_type, ly, before)
            if pending["before"] == pending["after"] and verdict is None:
                verdict = "no_change"
            self.recorder.settlement(decision_seq=pending["seq"], node_id=pending["after"],
                                     decided_at_window=pending["window"], settled_at_window=k,
                                     reward={"utility_gain_next_window": gain,
                                             "gain_per_cost": gain / max(pending["cost_value"], 1e-9)},
                                     verdict=verdict)
            health["settled"] = {"window": pending["window"], "gain": gain, "predicted": pending.get("predicted_gain"),
                                 "changed": pending["before"] != pending["after"]}
            for item in st["history"]:
                if item["window"] == pending["window"]:
                    item["gain_next_window"] = gain
            st["pending"] = None
        serving, static = self.tree.serving, self.tree.pointers["static"]
        diff, band, worse = 0.0, 0.0, False
        floor_track = st.setdefault("floor_track", [])
        if serving != static:
            worse, diff, band = gate.below_static(task_type, ly, self._predict(serving, lx),
                                                  self._predict(static, lx), lg, p, seed=k)
            st["below_static"] = st["below_static"] + 1 if worse else 0
            floor_track.append([diff, band])
            recent = floor_track[-p.static_floor_windows:]
            breach = st["below_static"] >= p.static_floor_windows or (
                len(recent) >= p.static_floor_windows and gate.cumulative_worse(recent))
            if breach:
                self._fall_back(k, ly, lx, lg, diff, band)
        else:
            floor_track.clear()
        health["vs_static"] = {"diff": diff, "band": band, "worse": bool(worse)}

    def _fall_back(self, k: int, ly, lx, lg, diff: float, band: float) -> None:
        """Below the static model (threshold ③): return to the model the opening validated if it still holds
        up against static on these fresh rows, else to static itself."""
        st, p, task_type = self.state, self.policy, self.session.task_type
        static, floor = self.tree.pointers["static"], st.get("floor_node")
        target = static
        if floor and floor != static and floor != self.tree.serving and floor in self.tree.nodes:
            floor_worse, _, _ = gate.below_static(task_type, ly, self._predict(floor, lx), self._predict(static, lx),
                                                  lg, p, seed=k)
            if not floor_worse:
                target = floor
        self.tree.pointers.update(confirmed=target, provisional=None, shadow=None)
        if target == static:
            st["floor_node"] = static
        st.update(resets=st["resets"] + 1, cooldown=p.cooldown_windows, below_static=0, confirmations=0,
                  floor_track=[], provisional_track=[])
        self.recorder.event("reset_to_static" if target == static else "reset_to_opening", window=k, diff=diff,
                            band=band, target=target)

    def _evidence(self, k: int, x, fresh) -> dict:
        s, p, st = self.session, self.policy, self.state
        previous = self.normalize(self.stream.features(k - 1)[1]) if k > 0 and (k - 1) in self.window_ids else None
        drift = self.meter.measure(self.normalize(x), previous)
        lx, labels, _, lg = self._arrays(fresh)
        out = self._predict(self.tree.serving, lx)
        now = metrics.utility(s.task_type, labels, out)
        noise = p.z * metrics.utility_sd(s.task_type, labels, out, lg, min(p.bootstrap_reps, 200), seed=k)
        errors: dict = {}
        if s.task_type == "classification":
            predicted = out.argmax(axis=1)
            counts = np.bincount(labels.astype(np.int64), minlength=len(s.classes))
            present = counts[counts > 0]
            errors["class_imbalance"] = float(present.max() / present.min()) if len(present) else 1.0
            errors["absent_classes"] = [s.classes[i] for i in range(len(s.classes)) if counts[i] == 0]
            confusion = [(s.classes[t], s.classes[q], float(np.mean(predicted[labels == t] == q)))
                         for t in range(len(s.classes)) for q in range(len(s.classes)) if t != q and counts[t] > 0]
            errors["top_confusions"] = sorted(confusion, key=lambda c: -c[2])[:3]
            prior = st.get("last_class_counts")
            if prior:
                prior = np.asarray(prior, dtype=float)
                errors["label_shift_tv"] = float(0.5 * np.abs(counts / counts.sum() - prior / prior.sum()).sum())
            st["last_class_counts"] = counts.tolist()
        else:
            error = out - labels
            errors.update(over_estimate_ratio=float(np.mean(error > 0)), bias=float(np.mean(error)),
                          rmse=float(np.sqrt(np.mean(error ** 2))))
        return {"window": k, "task_type": s.task_type, "n_labelled": int(len(labels)), "drift": drift,
                "performance": {"utility_now": now, "utility_reference": st["ref_utility"], "noise_band": noise,
                                "degraded": bool(now < st["ref_utility"] - noise)},
                "errors": errors, "history": st["history"][-3:],
                "labelled_pool": {w: len(v) for w, v in st["pool"].items()},
                "last_diagnosis": st["last_diagnosis"], "budget": self.ledger.remaining()}

    def _mask(self, labelled: int) -> list[str]:
        mask = ["observe"]
        cheapest = self.executor.estimate(max(labelled, 1), 1)
        if (labelled >= 8 and self.state["cooldown"] == 0 and self.policy.serve_mode != "freeze"
                and self.ledger.affordable(cheapest)):
            mask.append("train")
        if self.tree.pointers["provisional"]:
            mask.append("rollback")
        return mask

    def _decide(self, k: int, x, fresh, health: dict, completed: list[str] | None = None) -> dict:
        st, p = self.state, self.policy
        spent_before = self.ledger.spent
        work_before = self.executor.row_epochs
        serving_before = self.tree.serving
        event = self._event_refs(completed or [], fresh) if p.train_trigger == "on_completion" else None
        mask = self._mask(len(event[0]) if event is not None else len(fresh))
        if len(fresh) < 4:
            evidence, judgment = {"window": k, "n_labelled": len(fresh)}, DriftJudgment(
                "none", "adaptive", 0.0, "too few released labels to judge", "rule")
        else:
            evidence = self._evidence(k, x, fresh)
            judgment = rules.rule_judgment(evidence)
            if p.recent_windows_only:
                judgment = DriftJudgment(judgment.drift_type, f"recent:{p.recent_windows_only}", judgment.confidence,
                                         judgment.rationale + " (strategy: recent windows only)", judgment.source)
            health["drift"] = {"mmd": evidence["drift"]["mmd_vs_recent"], "threshold": evidence["drift"]["threshold"],
                               "shifted": evidence["drift"]["shifted"], "degraded": evidence["performance"]["degraded"]}
        result: dict = {}
        if "train" not in mask:
            action, reason = "observe", ("serving is frozen by the strategy" if p.serve_mode == "freeze"
                                         else "waiting for an entity to finish" if event is not None and not completed
                                         else "training not allowed (labels, cooldown or budget)")
        elif event is not None:
            action, reason = "train", f"entities finished: {len(completed)}; judge on them, train on the rest"
            result = self._event(k, event[0], event[1], completed)
        elif p.gate_mode == "next_window":
            action, reason = "train", "next_window: every fresh label trains one candidate, the next window judges it"
            result = self._direct(k, fresh)
        elif judgment.drift_type == "none":
            action, reason = "observe", "no drift"
        else:
            action, reason = "train", judgment.rationale
            result = self._search(k, fresh, judgment)
        if "utility_now" in evidence.get("performance", {}):
            st["ref_utility"] = result.get("gate_utility", evidence["performance"]["utility_now"])
        cost = self.ledger.spent - spent_before
        accepted = bool(result.get("gate", {}).get("accepted"))
        health.update(trained=action == "train", row_epochs=self.executor.row_epochs - work_before,
                      gate_gain=result.get("gate", {}).get("gain") if accepted else None, promoted=accepted,
                      stall=st["stall"], diagnosis=result.get("diagnosis"))
        row = self.recorder.decision(
            window=k, node_id=result.get("best"), code_version=self.code.version_id, policy_id=p.policy_id,
            strategy_version=st["strategy_version"], state=evidence, judgment=judgment.to_dict(),
            action={"kind": action, "reason": reason, "serve_mode": p.serve_mode, "split_mode": p.split_mode,
                    "mode": p.mode, "gate_mode": p.gate_mode, "train_trigger": p.train_trigger,
                    "candidates": result.get("candidates", [])},
            action_mask=mask, cost=cost.to_dict(), gate=result.get("gate"), diagnosis=result.get("diagnosis"),
            serving_before=serving_before, serving_after=self.tree.serving, shadow=self.tree.pointers.get("shadow"))
        st["pending"] = {"seq": row["seq"], "window": k, "before": serving_before, "after": self.tree.serving,
                         "cost_value": rules.cost_value(cost, p),
                         "predicted_gain": result.get("gate", {}).get("gain") if accepted else None}
        st["history"].append({"window": k, "drift_type": judgment.drift_type, "recipe": judgment.recipe,
                              "action": action, "promoted": accepted})
        return {"action": action, "promoted": accepted and p.serve_mode == "trial"}

    # ---------------------------------------------------------------- rhythm: next_window / on_completion
    def _anchor_file(self) -> Path:
        """Validation file the official trainer requires; its IDs never overlap a training file."""
        path = self.run_dir / "data" / "anchor.jsonl"
        if not path.is_file():
            write_lines(path, self.session.anchor.raw)
        return path

    def _fit_one(self, k: int, recipe: DataRecipe, pool: dict, refs: list[tuple[int, int]], name: str):
        s, p = self.session, self.policy
        path, rows, sha = s.materialize(self.run_dir / "data" / f"w{k:03d}" / f"{name}.jsonl", self.stream, recipe,
                                        pool, refs, p)
        if rows == 0:
            return None, 0
        self.recorder.event("recipe", window=k, recipe=recipe.to_dict(), rows=rows, sha256=sha, purpose=name,
                            cap_per_group=p.cap_per_group, cap_group_by=p.cap_group_by)
        parent = self.tree.pointers["static"] if p.branch_from == "static" else self.tree.serving
        node = self.executor.train(self.tree.nodes[parent], path, self._anchor_file(), rows, epochs=p.direct_epochs,
                                   learning_rate=s.base_lr * p.lr_scale, window=k, recipe_id=recipe.recipe_id, spec=name)
        return node, rows

    def _serve(self, node_id: str) -> None:
        if self.policy.serve_mode == "shadow":
            self.tree.set_shadow(node_id)
        else:
            self.tree.promote_provisional(node_id)
            self.state["confirmations"] = 0
            self.state["provisional_track"] = []
        self.state["promotions"] += 1

    def _forgets(self, node_id: str, serving: str) -> tuple[bool, float]:
        """Same anchor veto as the promotion gate: (vetoed, utility drop on the held-out official rows)."""
        s = self.session
        before = metrics.utility(s.task_type, s.anchor.y, self._predict(serving, s.anchor.x))
        drop = before - metrics.utility(s.task_type, s.anchor.y, self._predict(node_id, s.anchor.x))
        return drop > self.policy.anchor_veto * max(abs(before), 1e-9), drop

    def _direct(self, k: int, fresh: list[tuple[int, int]]) -> dict:
        """next_window: one candidate on every label held (official set + history + this window), on trial
        at once. The previous candidate reaching this point has held its window, so it is confirmed first."""
        s, p, st = self.session, self.policy, self.state
        pool = {int(w): np.asarray(v, dtype=np.int64) for w, v in st["pool"].items()}
        choice = f"recent:{p.recent_windows_only}" if p.recent_windows_only else "all"
        recipe = rules.resolve_recipe(choice, k, {w: len(v) for w, v in pool.items()}, len(fresh), p)
        if self.tree.pointers["provisional"]:
            self.tree.confirm()
            st["confirmations"] = 0
        serving = self.tree.serving
        try:
            node, rows = self._fit_one(k, recipe, pool, fresh, "direct")
        except BudgetError as error:
            self.recorder.event("budget_stop", window=k, error=str(error))
            return {"best": None, "diagnosis": diagnose.NONE, "candidates": []}
        if node is None:
            return {"best": None, "diagnosis": diagnose.NONE, "candidates": []}
        vetoed, drop = self._forgets(node.node_id, serving)
        ok = not vetoed
        if ok:
            self._serve(node.node_id)
        st["last_diagnosis"] = diagnose.NONE
        st["stall"] = 0 if ok else st["stall"] + 1
        verdict = {"accepted": ok, "gain": None, "band": None, "anchor_drop": drop,
                   "reason": "on trial; the next window confirms or rolls back" if ok
                   else f"anchor utility dropped by {drop:.4f} (catastrophic forgetting)"}
        return {"best": node.node_id, "gate": verdict, "diagnosis": diagnose.NONE,
                "candidates": [{"recipe": recipe.to_dict(), "rows": rows, "epochs": p.direct_epochs}],
                "split": {"mode": "next_window", "fit": len(fresh), "gate": 0}}

    def _event_refs(self, completed: list[str], fresh: list[tuple[int, int]]):
        """(rows of the entities that just finished, every other labelled row)."""
        if not completed:
            return [], []
        done = set(completed)
        refs = sorted({(int(w), int(q)) for w, v in self.state["pool"].items() for q in v} | set(fresh))
        judge, rest = [], []
        for w, q in refs:
            (judge if self._entity_key(self.stream.meta(w, [q])[0]) in done else rest).append((w, q))
        return judge, rest

    def _event(self, k: int, judge: list[tuple[int, int]], rest: list[tuple[int, int]], completed: list[str]) -> dict:
        """on_completion: the finished entities were never trained on, so they judge the procedure fairly.
        The decision rests on whole entities by construction, so the minimum-unit rule is replaced by the
        small-sample widening of the band (block bootstrap inside the few entities)."""
        s, p, st = self.session, self.policy, self.state
        seconds_before = self.ledger.spent.train_seconds
        gate_x, gate_y, metas, _ = self._arrays(judge)
        groups = np.asarray([self._entity_key(m) for m in metas])
        serving = self.tree.serving
        recipe = DataRecipe(True, (), "event")
        empty = {"best": None, "diagnosis": diagnose.NONE, "candidates": []}
        try:
            check, rows = self._fit_one(k, recipe, {}, rest, "event-check")
            if check is None:
                return empty
            result = gate.promotion_gate(
                s.task_type, gate_y, self._predict(check.node_id, gate_x), self._predict(serving, gate_x), groups,
                rules.cost_value(Cost(train_seconds=self.ledger.spent.train_seconds - seconds_before), p),
                p.replace(min_gate_units=1),
                anchor=(s.anchor.y, self._predict(check.node_id, s.anchor.x), self._predict(serving, s.anchor.x)), seed=k)
            best = check.node_id
            if result.accepted:
                final, _ = self._fit_one(k, recipe, {}, rest + judge, "event-refit")
                best = final.node_id
                self._serve(best)
        except BudgetError as error:
            self.recorder.event("budget_stop", window=k, error=str(error))
            return empty
        st["last_diagnosis"] = diagnose.NONE
        st["stall"] = 0 if result.accepted else st["stall"] + 1
        self.recorder.event("completion", window=k, entities=completed, judge_rows=len(judge), train_rows=rows,
                            accepted=result.accepted, gain=result.gain, band=result.band)
        return {"best": best, "gate": result.to_dict(), "diagnosis": diagnose.NONE,
                "candidates": [{"recipe": recipe.to_dict(), "rows": rows, "epochs": p.direct_epochs}],
                "split": {"mode": "on_completion", "fit": len(rest), "gate": len(judge), "entities": completed}}

    def _search(self, k: int, fresh: list[tuple[int, int]], judgment: DriftJudgment) -> dict:
        s, p, st = self.session, self.policy, self.state
        seconds_before = self.ledger.spent.train_seconds
        metas = [self.stream.meta(w, [q])[0] for w, q in fresh]
        fit_idx, gate_idx = s.split(p, self.seed, k, metas)
        fit_refs = [fresh[i] for i in fit_idx]
        gate_refs = [fresh[i] for i in gate_idx]
        gate_x, gate_y, _, gate_g = self._arrays(gate_refs)
        window_dir = self.run_dir / "data" / f"w{k:03d}"
        gate_lines = []
        for w in sorted({w for w, _ in gate_refs}):
            gate_lines += self.stream.labelled_lines(w, np.asarray([q for v, q in gate_refs if v == w], dtype=np.int64))
        val_path = write_lines(window_dir / "gate.jsonl", gate_lines)
        pool = {int(w): np.asarray(v, dtype=np.int64) for w, v in st["pool"].items()}
        specs = rules.candidates(judgment, st["last_diagnosis"], k, {w: len(v) for w, v in pool.items()},
                                 len(fit_refs), p)
        serving = self.tree.serving
        serving_gate = self._predict(serving, gate_x)
        serving_u = metrics.utility(s.task_type, gate_y, serving_gate)
        diag_band = p.z * metrics.utility_sd(s.task_type, gate_y, serving_gate, gate_g, min(p.bootstrap_reps, 200), seed=k)

        def step(chain: dict, parent_id: str) -> None:
            spec = chain["spec"]
            node = self.executor.train(self.tree.nodes[parent_id], chain["path"], val_path, chain["rows"], epochs=1,
                                       learning_rate=s.base_lr * spec.lr_scale * p.lr_scale, window=k,
                                       recipe_id=spec.recipe.recipe_id, spec=spec.label)
            u = metrics.utility(s.task_type, gate_y, self._predict(node.node_id, gate_x))
            chain["points"].append({"node": node.node_id, "epoch": node.epochs, "loss": node.history[-1]["loss"],
                                    "utility": u})

        chains: list[dict] = []
        try:
            for spec in specs:
                path, rows, sha = s.materialize(window_dir / f"{spec.recipe.recipe_id}.jsonl", self.stream,
                                                spec.recipe, pool, fit_refs, p)
                if rows == 0:
                    continue
                self.recorder.event("recipe", window=k, recipe=spec.recipe.to_dict(), rows=rows, sha256=sha,
                                    cap_per_group=p.cap_per_group, cap_group_by=p.cap_group_by)
                chain = {"spec": spec, "path": path, "rows": rows, "points": []}
                step(chain, self.tree.pointers["static"] if p.branch_from == "static" else serving)
                chains.append(chain)
            survivors = sorted(chains, key=lambda c: -c["points"][-1]["utility"])[:math.ceil(len(chains) / 2)]
            for chain in survivors:
                stale = 0
                while len(chain["points"]) < chain["spec"].max_epochs and stale < 2:
                    best_so_far = max(point["utility"] for point in chain["points"])
                    step(chain, chain["points"][-1]["node"])
                    stale = stale + 1 if chain["points"][-1]["utility"] <= best_so_far else 0
        except BudgetError as error:
            self.recorder.event("budget_stop", window=k, error=str(error))

        summary = [{"spec": c["spec"].to_dict(), "rows": c["rows"], "points": c["points"]} for c in chains]
        if not chains:
            return {"best": None, "diagnosis": diagnose.NONE, "candidates": summary}
        best_chain = max(chains, key=lambda c: max(point["utility"] for point in c["points"]))
        best_point = max(best_chain["points"], key=lambda point: point["utility"])
        best = best_point["node"]
        diagnosis = diagnose.diagnose_chain(best_chain["points"], serving_u, diag_band)
        window_cost = Cost(train_seconds=self.ledger.spent.train_seconds - seconds_before)
        anchor = (s.anchor.y, self._predict(best, s.anchor.x), self._predict(serving, s.anchor.x))
        result = gate.promotion_gate(s.task_type, gate_y, self._predict(best, gate_x), serving_gate, gate_g,
                                     rules.cost_value(window_cost, p), p, anchor=anchor, seed=k)
        if result.accepted:
            self._serve(best)
        st["last_diagnosis"] = diagnosis
        st["stall"] = 0 if result.accepted else st["stall"] + 1
        out = {"best": best, "gate": result.to_dict(), "diagnosis": diagnosis, "candidates": summary,
               "split": {"mode": p.split_mode, "fit": len(fit_refs), "gate": len(gate_refs)}}
        if result.accepted:
            out["gate_utility"] = best_point["utility"]
        return out

    # ================================================================ end of run
    def _finish(self) -> dict:
        st = self.state
        rows = st["rows"]
        scored = [r for r in rows if r["utility"] is not None]
        utility = [r["utility"] for r in scored] or [float("nan")]
        static = [r["static_utility"] for r in scored] or [float("nan")]
        summary = {
            "task": self.task.task, "run_dir": self.run_dir.as_posix(), "code_version": self.code.version_id,
            "policy_id": self.policy.policy_id, "label_seed": self.seed, "label_protocol": self.task.label_protocol,
            "windows": len(rows), "scored_windows": len(scored),
            "mean_utility": float(np.mean(utility)), "static_mean_utility": float(np.mean(static)),
            "gain_vs_static": float(np.mean(utility) - np.mean(static)),
            "worst_utility": float(np.min(utility)), "static_worst_utility": float(np.min(static)),
            "final_utility": utility[-1], "promotions": st["promotions"], "rollbacks": st["rollbacks"],
            "resets_to_static": st["resets"], "errors": st["errors"], "cost": self.ledger.spent.to_dict(),
            "train_row_epochs": self.executor.row_epochs, "strategy_versions": st["strategy_version"],
            "final_policy": st["policy"], "pointers": self.tree.pointers, "agent": self.agent.status(),
            "supervisor": st.get("supervisor_summary"), "per_window": rows}
        self.recorder.summary(json_ready(summary))
        if self.tracker is not None:
            try:
                self.tracker.finish(summary)
            except Exception as error:
                self.recorder.event("tracking_error", error=f"{type(error).__name__}: {error}")
        return summary
