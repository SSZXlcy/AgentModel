"""Generic analysis tools the agent calls by name (W1). None of them knows a task or a field name;
they only understand the official JSONL shape (inputs / answer / meta).

Scopes an agent may read:
  base         the official training set (with answers)
  stream       stream rows seen so far (features and meta; answers only where released)
  stream_paid  stream rows whose labels have been released
  anchor       the held-out slice of the official training set
Answers that were not released, future windows and sealed data are unreachable by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import diagnose, experiments, metrics, operators, signature, structure
from .config import PolicyConfig
from .contracts import Cost
from .drift import mmd_rbf, psi
from .stream import DataAccessError

RESULT_LIMIT = 4000

CATALOGUE = {
    "schema": "{} -> input dimension, answer type, meta fields with type and number of distinct values",
    "field_profile": "{field, scope='base'|'stream'} -> distribution of one meta field",
    "find_structure": "{scope='base'} -> best entity key / order field / whether the answer is a countdown",
    "sequence_check": "{group_by:[fields], order_by:field|null, scope='base'} -> inside each group: is the order "
                      "consecutive, is the answer decreasing / a countdown ending at 0",
    "group_stats": "{group_by:[fields]|['answer'], scope='base'|'stream_paid'} -> group sizes, largest share, imbalance",
    "error_breakdown": "{by:field|'answer_bins'|'class', scope='anchor'|'stream_paid', model='serving'|'static'} -> "
                       "error of a model per bucket",
    "training_curve": "{node:'official'|node_id} -> per-epoch training loss and validation metric",
    "compare_subsets": "{a:'base'|'stream_paid'|'window:K', b:...} -> input shift (MMD, PSI) and serving-model "
                       "utility on each side",
    "run_status": "{} -> windows done, recent health rows, current strategy, budget",
    "signature": "{} -> the situation signature (task-agnostic features; operator defaults are computed from it)",
    "experiment": "{operator, params, validation:{group_by, folds, windows, seeds}} -> validates one operator call "
                  "with the experiment the manual prescribes; returns the paired difference vs the current strategy",
    "read_memory": "{last:int} -> journal of earlier rounds and the experience library",
}


class ToolError(ValueError):
    """Bad tool name or arguments; returned to the agent as an error message."""


class Toolbox:
    def __init__(self, ctrl, memory, scratch: Path, code_store: Path):
        self.ctrl = ctrl
        self.memory = memory
        self.scratch = Path(scratch)
        self.code_store = Path(code_store)
        self.calls: list[dict] = []
        self.experiments: list[dict] = []

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | None) -> dict:
        args = dict(args or {})
        try:
            if name not in CATALOGUE:
                raise ToolError(f"unknown tool {name!r}; available: {sorted(CATALOGUE)}")
            result = getattr(self, "_" + name)(**args)
        except ToolError as error:
            result = {"error": str(error)}
        except TypeError as error:
            result = {"error": f"bad arguments for {name}: {error}"}
        except DataAccessError as error:
            result = {"error": f"not allowed: {error}"}
        except OSError as error:          # file-system trouble is not a data-access refusal
            result = {"error": f"file system error, try again or skip: {type(error).__name__}: {error}"}
        text = json.dumps(result, ensure_ascii=False, default=_plain)
        if len(text) > RESULT_LIMIT:
            result = {"partial_result": text[:RESULT_LIMIT],
                      "note": "truncated by the harness; ask a narrower question for the rest"}
        self.calls.append({"tool": name, "args": args, "result_chars": len(text)})
        return result

    # ------------------------------------------------------------ data scopes
    def _table(self, scope: str):
        s = self.ctrl.session
        if scope == "base":
            return s.base
        if scope == "anchor":
            return s.anchor
        raise ToolError(f"scope {scope!r} is not a table scope")

    def _stream_rows(self, paid_only: bool):
        """(window, position) references of stream rows seen so far."""
        st, stream = self.ctrl.state, self.ctrl.stream
        now = st["next_window"]
        refs = []
        for w in [w for w in self.ctrl.window_ids if w < now]:
            if paid_only:
                refs += [(w, int(q)) for q in st["pool"].get(str(w), [])]
            else:
                refs += [(w, q) for q in range(len(stream.windows[w]))]
        return refs

    def _stream_meta(self, paid_only: bool) -> list[dict]:
        return [self.ctrl.stream.meta(w, [q])[0] for w, q in self._stream_rows(paid_only)]

    # ------------------------------------------------------------ tools
    def _schema(self):
        s = self.ctrl.session
        fields = structure._fields(s.base)
        return {"rows": {"base": len(s.base), "stream_seen": len(self._stream_rows(False)),
                         "stream_paid": len(self._stream_rows(True))},
                "task_type": s.task_type, "input_dimension": int(s.base.x.shape[1]),
                "answer": {"classes": s.classes} if s.classes else
                          {"min": float(s.base.y.min()), "max": float(s.base.y.max()), "mean": float(s.base.y.mean())},
                "meta_fields": {f: structure.field_profile(s.base, f, top=3) for f in fields},
                "note": "meta fields are trace data; they are never model inputs and may be absent on the platform"}

    def _field_profile(self, field: str, scope: str = "base"):
        if scope == "stream":
            metas = self._stream_meta(False)
            values = [str(m.get(field)) for m in metas]
            unique, counts = np.unique(values, return_counts=True) if values else ([], [])
            return {"field": field, "rows": len(values), "distinct": len(unique),
                    "top_values": sorted(zip(map(str, unique), map(int, counts)), key=lambda t: -t[1])[:8]}
        return structure.field_profile(self._table(scope), field)

    def _find_structure(self, scope: str = "base"):
        return structure.detect(self._table(scope), self.ctrl.session.task_type)

    def _sequence_check(self, group_by: list, order_by: str | None = None, scope: str = "base"):
        return structure.sequence_check(self._table(scope), list(group_by), order_by, self.ctrl.session.task_type)

    def _group_stats(self, group_by: list, scope: str = "base"):
        s = self.ctrl.session
        if scope == "stream_paid":
            refs = self._stream_rows(True)
            if not refs:
                return {"error": "no released stream labels yet"}
            if list(group_by) == ["answer"]:
                ys = np.concatenate([self.ctrl.stream.labels(w, [q]) for w, q in refs])
                keys = np.asarray([s.classes[int(v)] if s.classes else str(v) for v in ys])
            else:
                metas = [self.ctrl.stream.meta(w, [q])[0] for w, q in refs]
                keys = np.asarray(["|".join(str(m.get(f)) for f in group_by) for m in metas])
            values, counts = np.unique(keys, return_counts=True)
            share = counts / counts.sum()
            return {"group_by": group_by, "scope": scope, "groups": int(len(values)),
                    "largest_share": float(share.max()), "max_over_median": float(counts.max() / max(np.median(counts), 1)),
                    "top": sorted(zip(map(str, values), map(int, counts)), key=lambda t: -t[1])[:8]}
        return structure.group_shares(self._table(scope), list(group_by), s.classes)

    def _error_breakdown(self, by: str = "answer_bins", scope: str = "anchor", model: str = "serving"):
        ctrl, s = self.ctrl, self.ctrl.session
        node = ctrl.tree.serving if model == "serving" else ctrl.tree.pointers["static"]
        if scope == "stream_paid":
            refs = self._stream_rows(True)
            if not refs:
                return {"error": "no released stream labels yet"}
            x, y, metas, _ = ctrl._arrays(refs)
        else:
            table = self._table(scope)
            x, y, metas = table.x, table.y, table.meta
        out = ctrl._predict(node, x)
        if by == "class" and s.classes:
            predicted = out.argmax(axis=1)
            return {"scope": scope, "model": model, "per_class": {
                s.classes[c]: {"n": int(np.sum(y == c)), "recall": float(np.mean(predicted[y == c] == c))}
                for c in np.unique(y)}, "macro_f1": metrics.macro_f1(y, predicted)}
        if by == "answer_bins" and not s.classes:
            edges = np.quantile(y, np.linspace(0, 1, 6))
            bins = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, 4)
            rows = []
            for b in range(5):
                m = bins == b
                if m.any():
                    e = out[m] - y[m]
                    rows.append({"answer_range": [float(y[m].min()), float(y[m].max())], "n": int(m.sum()),
                                 "rmse": float(np.sqrt(np.mean(e ** 2))), "mean_error": float(e.mean())})
            return {"scope": scope, "model": model, "bins": rows, "note": "mean_error > 0 means over-estimates"}
        keys = np.asarray([str(m.get(by)) for m in metas])
        rows = []
        for key in np.unique(keys)[:30]:
            m = keys == key
            rows.append({"value": key, "n": int(m.sum()), "utility": metrics.utility(s.task_type, y[m], out[m])})
        return {"scope": scope, "model": model, "by": by, "groups": sorted(rows, key=lambda r: r["utility"])[:15]}

    def _training_curve(self, node: str = "official"):
        if node == "official":
            path = self.ctrl.task.package / "history.jsonl"
        else:
            if node not in self.ctrl.tree.nodes:
                raise ToolError(f"unknown node {node}")
            path = (self.ctrl.run_dir / self.ctrl.tree.nodes[node].path).parent / "history.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"node": node, "epochs": [{"epoch": r["epoch"], "loss": r["loss"], "validation": r["validation"]}
                                         for r in rows]}

    def _subset(self, name: str):
        ctrl, s = self.ctrl, self.ctrl.session
        if name == "base":
            return s.base.x, s.base.y
        if name == "stream_paid":
            refs = self._stream_rows(True)
            x, y, _, _ = ctrl._arrays(refs)
            return x, y
        if name.startswith("window:"):
            w = int(name.split(":")[1])
            if w >= ctrl.state["next_window"] or w not in ctrl.window_ids:
                raise DataAccessError(f"window {w} has not happened yet")
            x = ctrl.stream.features(w)[1]
            paid = np.asarray(ctrl.state["pool"].get(str(w), []), dtype=np.int64)
            return x, (ctrl.stream.labels(w, paid) if len(paid) else None), paid
        raise ToolError(f"unknown subset {name!r}")

    def _compare_subsets(self, a: str, b: str):
        ctrl, s = self.ctrl, self.ctrl.session
        sides = {}
        for label, name in (("a", a), ("b", b)):
            got = self._subset(name)
            x, y = got[0], got[1]
            paid = got[2] if len(got) == 3 else None
            entry = {"name": name, "rows": int(len(x))}
            if y is not None and len(y):
                xs = x[paid] if paid is not None else x
                entry["serving_utility"] = metrics.utility(s.task_type, y, ctrl._predict(ctrl.tree.serving, xs))
                entry["static_utility"] = metrics.utility(s.task_type, y, ctrl._predict(ctrl.tree.pointers["static"], xs))
            sides[label] = (ctrl.normalize(x), entry)
        mmd = mmd_rbf(sides["a"][0], sides["b"][0])
        psi_max, psi_mean = psi(sides["a"][0], sides["b"][0])
        null = ctrl.meter.threshold(min(len(sides["a"][0]), len(sides["b"][0])))
        return {"a": sides["a"][1], "b": sides["b"][1], "mmd": mmd, "mmd_null_threshold": null,
                "psi_max": psi_max, "psi_mean": psi_mean}

    def _run_status(self):
        st = self.ctrl.state
        return {"next_window": st["next_window"], "windows_total": len(self.ctrl.window_ids),
                "recent_health": st["health"][-5:], "strategy_version": st["strategy_version"],
                "policy": {k: v for k, v in st["policy"].items() if k in experiments.EXPERIMENT_FIELDS},
                "code_version": st["code_version"], "promotions": st["promotions"], "rollbacks": st["rollbacks"],
                "budget": self.ctrl.ledger.remaining(), "last_diagnosis": st["last_diagnosis"]}

    def _signature(self):
        return signature.compute(self.ctrl)

    def _experiment(self, operator: str | None = None, params: dict | None = None, validation: dict | None = None):
        if not operator:
            raise ToolError("give an operator name and its params (see the manual)")
        try:
            inst = operators.instantiate(operator, params, signature.compute(self.ctrl), self.ctrl.policy,
                                         set(structure._fields(self.ctrl.session.base)))
        except operators.OperatorError as error:
            raise ToolError(str(error))
        return self.validate(inst, validation)

    def validate(self, inst, validation: dict | None = None, tag: str = "") -> dict:
        """Run the experiment the manual prescribes for one operator instance (also used by the supervisor)."""
        protocol = inst.operator.validation
        if protocol in ("none", "unverifiable", "probation"):
            ok = inst.kind in ("tightening", "meta", "mode")
            result = {"mode": protocol, "n": 0,
                      "skipped": ("applies at once; the probation that follows judges it" if protocol == "probation"
                                  else "applies at once" if ok else "this direction cannot be validated during the run")}
            record = {"operator": inst.name, "params": inst.params, "change": inst.changes,
                      "edits": inst.describe()["edits"], "result": result, "passed": ok, "why": result["skipped"]}
            self.experiments.append(record)
            return {**result, "passed": ok, "why": result["skipped"]}
        candidate = PolicyConfig(**{**self.ctrl.policy.to_dict(), **inst.changes})
        try:
            code = experiments.candidate_code(self.ctrl.code, inst.edits, self.code_store)
        except (SyntaxError, ValueError) as error:
            raise ToolError(f"slot edits rejected: {error}")
        spec = dict(validation or {})
        if inst.operator.name.startswith("data.cap") and "group_by" not in spec and inst.changes.get("cap_group_by") != ["answer"]:
            spec["group_by"] = inst.changes.get("cap_group_by")           # hold out whole groups of the capped key
        result = experiments.run(self.ctrl, protocol, spec, candidate, code,
                                 self.scratch / f"exp{len(self.experiments):02d}")
        self._charge(result)
        ok, why = experiments.passed(result)
        record = {"operator": inst.name, "params": inst.params, "change": inst.changes,
                  "edits": inst.describe()["edits"], "validation": spec, "result": result, "passed": ok, "why": why,
                  "by": tag}
        self.experiments.append(record)
        return {**result, "passed": ok, "why": why}

    def _charge(self, result: dict) -> None:
        """Validation experiments are training like any other: their work is counted in the run's cost."""
        spent = result.get("spent_row_epochs")
        if spent is None:
            spent = sum((result.get("row_epochs") or {}).values())
        if not spent:
            return
        ctrl = self.ctrl
        ctrl.executor.row_epochs += int(spent)
        ctrl.ledger.charge(Cost(train_seconds=float(spent) * ctrl.session.seconds_per_row_epoch))
        ctrl.state["experiment_row_epochs"] = ctrl.state.get("experiment_row_epochs", 0) + int(spent)

    def _read_memory(self, last: int = 5):
        return self.memory.prompt_view(last=int(last)) if self.memory else {"journal": [], "library": ""}


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
