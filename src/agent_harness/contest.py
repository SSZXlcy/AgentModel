"""Platform adapter: one round of the contest workspace protocol (contest_harness SDK 1.0.4).

The platform runs `python3 /app/harness.py train` once per checkpoint round. A round sees the current
stage's public train/validation projection and the parent checkpoint it must continue. The same three
layers as the offline harness decide what to do with it:

  strategy layer   read the data structure (entity, countdown, stream format, input novelty) from the
                   rows, pick a rhythm, and at the first round validate data and code operators by a
                   group k-fold on the training set (whole entities held out; nothing from validation)
  training layer   continue the parent checkpoint with the official trainer (Adam state, normalisation and
                   epoch counter preserved) under the validated strategy; one plain continuation runs
                   beside it as the control
  safety           the platform measures every candidate; locally the paired difference against the parent
                   on the public validation set (entity bootstrap) decides which choices are safe; the
                   model (Qwen via the host gateway) picks among the safe set, and a pick outside it falls
                   back to the rule

State that must outlive the round (validated strategy, rhythm, round ledger, experience credibility) goes
into the platform's state store. Gradient training and fitted statistics use the training set only; the
public validation set is used for measurement and selection.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from . import codebase, metrics
from . import slots as slot_rules
from .agent import parse_json_object
from .controller import _cap
from .data import load_table, write_lines
from .experiments import FUTILITY_FOLDS, _cut_points, _paired
from .signature import COVERED_NOVELTY, NOVELTY_QUANTILE, NOVELTY_REFERENCE_ROWS, _nearest
from .structure import detect

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
STATE_FILE = "harness_state.json"
STATE_VERSION = 1

OPENING_FOLDS = 3
OPENING_ROWS = 6000
ROUND_EPOCHS = 5            # epochs of one continuation candidate (the official continuation example)
CONSERVE_EPOCHS = 2
SELECTION_Z = 1.0           # a candidate is promoted by the rule when its gain exceeds 1 bootstrap sd
BOOTSTRAP_REPS = 200
CAP_IMBALANCE = 3.0         # E-cap: entity (or class) sizes differ by more than 3x the median
REPLY_TOKENS = 400
# The submitted model identity (written into the image). The platform binds the model of each round
# (rnd.model_name) and holds the key; this value is only recorded and compared, never sent anywhere.
MODEL_ID = os.environ.get("AGENT_HARNESS_MODEL", "Qwen/Qwen3.8-27B")


def log(message: str) -> None:
    print("[harness] " + message, file=sys.stderr, flush=True)


def _vendor() -> Path:
    """The official CPU baseline package (its `baselines/` directory is patched once into slot form)."""
    configured = os.environ.get("AGENT_HARNESS_BASELINE")
    if configured:
        return Path(configured)
    here = Path(__file__).resolve()
    for root in (here.parents[2], here.parents[1], Path("/opt/agent-harness")):
        candidate = root / "vendor" / "cpu-baseline"
        if (candidate / "baselines" / "model.py").is_file():
            return candidate
    raise FileNotFoundError("official baseline package not found; set AGENT_HARNESS_BASELINE")


def _sha_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- reading

class Situation:
    """What the strategy layer reads from this round's rows (inputs and meta; answers only from train)."""

    def __init__(self, rnd, work: Path, code: codebase.CodeVersion, parent_ckpt: Path):
        model_module = code.modules().model
        self.parent = model_module.Model.load(parent_ckpt)
        meta = self.parent.meta
        self.task = meta["task"]
        contract_task = (rnd.task or {}).get("task_type") or (rnd.task or {}).get("task")
        if contract_task in TASKS and contract_task != self.task:
            raise ValueError(f"task contract says {contract_task}, parent checkpoint is {self.task}")
        self.classes = list(meta["classes"])
        self.profile_name = meta.get("profile") or rnd.dataset_key
        base = code.profile(self.profile_name) if code.profile_path(self.profile_name).is_file() else {}
        self.profile = {"profile": self.profile_name, "task": self.task, "classes": self.classes,
                        "hidden": meta["hidden"], "seed": meta["seed"], "learning_rate": meta["learning_rate"],
                        "batch_size": meta["batch_size"], "epochs": int(base.get("epochs", 15))}
        if meta.get("target_unit"):
            self.profile["target_unit"] = meta["target_unit"]
        self.profile_path = work / f"profile-{self.profile_name}.json"
        self.profile_path.write_text(json.dumps(self.profile, indent=2) + "\n", encoding="utf-8")
        self.train_path = Path(rnd.data.train_path)
        self.validation_path = Path(rnd.data.validation_path)
        self.train = load_table(self.train_path, code, self.profile, [])
        self.validation = load_table(self.validation_path, code, self.profile, [])
        self.train_sha = _sha_file(self.train_path)
        found = detect(self.train, self.task)
        self.entity = found["entity"] or []
        self.countdown = bool(found.get("countdown"))
        self.structure = {k: v for k, v in found.items() if k != "checked"}
        self.train_groups = self.train.key(self.entity)
        # A validation projection without the entity fields is scored row by row.
        has_entity = bool(self.entity) and all(all(f in m for f in self.entity) for m in self.validation.meta[:50])
        self.val_groups = self.validation.key(self.entity) if has_entity else self.validation.ids
        self.stream_format = self._format(has_entity)
        self.novelty = self._novelty()

    def _format(self, has_entity: bool) -> str:
        if not self.entity:
            return "independent"
        if not has_entity:
            return "sequence"
        _, counts = np.unique(self.val_groups, return_counts=True)
        return "snapshot" if float(np.median(counts)) <= 1 else "sequence"

    def _novelty(self) -> float:
        """Share of validation inputs farther from the training inputs than training rows are from each other
        (inputs only, in the parent's normalisation)."""
        rng = np.random.default_rng(0)
        def sample(x):
            return x if len(x) <= NOVELTY_REFERENCE_ROWS else x[rng.choice(len(x), NOVELTY_REFERENCE_ROWS, replace=False)]
        base = sample(self.parent.normalized(self.train.x))
        radius = float(np.quantile(_nearest(base, base, exclude_self=True), NOVELTY_QUANTILE))
        probe = sample(self.parent.normalized(self.validation.x))
        return round(float(np.mean(_nearest(probe, base) > radius)), 4)

    def group_sizes(self) -> np.ndarray:
        if self.task == "classification":
            return np.bincount(self.train.y.astype(np.int64), minlength=len(self.classes))
        if not self.entity:
            return np.asarray([])
        _, counts = np.unique(self.train_groups, return_counts=True)
        return counts

    def signature(self) -> dict:
        return {"task": self.task, "entity": self.entity, "countdown": self.countdown,
                "stream_format": self.stream_format, "input_novelty": self.novelty,
                "inputs_covered": self.novelty <= COVERED_NOVELTY, "train_rows": len(self.train),
                "validation_rows": len(self.validation), "train_entities": len(set(self.train_groups.tolist()))
                if self.entity else None}


# ------------------------------------------------------------------------------------------ strategy

def choose_rhythm(sit: Situation, state: dict) -> tuple[str, str]:
    """(rhythm, experience id). Rhythm decides how much a round trains; it never skips the control."""
    last = (state.get("rounds") or [None])[-1]
    unchanged = bool(last) and last.get("train_sha") == sit.train_sha
    if last and (unchanged or sit.novelty <= COVERED_NOVELTY) and not last.get("promoted"):
        return "conserve", "E-covered"
    if not sit.entity:
        return "adaptive", "E-adaptive"
    if sit.countdown and sit.stream_format == "sequence":
        return "event", "E-event"
    if sit.novelty <= COVERED_NOVELTY:
        return "conserve", "E-covered"
    return "standard", "E-standard"


def opening_menu(sit: Situation) -> tuple[list[dict], list[dict]]:
    """Data and code operators worth an opening experiment in this situation (the seed experiences)."""
    data, code = [], []
    sizes = sit.group_sizes()
    if len(sizes) and np.median(sizes) > 0 and sizes.max() > CAP_IMBALANCE * np.median(sizes):
        if sit.task == "classification":
            data.append({"name": "data.class_balance", "experience": "E-class-balance",
                         "cap": int(np.median(sizes[sizes > 0])), "by": ["answer"]})
        else:
            data.append({"name": "data.cap_per_entity", "experience": "E-cap",
                         "cap": int(np.median(sizes)), "by": list(sit.entity)})
    if sit.task == "classification":
        names = [("sqrt_inverse_class_frequency", "E-rare-class-weight"), ("label_smoothing", "E-standard")]
    else:
        names = [("asymmetric_overestimate", "E-overestimate"), ("asymmetric_overestimate_x4", "E-overestimate"),
                 ("relative_error", "E-relative-error"), ("huber", "E-standard")]
    variants = {v.name: v for v in slot_rules.VARIANTS}
    for name, experience in names:
        if name in variants and sit.task in variants[name].tasks:
            code.append({"name": name, "experience": experience, "variant": variants[name]})
    return data, code


def _apply_recipe(sit: Situation, rows: np.ndarray, recipe: dict | None) -> list[str]:
    items = [(sit.train.raw[i], sit.train.meta[i], sit.train.y[i]) for i in rows]
    if recipe and recipe.get("cap"):
        items = _cap(items, int(recipe["cap"]), list(recipe["by"]), sit.classes)
    return [line for line, _, _ in items]


def _version(base: codebase.CodeVersion, variant, store: Path) -> codebase.CodeVersion:
    if variant is None:
        return base
    edits = [variant.edit()]
    return codebase.derive_version(base, slot_rules.apply_edits(base.slots_source(), edits), edits, store)


def _train_run(version, profile_path: Path, train: Path, val: Path, out: Path, stage: str, **kwargs) -> dict:
    for attempt in range(2):
        try:
            return version.modules().train.train_run(str(profile_path), train, val, out, stage_id=stage, **kwargs)
        except PermissionError:          # a Windows scanner holding the fresh file; the output does not exist
            if attempt or out.exists():
                raise
            time.sleep(0.5)


class Opening:
    """Group k-fold on the training set: does an operator make a better model from the official procedure?"""

    def __init__(self, sit: Situation, base: codebase.CodeVersion, scratch: Path, store: Path):
        self.sit, self.base, self.scratch, self.store = sit, base, scratch, store
        rows = np.arange(len(sit.train))
        if sit.entity:
            keys = sit.train_groups
            unique = sorted(set(keys.tolist()))
            self.k = min(OPENING_FOLDS, len(unique))
            fold_of = {g: i % self.k for i, g in enumerate(unique)}
            self.assign = np.asarray([fold_of[g] for g in keys.tolist()])
        else:
            self.k = OPENING_FOLDS
            self.assign = rows % self.k
        self.rows = rows
        self.baseline: dict[int, tuple[float, int]] = {}
        self.spent = 0
        self.snapshot = sit.stream_format == "snapshot" and bool(sit.entity)

    def _fold(self, f: int):
        test, train = self.rows[self.assign == f], self.rows[self.assign != f]
        if len(train) > OPENING_ROWS:
            train = train[np.linspace(0, len(train) - 1, OPENING_ROWS).astype(np.int64)]
        if self.snapshot:
            test = _cut_points(self.sit.train, test, self.sit.entity, seed=f)
        return test, train

    def _arm(self, f: int, name: str, recipe: dict | None, version) -> tuple[float, int]:
        test, train = self._fold(f)
        folder = self.scratch / f"fold{f}"
        val = folder / "val.jsonl"
        if not val.exists():
            write_lines(val, [self.sit.train.raw[i] for i in test])
        path = write_lines(folder / f"{name}.jsonl", _apply_recipe(self.sit, train, recipe))
        out = folder / f"{name}_out"
        report = _train_run(version, self.sit.profile_path, path, val, out, f"opening-{name}-{f}")
        model = version.modules().model.Model.load(out / "checkpoint.npz")
        u = metrics.utility(self.sit.task, self.sit.train.y[test], model.predict(self.sit.train.x[test]))
        work = report["train_rows"] * report["final_epochs"]
        self.spent += work
        return u, work

    def trial(self, name: str, recipe: dict | None, variant) -> dict:
        version = _version(self.base, variant, self.store)
        base_u, cand_u = [], []
        stopped = False
        for f in range(self.k):
            if f not in self.baseline:
                self.baseline[f] = self._arm(f, "baseline", None, self.base)
            base_u.append(self.baseline[f][0])
            cand_u.append(self._arm(f, name, recipe, version)[0])
            if len(cand_u) >= FUTILITY_FOLDS and len(cand_u) < self.k and np.mean(np.subtract(cand_u, base_u)) <= 0:
                stopped = True
                break
        result = _paired(base_u, cand_u, 2.0)
        result.update(name=name, folds_run=len(cand_u), folds=self.k, stopped_early=stopped,
                      scoring="cut_points" if self.snapshot else "all_rows")
        result["passed"] = bool(result["n"] >= 2 and result["mean_diff"] > 0
                                and result["mean_diff"] > SELECTION_Z * result["band"] / 2.0)
        return result


def run_opening(sit: Situation, base: codebase.CodeVersion, work: Path, deadline: float) -> dict:
    """Validate the menu, keep the best data operator and the best code operator, and their combination
    only if it beats both alone."""
    data_menu, code_menu = opening_menu(sit)
    opening = Opening(sit, base, work / "opening", work / "code")
    trials = []
    best_data = best_code = None
    for item in data_menu + code_menu:
        if time.monotonic() > deadline:
            trials.append({"name": item["name"], "skipped": "time budget"})
            continue
        recipe = {"cap": item["cap"], "by": item["by"]} if "cap" in item else None
        result = opening.trial(item["name"], recipe, item.get("variant"))
        result["experience"] = item["experience"]
        trials.append(result)
        log(f"opening {item['name']}: {result['mean_diff']:+.4f} band {result['band']:.4f} passed={result['passed']}")
        if result["passed"]:
            slot = "data" if "cap" in item else "code"
            current = best_data if slot == "data" else best_code
            if current is None or result["mean_diff"] > current[1]["mean_diff"]:
                if slot == "data":
                    best_data = (item, result)
                else:
                    best_code = (item, result)
    chosen = {"recipe": None, "variant": None, "experiences": []}
    picks = [p for p in (best_data, best_code) if p]
    if len(picks) == 2 and time.monotonic() < deadline:
        (d, dr), (c, cr) = picks
        combo = opening.trial("combined", {"cap": d["cap"], "by": d["by"]}, c["variant"])
        trials.append({**combo, "experience": d["experience"] + "+" + c["experience"]})
        if combo["passed"] and combo["mean_diff"] >= max(dr["mean_diff"], cr["mean_diff"]):
            picks = [best_data, best_code]
        else:
            picks = [max(picks, key=lambda p: p[1]["mean_diff"])]
    elif len(picks) == 2:
        picks = [max(picks, key=lambda p: p[1]["mean_diff"])]
    for item, result in picks:
        if "cap" in item:
            chosen["recipe"] = {"name": item["name"], "cap": item["cap"], "by": item["by"]}
        else:
            chosen["variant"] = item["name"]
        chosen["experiences"].append(item["experience"])
    shutil.rmtree(work / "opening", ignore_errors=True)
    return {"chosen": chosen, "trials": [{k: v for k, v in t.items() if k != "variant"} for t in trials],
            "spent_row_epochs": opening.spent}


# --------------------------------------------------------------------------------------- candidates

class Candidate:
    def __init__(self, key: str, directory: Path, report: dict, config: dict, hypothesis: str, change: str):
        self.key, self.directory, self.report, self.config = key, directory, report, config
        self.hypothesis, self.change = hypothesis, change
        self.artifact_id = None
        self.summary = None
        self.local = None
        self.diff = None


def build_candidate(sit: Situation, version, recipe: dict | None, parent_ckpt: Path, work: Path, key: str,
                    epochs: int, hypothesis: str, change: str, extra: dict) -> Candidate:
    rows = np.arange(len(sit.train))
    lines = _apply_recipe(sit, rows, recipe)
    train = write_lines(work / "data" / f"{key}.jsonl", lines) if recipe else sit.train_path
    out = work / "train" / key
    report = _train_run(version, sit.profile_path, train, sit.validation_path, out, key,
                        resume=parent_ckpt, epochs=epochs)
    directory = work / "candidates" / key
    directory.mkdir(parents=True)
    for name in ("checkpoint.npz", "training.json", "history.jsonl"):
        shutil.copyfile(out / name, directory / name)
    code_dir = directory / "baselines"
    code_dir.mkdir()
    for path in sorted(version.package_dir.glob("*.py")):
        shutil.copyfile(path, code_dir / path.name)
    config = {"epochs": epochs, "learning_rate": report["learning_rate"], "batch_size": report["batch_size"],
              "train_rows": report["train_rows"], "data_recipe": recipe, "code_version": version.version_id,
              "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
              "checkpoint_sha256": report["checkpoint_sha256"], "final_steps": report["final_steps"],
              "training_seconds": round(float(report["training_seconds"]), 3), **extra}
    return Candidate(key, directory, report, config, hypothesis, change)


def evaluate(sit: Situation, model_module, candidates: list[Candidate]) -> dict:
    """Paired difference of every candidate against the parent on the public validation set."""
    expected = sit.validation.y
    parent_out = sit.parent.predict(sit.validation.x)
    parent = metrics.report(sit.task, expected, parent_out)
    for c in candidates:
        out = model_module.Model.load(c.directory / "checkpoint.npz").predict(sit.validation.x)
        c.local = metrics.report(sit.task, expected, out)
        diff, sd = metrics.paired_difference(sit.task, expected, out, parent_out, sit.val_groups, BOOTSTRAP_REPS)
        sd *= metrics.small_sample_factor(sit.val_groups)
        c.diff = {"gain": float(diff), "sd": float(sd)}
    return parent


def rule_choice(candidates: list[Candidate], parent_id: str) -> tuple[str, str]:
    promotable = [c for c in candidates if c.artifact_id and c.diff["gain"] > 0
                  and c.diff["gain"] > SELECTION_Z * c.diff["sd"]]
    if not promotable:
        return parent_id, "no candidate beats the parent by more than its bootstrap sd on public validation"
    best = max(promotable, key=lambda c: c.diff["gain"])
    return best.artifact_id, f"{best.key} gains {best.diff['gain']:+.4f} (sd {best.diff['sd']:.4f}) over the parent"


_DECIDER = (
    "你是工业模型持续训练系统（Harness）的决策器。工业模型是官方 NumPy MLP，每轮只能从父检查点续训。"
    "系统已按数据结构选好节奏，并在训练集上做过按设备留出的开局实验；本轮候选都已训练并在公开验证集上测量。"
    "请在给出的 choices 中选择本轮采用的产物：只有收益明显超过波动（gain 大于 sd）才值得换掉父模型，"
    "收益相近时选训练量更小者，没有可靠收益就保留父模型。"
    "只返回 JSON：{\"select\": \"<artifact_id>\", \"reason\": \"一句话理由，不超过60字\"}。")


def ask_model(rnd, sit: Situation, rhythm: str, opening: dict | None, parent_metrics: dict,
              candidates: list[Candidate], rule: tuple[str, str], history: list) -> tuple[str | None, str, str]:
    choices = [{"artifact_id": rnd.parent.artifact_id, "what": "父模型（不换）", "validation": _brief(parent_metrics)}]
    for c in candidates:
        if c.artifact_id:
            choices.append({"artifact_id": c.artifact_id, "what": c.change, "validation": _brief(c.local),
                            "gain_vs_parent": round(c.diff["gain"], 5), "sd": round(c.diff["sd"], 5),
                            "row_epochs": c.config["train_rows"] * c.config["epochs"]})
    payload = {"dataset": rnd.dataset_key, "round": rnd.round_index, "signature": sit.signature(), "rhythm": rhythm,
               "opening": None if not opening else {"chosen": opening["chosen"],
                                                    "trials": [_trial_brief(t) for t in opening["trials"]]},
               "past_rounds": len(history), "choices": choices, "rule_recommendation": rule[0]}
    limit = max(1, min(REPLY_TOKENS, int(rnd.max_output_tokens)))
    receipt = rnd.model([{"role": "system", "content": _DECIDER},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                        max_tokens=limit, temperature=0, enable_thinking=False)
    try:
        reply = parse_json_object(receipt.text)
        return str(reply.get("select") or "") or None, str(reply.get("reason") or "")[:300], receipt.call_id
    except (ValueError, TypeError):
        return None, "reply was not a JSON object", receipt.call_id


def _brief(m: dict | None) -> dict:
    if not m:
        return {}
    keys = ("macro_f1", "accuracy") if "macro_f1" in m else ("rmse", "mae")
    return {k: round(float(m[k]), 4) for k in keys if k in m}


def _trial_brief(t: dict) -> dict:
    if "skipped" in t:
        return {"name": t["name"], "skipped": t["skipped"]}
    return {"name": t["name"], "gain": round(t["mean_diff"], 5), "band": round(t["band"], 5), "passed": t["passed"]}


# ------------------------------------------------------------------------------------------- state

def load_state(rnd) -> dict:
    if rnd.state_path:
        path = Path(rnd.state_path) / STATE_FILE
        if path.is_file():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("version") == STATE_VERSION:
                    return state
            except (OSError, ValueError):
                log("state unreadable; starting fresh")
    return {"version": STATE_VERSION, "rounds": [], "strategy": None, "credibility": {}}


def save_state(rnd, state: dict, work: Path) -> None:
    target = work / "state"
    if rnd.state_path and Path(rnd.state_path).is_dir():
        target = rnd.state_workspace(str(target))
    else:
        target.mkdir(parents=True, exist_ok=True)
    (target / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    rnd.save_state(target)


def _credit(state: dict, experiences: list[str], gain: float | None) -> None:
    """Credibility of the experiences that shaped the round moves with what the round realised."""
    table = state.setdefault("credibility", {})
    for e in experiences:
        value = table.get(e, 0.6)
        if gain is not None:
            value = min(0.95, value + 0.05) if gain > 0 else max(0.05, value - 0.1)
        table[e] = round(value, 3)


# ---------------------------------------------------------------------------------------------- train

def train(rnd) -> None:
    started = time.monotonic()
    root = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work = root / f"round-{rnd.round_index}"
    attempt = 1
    while work.exists():                        # a retried round never reuses a half-written directory
        attempt += 1
        work = root / f"round-{rnd.round_index}-{attempt}"
    work.mkdir(parents=True)
    budget = float(rnd.wall_seconds) if isinstance(rnd.wall_seconds, (int, float)) and rnd.wall_seconds else 900.0
    deadline = started + 0.45 * budget          # local experiments stop here; uploads and the model call follow

    state = load_state(rnd)
    history = rnd.history()
    descriptor = rnd.parent.descriptor
    checkpoint_name = descriptor["checkpoint_paths"][0]
    parent_ckpt = rnd.parent.path(checkpoint_name, root=os.environ.get("CONTEST_PREVIOUS"))
    base = codebase.base_version(_vendor(), work / "code")
    model_module = base.modules().model
    sit = Situation(rnd, work, base, parent_ckpt)
    signature = sit.signature()
    log(f"round {rnd.round_index} dataset {rnd.dataset_key}: {json.dumps(signature, ensure_ascii=False)}")
    if rnd.model_name.split("/")[-1].lower() != MODEL_ID.split("/")[-1].lower():
        log(f"platform bound model {rnd.model_name!r}, submitted model is {MODEL_ID!r}; using the platform's")

    rhythm, rhythm_experience = choose_rhythm(sit, state)
    opening = None
    if state.get("strategy") is None:
        try:
            opening = run_opening(sit, base, work, deadline)
            state["strategy"] = opening["chosen"]
        except Exception as error:          # the experiment is optional; the round still has its control
            log(f"opening failed ({type(error).__name__}: {error}); continuing with the official procedure")
            state["strategy"] = {"recipe": None, "variant": None, "experiences": [], "failed": str(error)[:200]}
    strategy = state["strategy"]
    variants = {v.name: v for v in slot_rules.VARIANTS}
    strategy_version = _version(base, variants.get(strategy.get("variant")), work / "code")
    epochs = CONSERVE_EPOCHS if rhythm == "conserve" else ROUND_EPOCHS
    common = {"rhythm": rhythm, "rhythm_experience": rhythm_experience, "signature": signature,
              "stage_id": rnd.stage_id, "round_index": rnd.round_index}

    candidates = [build_candidate(
        sit, base, None, parent_ckpt, work, "control", epochs,
        hypothesis=f"父模型在本阶段公开训练集上按官方流程续训 {epochs} 轮即可跟上数据变化（对照组）。",
        change=f"官方训练代码与配置，从父检查点续训 {epochs} 轮（Adam 状态、标准化与 epoch 计数保留）。",
        extra={**common, "arm": "control"})]
    has_strategy = bool(strategy.get("recipe") or strategy.get("variant"))
    if has_strategy and rhythm != "conserve":
        parts = []
        if strategy.get("recipe"):
            parts.append(f"{strategy['recipe']['name']}（每组上限 {strategy['recipe']['cap']} 行）")
        if strategy.get("variant"):
            parts.append(f"训练插槽 {strategy['variant']}")
        what = " + ".join(parts)
        candidates.append(build_candidate(
            sit, strategy_version, strategy.get("recipe"), parent_ckpt, work, "strategy", epochs,
            hypothesis=f"开局按设备留出验证通过的 {what} 在续训中同样优于官方流程（经验 {','.join(strategy['experiences'])}）。",
            change=f"从父检查点续训 {epochs} 轮，使用 {what}；网络结构、输入与检查点格式不变。",
            extra={**common, "arm": "strategy", "strategy": strategy}))

    parent_metrics = evaluate(sit, model_module, candidates)
    for c in candidates:
        c.config["local_validation"] = _brief(c.local)
        c.config["paired_vs_parent"] = {k: round(v, 6) for k, v in c.diff.items()}
        if opening and c.key == "strategy":
            c.config["opening"] = {"trials": [_trial_brief(t) for t in opening["trials"]],
                                   "spent_row_epochs": opening["spent_row_epochs"]}
        reply = rnd.submit_candidate(
            c.directory, hypothesis=c.hypothesis, change_summary=c.change, configuration=c.config,
            seed=int(sit.parent.meta["seed"]), checkpoint_paths=["checkpoint.npz"],
            code_paths=sorted("baselines/" + p.name for p in (c.directory / "baselines").glob("*.py")),
            predict_argv=PREDICT_ARGV)
        c.artifact_id = reply["artifact_id"]
        c.summary = reply.get("summary")
        log(f"{c.key}: artifact {c.artifact_id} local {c.config['local_validation']} paired {c.diff}")

    rule = rule_choice(candidates, rnd.parent.artifact_id)
    picked, why, call_id = ask_model(rnd, sit, rhythm, opening, parent_metrics, candidates, rule, history)
    allowed = {rnd.parent.artifact_id} | {c.artifact_id for c in candidates if c.artifact_id and c.diff["gain"] > 0}
    if picked in allowed:
        selected, reason = picked, f"模型选择：{why}（规则建议 {rule[0]}：{rule[1]}）"
    else:
        selected, reason = rule[0], f"规则选择：{rule[1]}（模型回复 {picked or '无效'} 不在安全集合内：{why}）"
    chosen = next((c for c in candidates if c.artifact_id == selected), None)
    gain = chosen.diff["gain"] if chosen else 0.0

    _credit(state, [rhythm_experience] + list(strategy.get("experiences") or []),
            max((c.diff["gain"] for c in candidates), default=None))
    state["rounds"].append({"round": rnd.round_index, "stage": rnd.stage_id, "train_sha": sit.train_sha,
                            "rhythm": rhythm, "selected": selected, "promoted": chosen is not None,
                            "gain_vs_parent": round(gain, 6), "novelty": sit.novelty,
                            "seconds": round(time.monotonic() - started, 1)})
    state["rounds"] = state["rounds"][-50:]
    save_state(rnd, state, work)

    final = chosen or candidates[0]
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=final.hypothesis if chosen else "本轮候选在公开验证集上没有超过波动的收益，保留父模型更稳。",
        change_summary=final.change if chosen else "未采用新候选；父检查点保持不变。",
        reason=reason[:3900],
        configuration={"rhythm": rhythm, "selected_arm": chosen.key if chosen else "parent",
                       "candidates": {c.key: c.artifact_id for c in candidates},
                       "paired_vs_parent": {c.key: c.config["paired_vs_parent"] for c in candidates},
                       "rule": rule[0], "model_pick": picked, "signature": signature,
                       "model": {"submitted": MODEL_ID, "bound": rnd.model_name}},
        seed=int(sit.parent.meta["seed"]),
        history_ids=[item["id"] for item in history],
        model_call_ids=[call_id])
    log(f"round finished in {time.monotonic() - started:.1f}s; selected {selected}")


# -------------------------------------------------------------------------------------------- predict

_MODELS: dict[str, object] = {}


def _official():
    vendor = _vendor() / "baselines"
    if str(vendor.parent) not in sys.path:
        sys.path.insert(0, str(vendor.parent))
    from baselines import data as official_data, model as official_model   # noqa: E402  (official package)
    return official_data, official_model


def predict(sample, model_dir):
    path = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    official_data, official_model = _official()
    if path not in _MODELS:
        _MODELS[path] = official_model.Model.load(path)
    model = _MODELS[path]
    paths, values = official_data.flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"value": float(max(0.0, output))}
    scores = [min(1.0, max(0.0, float(s))) for s in output]
    top = max(scores)
    seen = False
    for i, s in enumerate(scores):           # exactly one label may hold the top score
        if s == top:
            if seen:
                scores[i] = float(np.nextafter(s, -1.0))
            seen = True
    return [{"label": label, "score": score} for label, score in zip(model.meta["classes"], scores)]
