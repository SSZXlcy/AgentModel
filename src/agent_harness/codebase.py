"""Code versions θ of the official trainer.

v0 is the official `baselines/` package with a minimal, behaviour-preserving patch that routes
four computations through `baselines/slots.py`. Every later version differs from its parent
only in slots.py. Versions are immutable directories; promotion just moves a pointer.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from . import slots as slot_rules
from .contracts import digest

OFFICIAL_MODEL_SHA = "325026d80747eb36"

# (anchor, replacement) pairs applied to the official model.py; each anchor must occur once.
_MODEL_PATCH = [
    ("import numpy as np\n",
     "import numpy as np\n"
     "try:\n    from . import slots\nexcept ImportError:\n    import slots\n"),
    ("        epoch_losses = []\n",
     "        epoch_losses = []\n"
     "        weights = slots.sample_weights(x, target, self.meta[\"task\"], len(self.meta[\"classes\"]))\n"),
    ("""                if self.meta["task"] == "classification":
                    logits -= logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(indices))[:, None]
""",
     """                batch_weights = None if weights is None else weights[indices]
                batch_loss, delta = slots.loss_and_delta(logits, expected, self.meta["task"], batch_weights)
                loss_sum += batch_loss
"""),
    ("                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))\n",
     "                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))\n"
     "                for name, extra in slots.regularizer_gradients(self.params).items():\n"
     "                    gradients[name] = gradients[name] + extra\n"),
    ("                self.meta[\"steps\"] += 1\n",
     "                self.meta[\"steps\"] += 1\n"
     "                learning_rate = slots.learning_rate(self.meta[\"learning_rate\"], self.meta[\"steps\"], self.meta[\"epochs\"])\n"),
    ("self.params[name] -= self.meta[\"learning_rate\"] * first",
     "self.params[name] -= learning_rate * first"),
]

# slots.py joins the provenance digest the official report already records.
_TRAIN_PATCH = [
    ('("model.py", "data.py", "train.py", "predict.py")',
     '("model.py", "data.py", "train.py", "predict.py", "slots.py")'),
]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _patch(text: str, pairs: list[tuple[str, str]], name: str) -> str:
    for anchor, replacement in pairs:
        if text.count(anchor) != 1:
            raise ValueError(f"{name}: patch anchor not found exactly once: {anchor[:60]!r}")
        text = text.replace(anchor, replacement)
    return text


@dataclass
class CodeVersion:
    version_id: str
    root: Path
    parent_id: str | None

    @property
    def package_dir(self) -> Path:
        return self.root / "baselines"

    def slots_source(self) -> str:
        return (self.package_dir / "slots.py").read_text(encoding="utf-8")

    def profile_path(self, profile: str) -> Path:
        return self.package_dir / "configs" / f"{profile}.json"

    def profile(self, profile: str) -> dict:
        return json.loads(self.profile_path(profile).read_text(encoding="utf-8"))

    def info(self) -> dict:
        return json.loads((self.root / "version.json").read_text(encoding="utf-8"))

    def modules(self) -> SimpleNamespace:
        return _load(self)


_LOADED: dict[str, SimpleNamespace] = {}


def _load(version: CodeVersion) -> SimpleNamespace:
    """Import a version's package in-process under a unique name so versions never collide."""
    key = str(version.package_dir.resolve())
    if key not in _LOADED:
        # The same version id can live in two stores; the module name must name the directory too.
        name = "_harness_code_" + version.version_id.replace("-", "_") + "_" + _sha(key.encode("utf-8"))[:8]
        spec = importlib.util.spec_from_file_location(
            name, version.package_dir / "__init__.py", submodule_search_locations=[str(version.package_dir)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        has_slots = (version.package_dir / "slots.py").is_file()
        train = importlib.import_module(name + ".train")
        train.read_dataset = _memoized(train.read_dataset)
        _LOADED[key] = SimpleNamespace(
            train=train, model=importlib.import_module(name + ".model"),
            data=importlib.import_module(name + ".data"),
            slots=importlib.import_module(name + ".slots") if has_slots else None)
    return _LOADED[key]


_PARSED: "OrderedDict[tuple, dict]" = OrderedDict()


def _memoized(read_dataset):
    """Content-addressed cache around the official parser; the official file is untouched.

    train_run re-parses its train and validation files on every call, and parsing dominates run
    time (each feature path is JSON-encoded per row). Files with identical bytes and arguments
    return a copy of the first result, which the official function already validated.
    """
    def cached(path, task, classes=None, expected_paths=None):
        raw = Path(path).read_bytes()
        key = (_sha(raw), task, tuple(classes or ()), tuple(expected_paths) if expected_paths else None)
        if key not in _PARSED:
            _PARSED[key] = read_dataset(path, task, classes, expected_paths)
            while len(_PARSED) > 32:
                _PARSED.popitem(last=False)
        _PARSED.move_to_end(key)
        return {k: (v.copy() if isinstance(v, (np.ndarray, list)) else v) for k, v in _PARSED[key].items()}
    cached.__wrapped__ = read_dataset
    return cached


def _write_version(root: Path, info: dict) -> None:
    (root / "version.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def base_version(package: Path, store: Path) -> CodeVersion:
    """v0: the official package plus the slot patch. Idempotent."""
    source = package / "baselines"
    model_raw = (source / "model.py").read_bytes()
    if _sha(model_raw)[:16] != OFFICIAL_MODEL_SHA:
        raise ValueError(f"{source}/model.py is not the official version")
    identity = {name: _sha((source / name).read_bytes()) for name in ("model.py", "train.py", "data.py", "predict.py")}
    version_id = "v0-" + digest({"official": identity, "slots": slot_rules.DEFAULT_SOURCE}, 8)
    root = store / version_id
    if (root / "version.json").is_file():
        return CodeVersion(version_id, root, None)
    staging = store / f".{version_id}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging / "baselines", ignore=shutil.ignore_patterns("__pycache__"))
    target = staging / "baselines"
    (target / "model.py").write_text(_patch(model_raw.decode("utf-8"), _MODEL_PATCH, "model.py"), encoding="utf-8")
    train = (source / "train.py").read_text(encoding="utf-8")
    (target / "train.py").write_text(_patch(train, _TRAIN_PATCH, "train.py"), encoding="utf-8")
    (target / "slots.py").write_text(slot_rules.DEFAULT_SOURCE, encoding="utf-8")
    _write_version(staging, {"version_id": version_id, "parent": None, "official_package": package.as_posix(),
                             "official_sha256": identity, "edits": []})
    store.mkdir(parents=True, exist_ok=True)
    staging.replace(root)
    return CodeVersion(version_id, root, None)


def derive_version(parent: CodeVersion, slots_source: str, edits: list[dict], store: Path) -> CodeVersion:
    """A child version that differs from `parent` only in slots.py."""
    errors = slot_rules.check_module(slots_source)
    if errors:
        raise ValueError("; ".join(errors))
    version_id = "c" + digest({"parent": parent.version_id, "slots": slots_source}, 10)
    root = store / version_id
    if (root / "version.json").is_file():
        return CodeVersion(version_id, root, parent.version_id)
    staging = store / f".{version_id}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(parent.package_dir, staging / "baselines", ignore=shutil.ignore_patterns("__pycache__"))
    (staging / "baselines" / "slots.py").write_text(slots_source, encoding="utf-8")
    _write_version(staging, {"version_id": version_id, "parent": parent.version_id,
                             "slots_sha256": _sha(slots_source.encode("utf-8")),
                             "edits": [{k: v for k, v in e.items() if k != "source"} for e in edits]})
    staging.replace(root)
    return CodeVersion(version_id, root, parent.version_id)


def open_version(store: Path, version_id: str) -> CodeVersion:
    root = store / version_id
    info = json.loads((root / "version.json").read_text(encoding="utf-8"))
    return CodeVersion(version_id, root, info.get("parent"))
