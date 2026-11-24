"""Editable slots of the official trainer.

The outer layer never edits the official files freely. `baselines/model.py` is patched once
(see codebase.py) to call four functions in `baselines/slots.py`; only the bodies of those
functions may change. The default bodies reproduce the official arithmetic exactly, so the
unedited version produces byte-identical checkpoints.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import textwrap

SIGNATURES: dict[str, tuple[str, ...]] = {
    "sample_weights": ("x", "target", "task", "n_classes"),
    "loss_and_delta": ("logits", "expected", "task", "weights"),
    "regularizer_gradients": ("params",),
    "learning_rate": ("base_lr", "step", "epoch"),
}

# Two orthogonal groups proposed independently (the harness2 INV/DOM split).
GROUPS: dict[str, tuple[str, ...]] = {
    "OPT": ("regularizer_gradients", "learning_rate"),   # how to optimise
    "PHM": ("loss_and_delta", "sample_weights"),         # what the task and equipment demand
}

ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "os", "sys", "subprocess", "socket", "shutil",
    "pathlib", "Path", "importlib", "urllib", "requests", "pickle",
}

_CLASSIFICATION_DEFAULT = '''\
    if task == "classification":
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        picked = np.log(np.maximum(probabilities[np.arange(count), expected], 1e-300))
        delta = probabilities
        delta[np.arange(count), expected] -= 1
        if weights is None:
            delta /= count
            return -float(picked.sum()), delta
        delta *= (weights / count)[:, None]
        return -float((weights * picked).sum()), delta
'''

DEFAULT_SOURCE = '''\
"""Editable training slots. The harness may replace the bodies of these four functions only."""
import numpy as np


def sample_weights(x, target, task, n_classes):
    """Per-sample weights for one fit() call, or None for uniform weighting."""
    return None


def loss_and_delta(logits, expected, task, weights):
    """Return (summed loss, dL/dlogits averaged over the batch) for one mini-batch."""
    count = len(expected)
''' + _CLASSIFICATION_DEFAULT + '''\
    error = logits[:, 0] - expected
    if weights is None:
        return float(np.sum(error ** 2)), (2 * error / count)[:, None]
    return float(np.sum(weights * error ** 2)), (2 * weights * error / count)[:, None]


def regularizer_gradients(params):
    """Extra gradient terms added before clipping, keyed by parameter name."""
    return {}


def learning_rate(base_lr, step, epoch):
    """Learning rate for the update at cumulative step `step` (1-based)."""
    return base_lr
'''


# ---------------------------------------------------------------------------- checking

def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def check_module(source: str) -> list[str]:
    """Every reason this slots module is unacceptable; empty means acceptable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"syntax error: {error}"]
    errors: list[str] = []
    seen: list[str] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and index == 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(f"import {alias.name} is not allowed")
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and _private_or_constant(t.id)
                                                for t in node.targets):
            continue                     # module constants such as _EPS = 1e-9 or SMOOTHING = 0.05
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
            if node.decorator_list:
                errors.append(f"helper {node.name} may not use decorators")
            continue                     # private helpers called by the slots
        if isinstance(node, ast.FunctionDef):
            if node.name not in SIGNATURES:
                errors.append(f"function {node.name} is not a slot (helpers must start with _)")
                continue
            seen.append(node.name)
            args = tuple(arg.arg for arg in node.args.args)
            if args != SIGNATURES[node.name] or node.args.vararg or node.args.kwarg or node.decorator_list:
                errors.append(f"{node.name} must keep signature ({', '.join(SIGNATURES[node.name])})")
            continue
        errors.append(f"top-level statement at line {node.lineno} is outside every slot")
    for name in SIGNATURES:
        if seen.count(name) != 1:
            errors.append(f"slot {name} must be defined exactly once")
    helpers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_")]
    if len(helpers) != len(set(helpers)):
        errors.append("helper functions must have unique names")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal is not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            errors.append("imports inside slots are not allowed")
        if isinstance(node, ast.ImportFrom):
            errors.append("from-imports are not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"name {node.id} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute {node.attr} is not allowed")
    return sorted(set(errors))


def _private_or_constant(name: str) -> bool:
    return name.startswith("_") or name.isupper()


def changed_slots(old: str, new: str) -> list[str]:
    """Slots whose code differs semantically (formatting-only changes do not count)."""
    a, b = _functions(ast.parse(old)), _functions(ast.parse(new))
    return [name for name in SIGNATURES
            if ast.dump(a[name], include_attributes=False) != ast.dump(b[name], include_attributes=False)]


def apply_edits(base: str, edits: list[dict]) -> str:
    """Replace whole slot functions. Each edit is {"slot": name, "source": "def name(...): ..."}."""
    lines = base.splitlines(keepends=True)
    tree = ast.parse(base)
    functions = _functions(tree)
    replacements = []
    for edit in edits:
        slot = edit.get("slot")
        if slot not in SIGNATURES:
            raise ValueError(f"unknown slot {slot!r}")
        source = textwrap.dedent(str(edit.get("source", ""))).strip("\n") + "\n"
        parsed = ast.parse(source)
        defs = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == slot]
        extras = [n for n in parsed.body if not (isinstance(n, ast.FunctionDef) and n.name == slot)]
        if len(defs) != 1 or any(not ((isinstance(n, ast.FunctionDef) and n.name.startswith("_")) or
                                      (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and
                                                                          _private_or_constant(t.id) for t in n.targets)))
                                 for n in extras):
            raise ValueError(f"edit for {slot} must define {slot} once, plus optional _helpers and constants")
        node = functions[slot]
        replacements.append((node.lineno - 1, node.end_lineno, source))
    for start, end, source in sorted(replacements, reverse=True):
        lines[start:end] = [source]
    return "".join(lines)


# ---------------------------------------------------------------------------- variant catalogue

@dataclass(frozen=True)
class Variant:
    """A known slot body the local (offline) proposer can offer when no LLM is available."""

    name: str
    slot: str
    tasks: tuple[str, ...]
    hypothesis: str
    source: str

    @property
    def group(self) -> str:
        return next(group for group, names in GROUPS.items() if self.slot in names)

    def edit(self) -> dict:
        return {"slot": self.slot, "variant": self.name, "hypothesis": self.hypothesis, "source": self.source}


def _loss(regression_body: str, classification: str = _CLASSIFICATION_DEFAULT) -> str:
    return ("def loss_and_delta(logits, expected, task, weights):\n"
            '    """Return (summed loss, dL/dlogits averaged over the batch) for one mini-batch."""\n'
            "    count = len(expected)\n" + classification + regression_body)


_SMOOTHED_CLASSIFICATION = '''\
    if task == "classification":
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        classes = probabilities.shape[1]
        smoothing = 0.05
        targets = np.full_like(probabilities, smoothing / classes)
        targets[np.arange(count), expected] += 1.0 - smoothing
        per_sample = -(targets * np.log(np.maximum(probabilities, 1e-300))).sum(axis=1)
        scale = np.ones(count) if weights is None else weights
        delta = (probabilities - targets) * (scale / count)[:, None]
        return float((scale * per_sample).sum()), delta
'''

_DEFAULT_REGRESSION = '''\
    error = logits[:, 0] - expected
    if weights is None:
        return float(np.sum(error ** 2)), (2 * error / count)[:, None]
    return float(np.sum(weights * error ** 2)), (2 * weights * error / count)[:, None]
'''

VARIANTS: tuple[Variant, ...] = (
    Variant("sqrt_inverse_class_frequency", "sample_weights", ("classification",),
            "Rare classes dominate Macro-F1 but barely move the mean loss; weight samples by "
            "1/sqrt(class count) so every class contributes comparably without extreme weights.",
            '''def sample_weights(x, target, task, n_classes):
    """Per-sample weights for one fit() call, or None for uniform weighting."""
    if task != "classification":
        return None
    labels = target.astype(np.int64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts[labels], 1.0))
    return weights / weights.mean()
'''),
    Variant("target_range_balance", "sample_weights", ("regression",),
            "Long-lived units contribute most rows, so short remaining lives are under-weighted; "
            "balance ten equal-width target bins with 1/sqrt(bin count) weights.",
            '''def sample_weights(x, target, task, n_classes):
    """Per-sample weights for one fit() call, or None for uniform weighting."""
    if task != "regression":
        return None
    low, span = target.min(), max(float(target.max() - target.min()), 1e-12)
    bins = np.minimum(((target - low) / span * 10).astype(np.int64), 9)
    counts = np.bincount(bins, minlength=10).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts[bins], 1.0))
    return weights / weights.mean()
'''),
    Variant("asymmetric_overestimate", "loss_and_delta", ("regression",),
            "Over-estimating remaining life is the dangerous error (NASA score penalises it more); "
            "double the squared-error weight of over-estimates.",
            _loss('''\
    error = logits[:, 0] - expected
    scale = np.where(error > 0, 2.0, 1.0)
    if weights is not None:
        scale = scale * weights
    return float(np.sum(scale * error ** 2)), (2 * scale * error / count)[:, None]
''')),
    Variant("asymmetric_overestimate_x4", "loss_and_delta", ("regression",),
            "A single observation rarely tells how long a unit will last, so short remaining lives are read as "
            "average ones and over-estimated; weight the squared error of over-estimates four times.",
            _loss('''\
    error = logits[:, 0] - expected
    scale = np.where(error > 0, 4.0, 1.0)
    if weights is not None:
        scale = scale * weights
    return float(np.sum(scale * error ** 2)), (2 * scale * error / count)[:, None]
''')),
    Variant("relative_error", "sample_weights", ("regression",),
            "Rows far from the end dominate the squared error although they carry the least information; weight "
            "rows by 1 / (target above its minimum + the median of that), i.e. by relative rather than absolute "
            "error (invariant to the target scaling).",
            '''def sample_weights(x, target, task, n_classes):
    """Per-sample weights for one fit() call, or None for uniform weighting."""
    if task != "regression":
        return None
    shifted = target - target.min()
    weights = 1.0 / (shifted + max(float(np.median(shifted)), 1e-6))
    return weights / weights.mean()
'''),
    Variant("huber", "loss_and_delta", ("regression",),
            "Run-to-failure targets have heavy tails; a Huber loss (delta = 1 target std) limits "
            "the pull of the largest residuals.",
            _loss('''\
    error = logits[:, 0] - expected
    scale = np.ones_like(error) if weights is None else weights
    small = np.abs(error) <= 1.0
    loss = np.where(small, error ** 2, 2 * np.abs(error) - 1.0)
    gradient = np.where(small, 2 * error, 2 * np.sign(error))
    return float(np.sum(scale * loss)), (scale * gradient / count)[:, None]
''')),
    Variant("label_smoothing", "loss_and_delta", ("classification",),
            "Drifting sensors make the network over-confident on stale boundaries; 5% label "
            "smoothing keeps logits from saturating.",
            _loss(_DEFAULT_REGRESSION, _SMOOTHED_CLASSIFICATION)),
    Variant("l2_1e-4", "regularizer_gradients", ("classification", "regression"),
            "Weights keep growing during continued training; mild L2 (1e-4) on w1/w2 limits it.",
            '''def regularizer_gradients(params):
    """Extra gradient terms added before clipping, keyed by parameter name."""
    return {"w1": 1e-4 * params["w1"], "w2": 1e-4 * params["w2"]}
'''),
    Variant("l2_1e-3", "regularizer_gradients", ("classification", "regression"),
            "Validation degrades while training loss falls; stronger L2 (1e-3) trades fit for stability.",
            '''def regularizer_gradients(params):
    """Extra gradient terms added before clipping, keyed by parameter name."""
    return {"w1": 1e-3 * params["w1"], "w2": 1e-3 * params["w2"]}
'''),
    Variant("inverse_time_decay", "learning_rate", ("classification", "regression"),
            "Continued training from a converged checkpoint overshoots; decay the rate as 1/(1+step/2000).",
            '''def learning_rate(base_lr, step, epoch):
    """Learning rate for the update at cumulative step `step` (1-based)."""
    return base_lr / (1.0 + step / 2000.0)
'''),
)


def variants_for(task_type: str, group: str) -> list[Variant]:
    return [v for v in VARIANTS if task_type in v.tasks and v.slot in GROUPS[group]]
