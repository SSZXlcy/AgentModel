"""Operator manual (修改说明书): every way the system may change how the industrial model is
trained, selected or served, as a named operator with a contract.

Each operator states
  - what it changes, in which layer of the training pipeline
  - its parameters, their legal range, and how a default is computed from the situation signature
    (placeholders such as $entity are filled per task, so one experience transfers across tasks)
  - whether it tightens (applies at once) or tries to improve (must pass validation first)
  - how it is validated: before the stream (group k-fold on the training set) and/or during the run
  - its cost effect and how it is undone

The official trainer itself (architecture, normalisation, optimiser constants, the training loop,
parsing, checkpoint format) is not an operator: it never changes.

Experiences, the deterministic explorer and the LLM all speak this one vocabulary; the
supervisor executes it. `manual_markdown()` renders the same registry for people.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import signature as sig_rules
from . import slots as slot_rules
from .config import PolicyConfig

LAYERS = {
    "mode": "训练节奏", "label": "标注采购", "data": "训练数据", "train": "训练参数", "code": "训练代码",
    "split": "检验拆分", "gate": "晋升门槛", "serve": "上线方式", "supervise": "监督节奏",
}
VALIDATIONS = {
    "probation": "整套节奏：依据数据特征立即生效，随后进入观察期，按实际表现判定，变差就整体撤销（重放无法比较节奏，"
                 "因为不同节奏购买的标注不同）",
    "none": "收紧或节奏类修改：立即生效，无需验证",
    "fixed": "比较模型质量：开局用训练集按组 k 折（从头训练），运行中用固定流程重放已发生的批次（每批都训练、都上线）",
    "full": "比较选择规则：运行中用完整流程重放已发生的批次（含门槛与上线逻辑）",
    "unverifiable": "运行中无法验证（重放只能使用当时已购买的标注），放宽方向一律不采纳",
}
SERVE_RANK = {"trial": 0, "shadow": 1, "freeze": 2}


class OperatorError(ValueError):
    """An operator call that violates the manual; the message tells the caller what to fix."""


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                       # int | float | fields | enum | variant | source
    doc: str
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    default: object = None          # literal value or a placeholder such as "$entity"
    optional: bool = False


@dataclass(frozen=True)
class Operator:
    name: str
    layer: str
    title: str
    doc: str
    params: tuple[Param, ...]
    effect: Callable[[dict], dict]                         # resolved params -> policy changes
    direction: Callable[[PolicyConfig, dict], str]         # -> tightening | improvement | meta | mode
    validation: str                                        # see VALIDATIONS
    opening: bool                                          # validatable before the stream
    cost: str
    undo: str
    requires: dict = field(default_factory=dict)           # signature condition (signature.holds)
    requires_doc: str = "任何任务"
    slot: str | None = None                                # code operators only
    example: str = ""


@dataclass
class Instance:
    """An operator call with every parameter resolved for the current task and strategy."""
    operator: Operator
    params: dict
    changes: dict                    # PolicyConfig field -> new value
    edits: list[dict]                # slot edits (code operators)
    kind: str                        # tightening | improvement | meta | mode

    @property
    def name(self) -> str:
        return self.operator.name

    def describe(self) -> dict:
        return {"operator": self.name, "params": self.params, "kind": self.kind, "changes": self.changes,
                "edits": [{k: e.get(k) for k in ("slot", "variant", "hypothesis")} for e in self.edits]}


# ------------------------------------------------------------------------ direction rules

def _always(kind: str):
    return lambda policy, changes: kind


def _higher_is_tighter(field_name: str):
    return lambda policy, changes: "tightening" if changes[field_name] >= getattr(policy, field_name) else "improvement"


def _lower_is_tighter(field_name: str):
    return lambda policy, changes: "tightening" if changes[field_name] <= getattr(policy, field_name) else "improvement"


def _serve(policy: PolicyConfig, changes: dict) -> str:
    return "tightening" if SERVE_RANK[changes["serve_mode"]] >= SERVE_RANK[policy.serve_mode] else "improvement"


def _split(policy: PolicyConfig, changes: dict) -> str:
    return "improvement" if changes["split_mode"] == "random" and policy.split_mode != "random" else "tightening"


# ------------------------------------------------------------------------ the registry

def _p_int(name, doc, low, high, default=None):
    return Param(name, "int", doc, low, high, default=default)


def _p_float(name, doc, low, high, default=None):
    return Param(name, "float", doc, low, high, default=default)


_CODE_PARAMS = (
    Param("variant", "variant", "内置实现的名字（见“可用内置实现”），与 source 二选一", optional=True),
    Param("source", "source", "完整的插槽函数源码；可附带以 _ 开头的辅助函数和全大写常量。必须同时支持分类和回归"
          "（官方测试两种都会跑），只能用 np 和 math", optional=True),
    Param("hypothesis", "source", "为什么这样改、预期效果（写入记录）", optional=True),
)

REGISTRY: dict[str, Operator] = {}


def _register(op: Operator) -> Operator:
    REGISTRY[op.name] = op
    return op


_FRACTION = "每批请求标注的比例"
_RHYTHM = {
    "adaptive": ("积极跟随", "答案当批就到、输入在变化且更新有收益时：当批全部新答案（连同官方训练集和历史）训练一个候选，"
                 "直接试用；下一批的新答案确认它或回退。没有当批拆分，也不在当批过门槛，适应最快。",
                 {"gate_mode": "next_window", "train_trigger": "every_window", "serve_mode": "trial",
                  "label_sampling": "random"}, 0.5, {}, "任何任务", "每批训练一次（一个候选）"),
    "event": ("设备完成时更新", "答案要等设备（实体）跑完才有时：设备运转期间不训练；某台设备跑完时，用它（没人训练过）检验"
              "“用其余全部数据训练的候选”，更好才用全部数据重训并上线。标注沿时间均匀少买（同一设备相邻行几乎相同）。",
              {"gate_mode": "in_window", "train_trigger": "on_completion", "serve_mode": "trial",
               "label_sampling": "spread"}, 0.25, {"has_entity": True}, "数据有设备（实体）结构", "只在设备完成时训练"),
    "conserve": ("节省", "输入不变、更新没有收益时：暂停训练，少买标注（只用于监控）；环境突变或低于基线时监督器会重新打开。",
                 {"serve_mode": "freeze"}, 0.1, {}, "任何任务", "训练量降为 0，标注减少到 10%（只用于监控）"),
    "standard": ("标准", "默认节奏：当批一半答案训练、另一半检验，过门槛才试用，下一批确认。",
                 {"gate_mode": "in_window", "train_trigger": "every_window", "serve_mode": "trial",
                  "label_sampling": "random"}, 0.5, {}, "任何任务", "每批最多训练 max_candidates 个候选"),
}
for _mode, (_title, _doc, _fields, _fraction, _requires, _requires_doc, _cost) in _RHYTHM.items():
    _register(Operator(
        f"mode.{_mode}", "mode", f"切换为“{_title}”节奏", _doc,
        (_p_float("fraction", _FRACTION, 0.05, 1.0, default=_fraction),),
        (lambda m, f: lambda p: {"mode": m, **f, "label_fraction": p["fraction"]})(_mode, _fields), _always("mode"),
        "probation", False, _cost, "观察期不通过时退回切换前的整套策略",
        requires=_requires, requires_doc=_requires_doc,
        example=f'{{"operator": "mode.{_mode}", "params": {{}}}}'))

_register(Operator(
    "label.sampling", "label", "选择标注的抽取方式",
    "random：随机抽取；spread：沿时间均匀抽取（同一设备相邻行高度相关时，信息相同而标注更少）。",
    (Param("method", "enum", "抽取方式", choices=("random", "spread"), default="spread"),),
    lambda p: {"label_sampling": p["method"], "mode": "custom"}, _always("mode"),
    "probation", False, "不变", "恢复修改前的方式"))

_register(Operator(
    "gate.mode", "gate", "选择候选在哪一批接受检验",
    "in_window：当批答案一半训练一半检验，过门槛才试用；next_window：当批答案全部用于训练，候选直接试用，由下一批确认或回退。",
    (Param("mode", "enum", "检验时机", choices=("in_window", "next_window"), default="next_window"),),
    lambda p: {"gate_mode": p["mode"], "mode": "custom"}, _always("mode"),
    "probation", False, "next_window 每批只训练一个候选", "恢复修改前的方式"))

_register(Operator(
    "train.trigger", "train", "选择什么时候训练",
    "every_window：每批都可能训练；on_completion：只在有设备（实体）跑完时训练，并在跑完的设备上检验。",
    (Param("trigger", "enum", "训练时机", choices=("every_window", "on_completion"), default="on_completion"),),
    lambda p: {"train_trigger": p["trigger"], "mode": "custom"}, _always("mode"),
    "probation", False, "on_completion 训练次数大幅减少", "恢复修改前的方式",
    requires={"has_entity": True}, requires_doc="数据有设备（实体）结构"))

_register(Operator(
    "label.fraction", "label", "调整每批购买标注的比例",
    "每批数据中请求标注的比例。降低能直接节省标注成本，但会减少可用于训练和检验的数据。",
    (_p_float("fraction", "每批请求标注的比例", 0.05, 1.0),),
    lambda p: {"label_fraction": p["fraction"]}, _lower_is_tighter("label_fraction"),
    "unverifiable", False, "标注量随比例同向变化", "恢复修改前的比例",
    example='{"operator": "label.fraction", "params": {"fraction": 0.3}}'))

_register(Operator(
    "data.recent_only", "data", "只用最近 N 批训练",
    "训练数据只取最近 N 个已有答案的批次，不再加入官方训练集。适合旧数据描述的规律已经过时的情况。",
    (_p_int("n", "最近几批", 1, 20, default=2),),
    lambda p: {"recent_windows_only": p["n"]}, _always("improvement"),
    "fixed", False, "训练行数通常减少", "恢复为不限批次",
    example='{"operator": "data.recent_only", "params": {"n": 2}}'))

_register(Operator(
    "data.all_history", "data", "使用全部历史数据训练",
    "取消“只用最近 N 批”，恢复为按配方使用官方训练集和全部历史批次。",
    (), lambda p: {"recent_windows_only": None}, _always("improvement"),
    "fixed", False, "训练行数通常增加", "恢复为修改前的批次限制"))

_register(Operator(
    "data.cap_per_group", "data", "每组训练数据设上限",
    "按某个分组（通常是设备）给训练数据设上限，超出的部分沿时间均匀抽取。防止少数大组主导模型；"
    "同时减少训练量。",
    (Param("group_by", "fields", "分组字段（meta 字段名的列表）", default="$entity"),
     _p_int("n", "每组最多保留的行数", 5, 1_000_000, default="$median_group_size")),
    lambda p: {"cap_per_group": p["n"], "cap_group_by": p["group_by"]}, _always("improvement"),
    "fixed", True, "训练行数减少", "恢复为不设上限",
    requires={"has_entity": True}, requires_doc="数据有设备（实体）结构",
    example='{"operator": "data.cap_per_group", "params": {"group_by": "$entity", "n": "$median_group_size"}}'))

_register(Operator(
    "data.class_balance", "data", "按类别均衡训练数据",
    "给每个类别的训练数据设上限，削减多数类，让稀有类在训练中的分量更大。",
    (_p_int("n", "每类最多保留的行数", 5, 1_000_000, default="$median_class_size"),),
    lambda p: {"cap_per_group": p["n"], "cap_group_by": ["answer"]}, _always("improvement"),
    "fixed", True, "训练行数减少", "恢复为不设上限",
    requires={"task_type": "classification"}, requires_doc="分类任务"))

_register(Operator(
    "data.no_cap", "data", "取消分组上限",
    "取消任何按组或按类的训练数据上限。",
    (), lambda p: {"cap_per_group": None, "cap_group_by": []}, _always("improvement"),
    "fixed", True, "训练行数增加", "恢复修改前的上限"))

_register(Operator(
    "data.min_samples", "data", "调整训练数据量下限",
    "“自适应”配方向前追溯历史批次，直到训练行数达到这个下限；不够时加入官方训练集。",
    (_p_int("n", "最少训练行数", 10, 100_000),),
    lambda p: {"min_samples": p["n"]}, _always("improvement"),
    "fixed", False, "训练行数随下限同向变化", "恢复修改前的下限"))

_register(Operator(
    "train.lr_scale", "train", "调整学习率倍数",
    "所有候选的学习率 = 官方学习率 × 该倍数。小于 1 让续训更稳，大于 1 让适应更快。",
    (_p_float("scale", "学习率倍数", 0.05, 2.0, default=0.5),),
    lambda p: {"lr_scale": p["scale"]}, _always("improvement"),
    "fixed", True, "不变", "恢复修改前的倍数"))

_register(Operator(
    "train.max_epochs", "train", "调整每个候选最多训练几轮",
    "每个候选模型最多连续训练的轮数（每轮单独保存，可从任一轮出发）。",
    (_p_int("n", "最多轮数", 1, 8),),
    lambda p: {"max_epochs": p["n"]}, _always("improvement"),
    "fixed", False, "训练量随轮数同向变化", "恢复修改前的轮数"))

_register(Operator(
    "train.branch_from", "train", "选择候选从哪个模型接着训练",
    "serving：从当前服务的模型接着练（默认）；static：每次都从官方模型重新出发，适合累积适应已经跑偏的情况。",
    (Param("target", "enum", "出发点", choices=("serving", "static"), default="static"),),
    lambda p: {"branch_from": p["target"]}, _always("improvement"),
    "fixed", False, "不变", "恢复修改前的出发点"))

for _slot, _name, _title, _doc in (
        ("loss_and_delta", "code.loss", "替换损失函数", "决定怎样衡量“错了多少”，例如对高估加重惩罚、Huber、标签平滑。"),
        ("sample_weights", "code.sample_weight", "替换样本权重", "决定哪些训练样本更重要，例如稀有类加权、按目标区间均衡。"),
        ("regularizer_gradients", "code.regularizer", "替换正则化", "在梯度上附加正则项，例如 L2，限制权重持续增大。"),
        ("learning_rate", "code.lr_schedule", "替换学习率变化方式", "每一步的学习率如何随累计步数变化，例如逐步衰减。")):
    _register(Operator(
        _name, "code", _title,
        _doc + f" 修改官方训练器的插槽 `{_slot}({', '.join(slot_rules.SIGNATURES[_slot])})`；"
        "插槽保持默认时训练结果与官方逐字节一致。",
        _CODE_PARAMS, lambda p: {}, _always("improvement"),
        "fixed", True, "通常不变；每步训练明显变慢（>2 倍）的实现会被拒绝", "回到修改前的代码版本",
        slot=_slot,
        example=f'{{"operator": "{_name}", "params": {{"variant": "<内置实现名>"}}}}'))

_register(Operator(
    "split.by_group", "split", "按分组拆分训练和检验",
    "检验用的数据整组留出（例如整台设备），候选永远不在自己训练过的设备上被检验；"
    "当批只有一个分组时自动改为按时间拆分。同时让波动范围按组计算。",
    (Param("group_by", "fields", "分组字段", default="$entity"),),
    lambda p: {"split_mode": "by_group", "split_group_by": p["group_by"]}, _split,
    "none", False, "不变", "恢复修改前的拆分方式",
    requires={"has_entity": True}, requires_doc="数据有设备（实体）结构"))

_register(Operator(
    "split.by_order", "split", "按时间拆分训练和检验",
    "当批较早的数据用于训练，较晚的数据用于检验，让候选“往后预测”而不是“填空”。",
    (), lambda p: {"split_mode": "by_order"}, _split,
    "none", False, "不变", "恢复修改前的拆分方式"))

_register(Operator(
    "split.random", "split", "随机拆分训练和检验",
    "随机拆分。只适合每条数据彼此独立的情况；从更严格的拆分放宽到随机需要验证。",
    (), lambda p: {"split_mode": "random", "split_group_by": []}, _split,
    "full", False, "不变", "恢复修改前的拆分方式"))

_register(Operator(
    "gate.z", "gate", "调整晋升门槛的严格程度",
    "候选的提升必须超过 z 倍波动范围才能晋升。调高更保守；调低更容易晋升，需要验证。",
    (_p_float("value", "z 值", 1.0, 4.0, default=2.5),),
    lambda p: {"z": p["value"]}, _higher_is_tighter("z"),
    "full", False, "不变", "恢复修改前的 z"))

_register(Operator(
    "gate.min_units", "gate", "调整晋升所需的最少独立单位",
    "检验数据里独立单位（如设备）少于该值时不做晋升决定。调高更保守。",
    (_p_int("n", "最少独立单位数", 2, 50, default=8),),
    lambda p: {"min_gate_units": p["n"]}, _higher_is_tighter("min_gate_units"),
    "full", False, "不变", "恢复修改前的值"))

_register(Operator(
    "gate.confirm", "gate", "调整试用转正所需批数",
    "试用模型要连续多少批不被否决才转正。调高更保守。",
    (_p_int("n", "批数", 1, 5, default=3),),
    lambda p: {"confirm_windows": p["n"]}, _higher_is_tighter("confirm_windows"),
    "full", False, "不变", "恢复修改前的批数"))

_register(Operator(
    "gate.cooldown", "gate", "调整回退后的冷却期",
    "回退或退回基线后，多少批内不再训练。调高更保守。",
    (_p_int("n", "批数", 0, 5, default=2),),
    lambda p: {"cooldown_windows": p["n"]}, _higher_is_tighter("cooldown_windows"),
    "full", False, "冷却期内不训练，训练量减少", "恢复修改前的批数"))

for _mode, _title, _doc, _cost in (
        ("trial", "试用后上线", "通过门槛的候选立即试用，下一批确认或回退。适应最快。", "不变"),
        ("shadow", "影子对比后上线", "通过门槛的候选先不上线，下一批在新数据上显著更好才上线。当批检验不可靠时使用。", "不变"),
        ("freeze", "暂停更新", "不再训练，一直使用当前模型。更新没有收益时用来节省成本；环境突变或失守时会重新分析。",
         "训练量降为 0")):
    _register(Operator(
        f"serve.{_mode}", "serve", _title, _doc, (), (lambda m: lambda p: {"serve_mode": m})(_mode), _serve,
        "full", False, _cost, "恢复修改前的上线方式"))

_register(Operator(
    "supervise.tune", "supervise", "调整监督节奏",
    "观察期长度、定期复查间隔、健康检查回看的批数。只影响何时复查和更新，不影响模型。",
    (Param("probation_windows", "int", "观察期批数", 1, 10, optional=True),
     Param("review_every", "int", "定期复查间隔（批）", 2, 50, optional=True),
     Param("health_window", "int", "健康检查回看批数", 2, 10, optional=True)),
    lambda p: {k: v for k, v in p.items() if v is not None}, _always("meta"),
    "none", False, "不变", "恢复修改前的值"))


# ------------------------------------------------------------------------ instantiation

def instantiate(name: str, params: dict | None, sig: dict, policy: PolicyConfig,
                meta_fields: set[str] | None = None) -> Instance:
    """Resolve placeholders and defaults, check every parameter against the manual, and compute the
    exact strategy change. Raises OperatorError with a message the caller can act on."""
    op = REGISTRY.get(name)
    if op is None:
        raise OperatorError(f"unknown operator {name!r}; see the manual for the list")
    if op.requires and not sig_rules.holds(op.requires, sig):
        raise OperatorError(f"{name} requires: {op.requires_doc}")
    given = dict(params or {})
    unknown = set(given) - {p.name for p in op.params}
    if unknown:
        raise OperatorError(f"{name}: unknown parameters {sorted(unknown)}")
    resolved: dict = {}
    for param in op.params:
        value = given.get(param.name, param.default)
        try:
            value = sig_rules.resolve(value, sig)
        except ValueError as error:
            raise OperatorError(f"{name}.{param.name}: {error}")
        if value is None:
            if param.optional or param.kind in ("variant", "source"):
                continue
            raise OperatorError(f"{name}: parameter {param.name} is required")
        resolved[param.name] = _check(name, param, value, meta_fields)
    edits: list[dict] = []
    if op.slot:
        edits = [_slot_edit(op, resolved, sig)]
        resolved = {k: v for k, v in resolved.items() if k != "source"}
    changes = op.effect(resolved)
    kind = op.direction(policy, changes) if changes else "improvement"
    return Instance(op, resolved, changes, edits, kind)


def _check(name: str, param: Param, value, meta_fields: set[str] | None):
    if param.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
            raise OperatorError(f"{name}.{param.name} must be an integer")
        value = int(value)
    elif param.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperatorError(f"{name}.{param.name} must be a number")
        value = float(value)
    if param.kind in ("int", "float") and not param.low <= value <= param.high:
        raise OperatorError(f"{name}.{param.name}={value} outside [{param.low}, {param.high}]")
    if param.kind == "enum" and value not in param.choices:
        raise OperatorError(f"{name}.{param.name} must be one of {list(param.choices)}")
    if param.kind == "fields":
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise OperatorError(f"{name}.{param.name} must be a non-empty list of field names")
        if meta_fields is not None and value != ["answer"]:
            missing = [v for v in value if v not in meta_fields]
            if missing:
                raise OperatorError(f"{name}.{param.name}: fields {missing} do not exist; available {sorted(meta_fields)}")
    if param.kind in ("variant", "source") and not isinstance(value, str):
        raise OperatorError(f"{name}.{param.name} must be a string")
    return value


def _slot_edit(op: Operator, params: dict, sig: dict) -> dict:
    if params.get("variant"):
        match = [v for v in slot_rules.VARIANTS if v.name == params["variant"] and v.slot == op.slot]
        if not match:
            names = [v.name for v in slot_rules.VARIANTS if v.slot == op.slot]
            raise OperatorError(f"{op.name}: unknown variant {params['variant']!r}; available {names}")
        if sig.get("task_type") not in match[0].tasks:
            raise OperatorError(f"{op.name}: variant {params['variant']} does not apply to {sig.get('task_type')}")
        return match[0].edit()
    if params.get("source"):
        return {"slot": op.slot, "variant": "custom", "hypothesis": params.get("hypothesis", ""),
                "source": params["source"]}
    raise OperatorError(f"{op.name}: give either a variant or a source")


def applicable(sig: dict) -> list[Operator]:
    return [op for op in REGISTRY.values() if not op.requires or sig_rules.holds(op.requires, sig)]


# ------------------------------------------------------------------------ rendering

def _param_text(param: Param, sig: dict | None) -> str:
    bits = [param.name]
    if param.kind in ("int", "float"):
        bits.append(f"{param.kind} [{param.low:g}, {param.high:g}]")
    elif param.kind == "enum":
        bits.append("|".join(param.choices))
    else:
        bits.append(param.kind)
    if param.default is not None:
        default = param.default
        if sig is not None:
            try:
                default = sig_rules.resolve(default, sig)
            except ValueError:
                pass
        bits.append(f"默认 {default}")
    if param.optional:
        bits.append("可选")
    return " ".join(str(b) for b in bits)


def prompt_manual(sig: dict) -> str:
    """Compact manual for the LLM: only operators that apply to this task, defaults resolved."""
    lines = []
    for op in applicable(sig):
        params = "; ".join(_param_text(p, sig) for p in op.params) or "无参数"
        kind = {"none": "收紧/立即", "probation": "节奏/立即生效+观察期"}.get(op.validation, op.validation + " 验证")
        lines.append(f"- {op.name}（{op.title}）参数: {params} | 类型: {kind} | {op.doc}")
    variants = [f"{v.slot}:{v.name}" for v in slot_rules.VARIANTS if sig.get("task_type") in v.tasks]
    lines.append("可用内置实现: " + ", ".join(variants))
    return "\n".join(lines)


def manual_markdown() -> str:
    lines = ["# 修改说明书（算子手册）", "",
             "系统改变“怎样训练、采用哪个模型、怎样上线”的**唯一**方式，是调用下列算子。经验库、确定性兜底分析器和大模型"
             "都用这套名字，监督器按本手册校验参数、判断收紧或改进、选择验证方式并执行。本文件由 "
             "`src/agent_harness/operators.py` 自动生成，代码即手册。", "",
             "## 不变的部分", "",
             "官方训练器的核心不是算子，永远不变：网络结构（单隐层、tanh、宽度）、数据标准化参数、Adam 常数、梯度与特征裁剪、"
             "训练循环、数据解析、检查点格式、随机种子。", "",
             "## 生效规则", ""]
    lines += [f"- **{key}**：{text}" for key, text in VALIDATIONS.items()]
    lines += ["- **收紧 / 改进的判定**：同一个算子按方向决定，例如 `gate.z` 调高是收紧、调低是改进；"
              "`serve.*` 按 试用 < 影子 < 暂停 的顺序，往后是收紧。",
              "- **占位符**：`$entity`（识别出的设备字段）、`$order`（顺序字段）、`$median_group_size`（中位设备行数）、"
              "`$median_class_size`（中位类别行数），在每个任务上按其数据自动填值，所以同一条经验可以跨任务套用。",
              "- **撤销**：每次修改生成一个策略版本；观察期不通过时整体退回上一版本。",
              "- **节奏（`mode.*`）**：一次设定检验时机、训练时机、上线方式和标注抽取方式。三种节奏对应三类数据——"
              "答案及时且输入在变（积极跟随）、答案要等设备跑完（设备完成时更新）、输入不变且更新无收益（节省）。"
              "选哪一种由经验库里带条件的经验根据情境特征决定（`configs/experiences/seed.yaml`），运行中特征变化时"
              "监督器触发“情境与节奏不符”（T10）重新选择。每轮最多切换一次节奏。", "",
              "## 总览", "",
              "| 算子 | 环节 | 作用 | 验证 | 开局可验证 | 适用 |", "|---|---|---|---|---|---|"]
    for op in REGISTRY.values():
        lines.append(f"| `{op.name}` | {LAYERS[op.layer]} | {op.title} | {op.validation} | "
                     f"{'是' if op.opening else '否'} | {op.requires_doc} |")
    for layer, layer_title in LAYERS.items():
        ops = [op for op in REGISTRY.values() if op.layer == layer]
        if not ops:
            continue
        lines += ["", f"## {layer_title}", ""]
        for op in ops:
            lines += [f"### `{op.name}` —— {op.title}", "", op.doc, "",
                      "| 项 | 内容 |", "|---|---|",
                      f"| 参数 | {'<br>'.join(f'`{p.name}`（{_param_text(p, None)}）：{p.doc}' for p in op.params) or '无'} |",
                      f"| 收紧还是改进 | {_direction_doc(op)} |",
                      f"| 验证 | {VALIDATIONS[op.validation]}{'；开局可用训练集按组 k 折验证' if op.opening else ''} |",
                      f"| 成本影响 | {op.cost} |", f"| 撤销 | {op.undo} |", f"| 适用条件 | {op.requires_doc} |"]
            if op.example:
                lines.append(f"| 调用示例 | `{op.example}` |")
            lines.append("")
    lines += ["## 训练代码插槽的内置实现", "", "| 插槽 | 名字 | 适用任务 | 假设 |", "|---|---|---|---|"]
    lines += [f"| `{v.slot}` | `{v.name}` | {', '.join(v.tasks)} | {v.hypothesis} |" for v in slot_rules.VARIANTS]
    return "\n".join(lines) + "\n"


def _direction_doc(op: Operator) -> str:
    probe = {"label.fraction": "调低为收紧，调高为改进", "split.random": "放宽为改进（需验证）",
             "split.by_group": "收紧", "split.by_order": "收紧", "supervise.tune": "节奏类，立即生效"}
    if op.name in probe:
        return probe[op.name]
    if op.validation == "probation":
        return "节奏：立即生效，观察期判定"
    if op.layer == "gate":
        return "调高为收紧，调低为改进"
    if op.layer == "serve":
        return "向“暂停”方向为收紧，反向为改进"
    return "改进（需验证）"
