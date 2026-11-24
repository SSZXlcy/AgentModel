"""Deterministic explorer (W5): the same explore → plan loop, without an LLM.

It reads the situation signature and calls the same tools the agent would, and writes its plan as
operator calls from the manual (operators.py), using placeholders such as $entity. At the opening
and whenever the rhythm no longer fits (T10) its corrections are the experiences whose conditions
hold (experiences.py): the most credible rhythm plus every applicable data experience. Every rule is
a statistic: no task names and no field names appear here. It guarantees a floor, it backfills the
LLM plan, and it is the control group for the LLM explorer.
"""
from __future__ import annotations

from . import signature as sig_rules
from .agent import local_proposal


def _fruitless_rounds(ctrl) -> int:
    """Earlier no-return / stall rounds in this run that changed nothing."""
    memory = ctrl.supervisor.memory if ctrl.supervisor else None
    if memory is None:
        return 0
    return sum(1 for r in memory.rounds() if r.get("trigger") in ("T5", "T6") and not r.get("applied"))


def explore(box, trigger: str, ctrl, rhythm=None, experiences=(), fired=()) -> dict:
    p = ctrl.policy
    sig = sig_rules.compute(ctrl)
    findings, hypotheses, corrections, risks = [], [], [], []

    def add(operator: str, params: dict, why: str, validation: dict | None = None, experience: str | None = None):
        corrections.append({"id": f"C{len(corrections) + 1}", "operator": operator, "params": params,
                            "validation": validation, "why": why, **({"experience": experience} if experience else {})})

    def follow(e) -> None:
        add(e.operator, dict(e.action.get("params") or {}), f"{e.id} (credibility {e.credibility:.2f}): {e.why}",
            experience=e.id)

    def timely() -> bool:
        return sig.get("label_timeliness") is None or sig["label_timeliness"] >= 0.5

    def follows_fresh_answers() -> bool:
        """Training on every fresh answer at once suits rows that are independent observations; rows of an
        entity that is still running would mostly teach the model that one entity."""
        return timely() and (not sig["has_entity"] or sig.get("stream_format") == "snapshot")

    if trigger == "T1":
        box.call("schema", {})
        box.call("find_structure", {"scope": "base"})
        if sig["has_entity"]:
            findings.append({"claim": f"rows form {sig['n_entities']} entities keyed by $entity, ordered by $order",
                             "evidence": "signature / find_structure"})
            if sig["countdown"]:
                findings.append({"claim": "inside each entity the answer counts down to 0: it exists only after the "
                                          "entity has finished, and rows of one entity are strongly correlated",
                                 "evidence": "signature.countdown"})
            box.call("group_stats", {"group_by": sig["entity"], "scope": "base"})
        else:
            findings.append({"claim": "no repeating entity with an order was found: rows look independent",
                             "evidence": "signature / find_structure"})
        if sig["task_type"] == "classification" and (sig.get("class_imbalance") or 1.0) > 3.0:
            box.call("group_stats", {"group_by": ["answer"], "scope": "base"})
        box.call("training_curve", {"node": "official"})
        for i, e in enumerate(([rhythm] if rhythm else []) + list(experiences)):
            hypotheses.append({"id": f"H{i + 1}", "statement": e.why, "expected_effect": f"{e.operator} helps here",
                               "experience": e.id})
            follow(e)
        risks.append("opening experiments train from scratch on the training set; a validated procedure is refit "
                     "before the stream, a rhythm is judged by the probation that follows")
    elif trigger == "T10":
        findings.append({"claim": f"the rhythm in use ({p.mode}) no longer matches the situation: answers arriving in "
                                  f"the window {sig.get('label_timeliness')}, exploitable drift "
                                  f"{sig.get('drift_exploitable')}", "evidence": "signature"})
    elif trigger == "T2":
        findings.append({"claim": "the serving model fell below the static model on fresh labels", "evidence": "health H1"})
        add("serve.shadow", {}, "candidates must win beside the serving model before they serve")
    elif trigger == "T3":
        findings.append({"claim": "promotions did not hold on the next window: the gate is optimistic",
                         "evidence": "health H2"})
        if p.gate_mode == "next_window" and follows_fresh_answers():
            # independent rows: the rhythm itself decides (drift_exploitable); halving the answers for an
            # in-window gate was the slow choice that lost to plain continuous training
            findings.append({"claim": "the adaptive rhythm rolls back often; whether it earns anything is judged by "
                                      "the settled gains", "evidence": "health H2"})
        elif p.gate_mode == "next_window":
            add("mode.standard", {}, "candidates trained on every answer keep failing the next window: judge them "
                                     "in the window again before they serve")
        elif p.split_mode == "random":
            add("split.by_order", {}, "judge on the latest rows instead of interleaved ones")
        elif p.serve_mode == "trial":
            add("serve.shadow", {}, "in-window checks keep overstating gains: a candidate must first win on the next "
                                    "window's fresh rows")
        else:
            add("gate.z", {"value": round(min(p.z + 0.5, 4.0), 3)}, "widen the band a gain must clear")
    elif trigger == "T4":
        box.call("compare_subsets", {"a": "base", "b": f"window:{ctrl.state['next_window'] - 1}"})
        findings.append({"claim": "input shift larger than anything seen so far in this run", "evidence": "health H3"})
        if p.serve_mode == "freeze" and sig.get("inputs_covered"):
            findings.append({"claim": "the shift stays inside the region the training data covers "
                                      f"(input novelty {sig.get('input_novelty')}): keep saving", "evidence": "signature"})
        elif p.serve_mode == "freeze":
            if follows_fresh_answers():
                add("mode.adaptive", {}, "a new regime appeared and answers arrive in time: follow it again")
            elif sig["has_entity"] and not timely():
                add("mode.event", {}, "a new regime appeared: learn again when entities finish")
            elif sig["has_entity"]:
                add("mode.standard", {}, "a new regime appeared: judge candidates on held-out entities again")
            else:
                add("serve.shadow", {}, "a new regime appeared: allow candidates again, but only after they win "
                                        "beside the serving model")
        if not sig.get("inputs_covered"):
            add("data.recent_only", {"n": 2}, "old data may describe a regime that no longer holds")
    elif trigger == "T5":
        cmp = box.call("compare_subsets", {"a": "base", "b": "stream_paid"})
        no_shift = cmp.get("mmd", 0.0) <= cmp.get("mmd_null_threshold", 0.0)
        findings.append({"claim": "training has cost work without settled gains" +
                                  (" and the stream looks like the training data" if no_shift else ""),
                         "evidence": "health H4, compare_subsets"})
        if no_shift:
            add("mode.conserve", {}, "nothing in the stream for an update to learn: stop paying for training")
        else:
            add("serve.shadow", {}, "only adopt candidates that win beside serving")
    elif trigger == "T6":
        status = box.call("run_status", {})
        findings.append({"claim": "drift persists but no candidate passes the gate", "evidence": "run_status"})
        if p.gate_mode == "in_window" and p.train_trigger == "every_window" and follows_fresh_answers():
            # The stream moves but in-window gates keep refusing: half the answers are spent on judging and the
            # rest is too little to adapt. Train on all of them and let the next window judge.
            add("mode.adaptive", {}, "the stream keeps moving and in-window gates refuse every candidate: train on "
                                     "every fresh answer and let the next window confirm or roll back")
        errors = box.call("error_breakdown", {"by": "class" if sig["task_type"] == "classification" else "answer_bins",
                                              "scope": "stream_paid"})
        evidence = {"errors": {}, "diagnoses": {status.get("last_diagnosis") or "": 1}}
        if sig["task_type"] != "classification" and "bins" in errors:
            total = max(sum(b["n"] for b in errors["bins"]), 1)
            evidence["errors"]["over_estimate_ratio"] = sum(b["n"] for b in errors["bins"] if b["mean_error"] > 0) / total
        if sig["task_type"] == "classification" and sig.get("class_imbalance"):
            evidence["errors"]["class_imbalance"] = sig["class_imbalance"]
        tried = ctrl.supervisor.memory.tried_variants() if ctrl.supervisor and ctrl.supervisor.memory else []
        slot_operators = {"loss_and_delta": "code.loss", "sample_weights": "code.sample_weight",
                          "regularizer_gradients": "code.regularizer", "learning_rate": "code.lr_schedule"}
        for group in ("PHM", "OPT"):
            for edit in local_proposal(group, 1, sig["task_type"], evidence, tried):
                add(slot_operators[edit["slot"]], {"variant": edit["variant"]}, edit.get("hypothesis", ""))
        add("data.recent_only", {"n": 2}, "adapt to the latest regime only")
        add("train.lr_scale", {"scale": 0.5}, "continued training may be overshooting; halve the learning rate")
    elif trigger == "T7":
        add("label.fraction", {"fraction": round(max(0.05, p.label_fraction * 0.7), 3)},
            "spending runs ahead of progress; buy fewer labels")
    if rhythm is not None and trigger != "T1" and ("T10" in fired or trigger == "T10") \
            and rhythm.operator != f"mode.{p.mode}" and not any(c["operator"].startswith("mode.") for c in corrections):
        follow(rhythm)
    return {"findings": findings, "hypotheses": hypotheses, "corrections": corrections, "risks": risks}
