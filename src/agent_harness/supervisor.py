"""Supervisor (W0): when to keep the strategy, when to review it, when to update it, and when the
agent steps in.

Every window: five health checks computed from the run's own records (no LLM, no task knowledge).
Triggers lift the tier from *continue* to *review* (one cheap LLM call, or a rule) or *update*
(a full explore → plan → validate → decide → summarize round). After an update the strategy is
on probation for M windows; it is then confirmed or reverted, and the verdict is back-filled into
the journal and into the statistics of the experiences the round applied, so later rounds and runs
learn which kinds of change really held.

The rhythm (mode.*) is chosen from experiences whose conditions hold on the situation signature;
when the signature changes so that another rhythm fits (answers turn out to arrive late, the drift
turns out to be unexploitable, ...), T10 starts a round that switches it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import codebase, experiments, explorer, fallback_explorer, operators, sandbox, signature, structure
from . import slots as slot_rules
from .agent import Agent
from .config import PolicyConfig
from .contracts import Cost
from .experiences import Experiences, mode_of
from .memory import Memory
from .tools import Toolbox

TRIGGERS = {
    "T1": ("开局", "update"), "T2": ("失守：低于静态基线", "update"), "T3": ("门槛失准", "update"),
    "T4": ("环境突变", "update"), "T5": ("投入无回报", "review"), "T6": ("有漂移但停滞", "review"),
    "T7": ("预算转折", "review"), "T8": ("定期复查", "review"), "T9": ("结束", "summary"),
    "T10": ("情境与节奏不符", "update"),
}
SEVERITY = ["T2", "T3", "T4", "T1", "T10", "T6", "T5", "T7", "T8"]
NAMED_MODES = ("standard", "adaptive", "event", "conserve")
KIND_TEXT = {"tightening": "收紧，立即生效", "meta": "节奏参数，立即生效", "mode": "切换节奏，观察期判定",
             "improvement": "验证通过"}


class Supervisor:
    def __init__(self, agent: Agent | None, memory_root: Path, code_store: Path, *, use_llm: bool,
                 seed_experiences: Path | None = None):
        self.agent = agent or Agent()
        self.memory_root = Path(memory_root)
        self.code_store = Path(code_store)
        self.use_llm = use_llm and self.agent.provider is not None
        self.seed_experiences = seed_experiences
        self.memory: Memory | None = None
        self.experiences: Experiences | None = None
        self._known_failures: set[str] | None = None
        self._previous: dict | None = None
        self._repair_tokens = 0

    # ================================================================ hooks
    def _state(self, ctrl) -> dict:
        sv = ctrl.state.setdefault("supervisor", {})
        sv.setdefault("tiers", {"continue": 0, "review": 0, "update": 0})
        sv.setdefault("triggers", {})
        sv.setdefault("interventions", [])
        sv.setdefault("updates", 0)
        sv.setdefault("last_review", -1)
        sv.setdefault("probation", None)
        sv.setdefault("budget_marks", [])
        sv.setdefault("review_escalations", {})
        sv.setdefault("outcomes", {"confirmed": 0, "reverted": 0, "inconclusive": 0})
        sv.setdefault("mode_switches", 0)
        sv.setdefault("mode_since", 0)          # first window served under the current rhythm
        sv.setdefault("mode_held", False)       # its probation confirmed it and it did not fall below static
        sv.setdefault("blocked_experiences", [])
        if self.memory is None:
            self.memory = Memory(ctrl.run_dir, self.memory_root, ctrl.task.guarded_markers, ctrl.run_dir.name, ctrl.task.task)
            self.experiences = Experiences(self.memory_root, ctrl.run_dir.name, self.seed_experiences)
        return sv

    def choose(self, ctrl, sig: dict, before_round: int | None = None):
        """(rhythm experience, other applicable experiences) for this signature."""
        sv = self._state(ctrl)
        return self.experiences.choose(sig, self.memory.library, before_round, set(sv["blocked_experiences"]))

    def _mode_mismatch(self, ctrl, emergency: bool = False) -> bool:
        """T10: another rhythm fits the current signature better than the one in use.

        A rhythm is given time to show what it does (at least `probation_windows` windows), and a rhythm
        whose probation confirmed it stays unless the run falls below the static model: switching away from
        something that works on thin evidence was the costliest mistake seen so far."""
        if not hasattr(ctrl, "session") or ctrl.policy.mode not in NAMED_MODES:
            return False                       # "custom": the agent has taken the rhythm in hand
        sv = self._state(ctrl)
        if sv["mode_switches"] >= ctrl.policy.max_updates:
            return False
        if ctrl.state["next_window"] - sv["mode_since"] < ctrl.policy.probation_windows:
            return False
        if sv["mode_held"] and not emergency:
            return False
        rhythm, _ = self.choose(ctrl, signature.compute(ctrl))
        wanted = mode_of(rhythm)
        return wanted is not None and wanted != ctrl.policy.mode

    def opening(self, ctrl) -> None:
        sv = self._state(ctrl)
        sv["tiers"]["update"] += 1
        self._round(ctrl, "T1", ["T1"], {})

    def after_window(self, ctrl, k: int) -> None:
        sv, p = self._state(ctrl), ctrl.policy
        health = self.health(ctrl)
        fired = self.triggers(ctrl, health, k)
        probation = sv["probation"]
        if probation and k + 1 >= probation["until"]:
            self._judge_probation(ctrl, probation)
            sv["probation"] = None
        if sv["probation"] is not None:
            # Only the emergency interrupts a probation, and evidence about the rhythm interrupts the
            # opening one: it was chosen before a single window was seen.
            allowed = {"T2"} | ({"T10"} if sv["probation"].get("trigger") == "T1" else set())
            fired = [t for t in fired if t in allowed]
            if "T10" in fired:
                sv["outcomes"]["inconclusive"] += 1
                self.memory.backfill(sv["probation"]["round"], {"probation": "superseded"})
                sv["probation"] = None
        for t in fired:
            sv["triggers"][t] = sv["triggers"].get(t, 0) + 1
        tier = "continue"
        if any(TRIGGERS[t][1] == "update" for t in fired):
            tier = "update"
        elif fired:
            tier = "review"
        decision_note = ""
        if tier == "review":
            sv["last_review"] = k
            decision, decision_note = self._review(ctrl, health, fired)
            tier = "update" if decision == "explore" else "review"
        rhythm_only = [t for t in fired if TRIGGERS[t][1] == "update"] == ["T10"]
        if tier == "update" and sv["updates"] >= p.max_updates and not rhythm_only:
            decision_note = f"update wanted but max_updates={p.max_updates} reached"
            tier = "review"
        sv["tiers"][tier if tier != "update" else "update"] += 1
        ctrl.recorder.event("supervisor", window=k, tier=tier, triggers=fired, note=decision_note,
                            health={key: health[key] for key in ("H1", "H2", "H3", "H4", "H5")})
        if tier == "update":
            main = next(t for t in SEVERITY if t in fired) if fired else "T8"
            self._round(ctrl, main, fired, health)

    def closing(self, ctrl) -> None:
        sv = self._state(ctrl)
        mem = self.memory
        rounds = mem.rounds()
        stats = {"windows": len(ctrl.state["rows"]), "tiers": sv["tiers"], "triggers": sv["triggers"],
                 "updates": sv["updates"], "mode_switches": sv["mode_switches"], "mode": ctrl.policy.mode,
                 "outcomes": sv["outcomes"], "promotions": ctrl.state["promotions"],
                 "rollbacks": ctrl.state["rollbacks"], "strategy_versions": ctrl.state["strategy_version"]}
        payload = {"trigger": "T9 结束", "run_stats": stats,
                   "rounds": [{"round": r["round"], "trigger": r.get("trigger"), "applied": r.get("applied"),
                               "outcome": r.get("outcome"), "lessons": (r.get("summary") or {}).get("lessons")}
                              for r in rounds]}
        # one closing summary by the LLM when the run budget still allows it: it turns the run into lessons
        summary, tokens = (self.agent.summarize_round(payload) if self._llm_allowed(ctrl, 4000) else (None, 0))
        if tokens:
            ctrl.ledger.charge(Cost(llm_tokens=tokens))
        if summary is None:
            summary = _closing_template(stats, rounds)
        mem.append({"round": mem.next_round(), "trigger": "T9", "window": ctrl.state["next_window"] - 1,
                    "source": "llm" if self.use_llm and tokens else "template", "summary": summary, "stats": stats,
                    "tokens": tokens})
        ctrl.state["supervisor_summary"] = {**stats, "interventions": sv["interventions"],
                                            "llm": self.agent.status()}

    # ================================================================ health and triggers
    def health(self, ctrl) -> dict:
        p, rows = ctrl.policy, ctrl.state["health"]
        k = p.health_window
        recent = rows[-k:]
        last_two = [r.get("vs_static", {}).get("worse") for r in rows[-2:]]
        # Small losses that are not significant in any one window still add up: also test the
        # mean over the last k windows against the combined band of those windows.
        vs = [r["vs_static"] for r in recent if r.get("vs_static")]
        cumulative_bad = False
        if len(vs) >= k and any(v["diff"] != 0 for v in vs):
            combined = float(np.sqrt(sum(v["band"] ** 2 for v in vs))) / len(vs)
            cumulative_bad = float(np.mean([v["diff"] for v in vs])) < -combined
        h1 = {"ok": not ((len(last_two) == 2 and all(last_two)) or cumulative_bad),
              "last_diff": rows[-1].get("vs_static", {}).get("diff") if rows else None,
              "recent_mean_diff": float(np.mean([v["diff"] for v in vs])) if vs else None,
              "cumulative_loss": cumulative_bad}
        settled = [r["settled"] for r in rows[-2 * k:] if r.get("settled") and r["settled"].get("changed")
                   and r["settled"].get("predicted") is not None]
        realized = [s["gain"] for s in settled]
        predicted = [s["predicted"] for s in settled]
        rollbacks = sum(1 for r in rows[-2 * k:] if r.get("rollback"))
        promotions = sum(1 for r in rows[-2 * k:] if r.get("promoted"))
        h2_bad = (len(settled) >= 2 and np.mean(realized) < 0.25 * np.mean(predicted)) or \
                 (promotions >= 2 and rollbacks >= 0.5 * promotions)
        h2 = {"ok": not h2_bad, "settled": len(settled), "realized_mean": float(np.mean(realized)) if realized else None,
              "predicted_mean": float(np.mean(predicted)) if predicted else None, "rollbacks": rollbacks,
              "promotions": promotions}
        drifts = [r["drift"]["mmd"] for r in rows if r.get("drift")]
        h3_bad = False
        if len(drifts) >= 4 and rows[-1].get("drift"):
            current, before = drifts[-1], drifts[:-1]
            h3_bad = current > 1.5 * max(before) and rows[-1]["drift"]["shifted"]
        h3 = {"ok": not h3_bad, "current": drifts[-1] if drifts else None, "max_before": max(drifts[:-1]) if len(drifts) > 1 else None}
        trained = [r for r in recent if r.get("trained")]
        # Return = what changes of the serving model actually earned on the next window, not how
        # often a candidate was promoted.
        realized_recent = [r["settled"]["gain"] for r in recent if r.get("settled") and r["settled"].get("changed")]
        no_return = len(recent) >= k and len(trained) == k and sum(realized_recent) <= 0
        shifted = [r["drift"]["shifted"] for r in recent if r.get("drift")]
        h4 = {"ok": not no_return, "trained": len(trained), "promoted": sum(1 for r in recent if r.get("promoted")),
              "realized_gain": float(sum(realized_recent)),
              "mostly_shifted": bool(shifted) and sum(shifted) > len(shifted) / 2}
        spent = ctrl.ledger.spent.train_seconds
        limit = ctrl.ledger.max_train_seconds
        progress = len(ctrl.state["rows"]) / max(len(ctrl.window_ids), 1)
        spent_fraction = spent / limit if limit else 0.0
        h5 = {"ok": not (limit and spent_fraction > progress + 0.25), "spent_fraction": spent_fraction,
              "progress": progress}
        return {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}

    def triggers(self, ctrl, health: dict, k: int) -> list[str]:
        sv, p = self._state(ctrl), ctrl.policy
        fired = []
        if not health["H1"]["ok"]:
            fired.append("T2")
        if not health["H2"]["ok"]:
            fired.append("T3")
        if not health["H3"]["ok"]:
            fired.append("T4")
        # Under the adaptive rhythm every window trains once by design; whether that earns anything is the
        # rhythm's own question (drift_exploitable → T10), not a stall to repair with experiments.
        if not health["H4"]["ok"] and ctrl.policy.gate_mode != "next_window":
            fired.append("T6" if health["H4"]["mostly_shifted"] else "T5")
        if self._mode_mismatch(ctrl, emergency="T2" in fired):
            fired.append("T10")
        remaining = 1 - health["H5"]["spent_fraction"]
        marks = [m for m in (0.5, 0.2) if remaining <= m and m not in sv["budget_marks"] and ctrl.ledger.max_train_seconds]
        if not health["H5"]["ok"] or marks:
            sv["budget_marks"] += marks
            fired.append("T7")
        if not fired and k - sv["last_review"] >= p.review_every:
            fired.append("T8")
        return fired

    def _review(self, ctrl, health: dict, fired: list[str]) -> tuple[str, str]:
        sv = self._state(ctrl)
        for t in ("T5", "T6"):
            sv["review_escalations"][t] = sv["review_escalations"].get(t, 0) + 1 if t in fired else 0
        repeat = any(t in fired for t in ("T5", "T6")) and fallback_explorer._fruitless_rounds(ctrl) >= 1
        rule = "explore" if repeat or any(sv["review_escalations"].get(t, 0) >= 2 for t in ("T5", "T6")) else "continue"
        # A review is a rule decision unless the rules have already failed in this run (tokens are scored).
        if self._llm_allowed(ctrl, 4000) and self._rules_failed(ctrl):
            payload = {"triggers": [f"{t} {TRIGGERS[t][0]}" for t in fired], "health": health,
                       "strategy": {k: v for k, v in ctrl.state["policy"].items() if k in experiments.EXPERIMENT_FIELDS},
                       "journal": self.memory.prompt_view(last=2)["journal"]}
            decision, why, tokens = self.agent.review(payload)
            if tokens:
                ctrl.ledger.charge(Cost(llm_tokens=tokens))
            if decision:
                sv["interventions"].append({"window": ctrl.state["next_window"] - 1, "tier": "review", "triggers": fired,
                                            "source": "llm", "decision": decision, "why": why, "tokens": tokens})
                return decision, why
        sv["interventions"].append({"window": ctrl.state["next_window"] - 1, "tier": "review", "triggers": fired,
                                    "source": "rule", "decision": rule})
        return rule, "rule: escalate after the same no-return/stall trigger twice in a row"

    # ================================================================ one round
    def _round(self, ctrl, trigger: str, fired: list[str], health: dict) -> None:
        sv, p, mem = self._state(ctrl), ctrl.policy, self.memory
        r = mem.next_round()
        window = ctrl.state["next_window"] - 1
        box = Toolbox(ctrl, mem, ctrl.run_dir / "scratch" / f"r{r:02d}", self.code_store)
        sig = signature.compute(ctrl)
        rhythm, others = self.choose(ctrl, sig, before_round=r)
        matched = ([rhythm] if rhythm else []) + others
        context = {"trigger": {"code": trigger, "meaning": TRIGGERS[trigger][0], "all_fired": fired},
                   "window": window, "windows_total": len(ctrl.window_ids), "signature": sig, "health": health,
                   "current_strategy": {"policy": {k: v for k, v in ctrl.state["policy"].items()
                                                   if k in experiments.EXPERIMENT_FIELDS},
                                        "code_version": ctrl.state["code_version"]},
                   "experiences": {"recommended_rhythm": rhythm.view() if rhythm else None,
                                   "other_matching": [e.view() for e in others]},
                   "memory": mem.prompt_view(before_round=r), "budget": ctrl.ledger.remaining()}
        plan, trace, tokens, note, source = None, [], 0, "", "fallback"
        wanted, why_llm = self._llm_wanted(ctrl, trigger, rhythm)
        if wanted and self._llm_allowed(ctrl, p.explore_tokens):
            budget = min(p.explore_tokens, p.run_llm_tokens - ctrl.ledger.spent.llm_tokens)
            plan, trace, tokens, note = explorer.explore(self.agent, box, context, p.explore_steps, budget)
            if tokens:
                ctrl.ledger.charge(Cost(llm_tokens=tokens))
            if plan is not None:
                source = "llm"
        elif self.use_llm:
            note = f"rules only: {why_llm}"
        rule_plan = fallback_explorer.explore(
            Toolbox(ctrl, mem, ctrl.run_dir / "scratch" / f"r{r:02d}_rule", self.code_store), trigger, ctrl,
            rhythm=rhythm, experiences=others, fired=fired)
        if plan is None:
            plan = rule_plan
        else:
            plan = _backfill(plan, rule_plan)          # the LLM plan never loses what the rules would try
        serving_before = ctrl.tree.serving
        applied, rejected, version = self._apply(ctrl, box, plan, r, source, trigger, sig)
        known = {e.id for e in matched}
        used = sorted({a["experience"] for a in applied if a.get("experience") in known})
        refit = None
        if any(a["kind"] == "improvement" and (a.get("result") or {}).get("mode") == "group_kfold" for a in applied):
            refit = ctrl.refit_opening(f"round {r}: serve the procedure validated by group k-fold")
        blind = {"trigger": f"{trigger} {TRIGGERS[trigger][0]}", "signature": sig,
                 "findings": plan.get("findings"), "hypotheses": plan.get("hypotheses"),
                 "tool_calls": [{"tool": c["tool"], "args": c["args"]} for c in box.calls],
                 "experiments": [{"operator": e.get("operator"), "params": e.get("params"),
                                  "validation_passed": e.get("passed"),
                                  "result": {k: e["result"].get(k) for k in
                                             ("mode", "n", "mean_diff", "band", "cost_change", "skipped")}}
                                 for e in box.experiments]}
        summary, stokens = (self.agent.summarize_round(blind)
                            if source == "llm" and self._llm_allowed(ctrl, 3000) else (None, 0))
        if stokens:
            ctrl.ledger.charge(Cost(llm_tokens=stokens))
        if summary is None:
            summary = _template(r, trigger, window, plan, applied, rejected, box.experiments)
        stored = mem.add_lessons(r, summary.get("lessons"), ctrl.state["code_version"])
        mem.append({"round": r, "trigger": trigger, "trigger_meaning": TRIGGERS[trigger][0], "window": window,
                    "source": source, "note": note, "fired": fired, "signature": sig, "plan": _without_sources(plan),
                    "trace": [{"tool": t["tool"], "args": t["args"], "why": t.get("why")} for t in trace],
                    "tool_calls": box.calls, "experiments": box.experiments, "applied": applied,
                    "rejected": rejected, "strategy_version": version, "summary": summary,
                    "experiences_matched": [e.view() for e in matched], "experiences_used": used,
                    "opening_refit": refit, "mode_after": ctrl.policy.mode,
                    "lessons_stored": stored, "tokens": tokens + stokens + self._repair_tokens})
        sv["interventions"].append({"window": window, "tier": "update", "trigger": trigger, "source": source,
                                    "applied": [a["id"] for a in applied], "rejected": [x["id"] for x in rejected],
                                    "experiences": used, "mode": ctrl.policy.mode,
                                    "tokens": tokens + stokens + self._repair_tokens, "round": r})
        self._repair_tokens = 0
        if applied:
            mode_change = any(a["kind"] == "mode" for a in applied)
            if mode_change:
                sv.update(mode_since=window + 1, mode_held=False)
            sv["mode_switches" if trigger == "T10" else "updates"] += 1
            sv["probation"] = {"round": r, "version": version, "start": window + 1, "trigger": trigger,
                               "until": window + 1 + ctrl.policy.probation_windows, "previous": self._previous,
                               "experiences": used, "refit": refit, "serving_before": serving_before,
                               "mode_change": mode_change or trigger == "T1"}
        if trigger == "T1":
            # the model the run falls back to when later updates go below the static model
            ctrl.state["floor_node"] = ctrl.tree.serving

    # ================================================================ when the LLM is worth its tokens
    def _llm_allowed(self, ctrl, need: int) -> bool:
        p = ctrl.policy
        return (self.use_llm and ctrl.ledger.affordable(Cost(llm_tokens=need))
                and ctrl.ledger.spent.llm_tokens + min(need, p.explore_tokens) <= p.run_llm_tokens)

    def _rules_failed(self, ctrl) -> bool:
        """The deterministic rules already tried and failed in this run: a probation was reverted, or a
        no-return / stall round changed nothing."""
        sv = self._state(ctrl)
        return sv["outcomes"]["reverted"] > 0 or fallback_explorer._fruitless_rounds(ctrl) >= 1

    def _llm_wanted(self, ctrl, trigger: str, rhythm) -> tuple[bool, str]:
        """The rules handle what the experiences cover; the LLM is called where they do not: an opening no
        credible experience fits, or trouble after the rules have already failed in this run. A rhythm
        switch (T10) is a computed fact and never needs it."""
        if not self.use_llm:
            return False, "no LLM"
        if trigger == "T10":
            return False, "a rhythm switch follows from the signature"
        if trigger == "T1":
            if rhythm is None or rhythm.credibility < 0.6:
                return True, "no credible rhythm experience fits this task"
            return False, f"experience {rhythm.id} (credibility {rhythm.credibility:.2f}) fits"
        if trigger in ("T2", "T3", "T4", "T5", "T6") and self._rules_failed(ctrl):
            return True, "the rules already failed in this run"
        return False, "the rules cover this trigger"

    # ================================================================ applying a plan
    def _apply(self, ctrl, box: Toolbox, plan: dict, r: int, source: str, trigger: str, sig: dict):
        """Every correction is translated into operator calls and executed by the manual:
        invalid calls are refused with the manual's reason, tightening calls apply at once, improving
        calls must pass the experiment the manual prescribes (code first passes the official tests).
        At most one improvement is adopted per round, so its effect stays attributable."""
        meta_fields = set(structure._fields(ctrl.session.base))
        tightening: dict = {}
        rhythm: dict | None = None
        improvements = []
        applied, rejected = [], []
        tried: dict[str, int] = {}             # validation experiments per layer group (they are paid for)
        for correction in (plan.get("corrections") or [])[:8]:
            cid = str(correction.get("id", f"C{len(applied) + len(rejected) + 1}"))
            by = correction.get("by", source)
            for name, params in calls_from(correction):
                call = {"id": cid, "by": by, "operator": name, "params": _plain_params(params),
                        "why": correction.get("why")}
                if correction.get("experience"):
                    call["experience"] = str(correction["experience"])
                try:
                    inst = operators.instantiate(name, params, sig, ctrl.policy, meta_fields)
                except operators.OperatorError as error:
                    rejected.append({**call, "reason": str(error)})
                    continue
                effective = {k: v for k, v in inst.changes.items() if getattr(ctrl.policy, k) != v}
                if not effective and not inst.edits:
                    rejected.append({**call, "reason": "already in effect"})
                    continue
                inst.changes = effective
                if inst.kind == "mode":
                    if rhythm is not None:
                        rejected.append({**call, "reason": "one rhythm change per round; an earlier one was taken"})
                        continue
                    rhythm = effective
                    applied.append({**call, "kind": "mode", "change": effective})
                    continue
                if inst.kind in ("tightening", "meta"):
                    tightening.update(effective)
                    applied.append({**call, "kind": inst.kind, "change": effective})
                    continue
                if inst.operator.validation == "unverifiable":
                    rejected.append({**call, "reason": "this direction cannot be validated during the run"})
                    continue
                group = _layer_group(inst)
                if tried.get(group, 0) >= ctrl.policy.opening_candidates:
                    rejected.append({**call, "reason": f"experiment budget: {ctrl.policy.opening_candidates} "
                                                       f"{group} candidates per round were already tried"})
                    continue
                tried[group] = tried.get(group, 0) + 1
                if inst.edits:
                    ok, why, inst = self._sandbox(ctrl, inst)
                    if not ok:
                        rejected.append({**call, "edits": inst.describe()["edits"], "reason": why})
                        continue
                result = box.validate(inst, correction.get("validation"), tag=by)
                if not result.get("n") and correction.get("validation"):
                    # a validation design that produced no evidence at all is replaced by the manual's default
                    result = box.validate(inst, None, tag=f"{by}:default-validation")
                if result.get("passed"):
                    relative = result["mean_diff"] / max(abs(result.get("baseline_mean") or 0.0), 1e-9)
                    improvements.append((relative, call, inst, result))
                else:
                    rejected.append({**call, "change": inst.changes, "edits": inst.describe()["edits"],
                                     "reason": result.get("why") or result.get("error") or "experiment failed",
                                     "result": {k: result.get(k) for k in ("mode", "n", "mean_diff", "band")}})
        chosen = self._choose_improvements(box, improvements, trigger)
        taken = {id(c[2]) for c in chosen}
        for item in improvements:
            if id(item[2]) not in taken:
                rejected.append({**item[1], "reason": "passed, but another improvement gained more"})
        new_policy = {**ctrl.state["policy"], **(rhythm or {}), **tightening}
        new_code = None
        edits = [e for _, _, inst, _ in chosen for e in inst.edits]
        for _, call, inst, result in chosen:
            new_policy.update(inst.changes)
            applied.append({**call, "kind": "improvement", "change": inst.changes, "edits": inst.describe()["edits"],
                            "result": {k: result.get(k) for k in ("mode", "n", "mean_diff", "band", "cost_change",
                                                                  "combined")}})
        if edits:
            new_code = experiments.candidate_code(ctrl.code, edits, self.code_store)
        self._previous = {"policy": dict(ctrl.state["policy"]), "code_version": ctrl.state["code_version"]}
        if not applied:
            return applied, rejected, ctrl.state["strategy_version"]
        version = ctrl.apply_strategy(
            policy=PolicyConfig(**new_policy), code=new_code,
            reason="; ".join(f"{a['id']} {a['operator']}{' (validated)' if a['kind'] == 'improvement' else ''}"
                             f"{' [' + a['experience'] + ']' if a.get('experience') else ''}" for a in applied),
            source=f"{source}:{trigger}", round_index=r, evidence={"applied": applied, "rejected": rejected})
        return applied, rejected, version

    def _choose_improvements(self, box: Toolbox, improvements: list, trigger: str) -> list:
        """One improvement per round keeps its effect attributable. At the opening, the best data change and
        the best training-code change may both be adopted, but only if the two together beat each alone in
        one more experiment (they need not add up)."""
        if not improvements:
            return []
        best = max(improvements, key=lambda item: item[0])
        if trigger != "T1":
            return [best]
        by_group: dict[str, tuple] = {}
        for item in improvements:
            group = _layer_group(item[2])
            if group not in by_group or item[0] > by_group[group][0]:
                by_group[group] = item
        if len(by_group) < 2 or "code" not in by_group:
            return [best]
        code_item = by_group.pop("code")
        data_item = max(by_group.values(), key=lambda item: item[0])
        joint = operators.Instance(code_item[2].operator, {**data_item[2].params, **code_item[2].params},
                                   {**data_item[2].changes, **code_item[2].changes}, list(code_item[2].edits),
                                   "improvement")
        result = box.validate(joint, None, tag="combined")
        if result.get("passed") and result["mean_diff"] >= max(data_item[3]["mean_diff"], code_item[3]["mean_diff"]):
            note = {"combined": {"mean_diff": result["mean_diff"], "band": result["band"]}}
            return [(data_item[0], data_item[1], data_item[2], {**data_item[3], **note}),
                    (code_item[0], code_item[1], code_item[2], {**code_item[3], **note})]
        return [best]

    def _sandbox(self, ctrl, inst):
        """Official tests in a subprocess. A custom slot written by the LLM gets one repair attempt
        with the failure shown to it."""
        base = codebase.base_version(ctrl.task.package, self.code_store)
        if self._known_failures is None:
            self._known_failures = sandbox.reference_failures(base, self.code_store.parent / "sandbox_reference.json")
        for attempt in range(2):
            try:
                version = experiments.candidate_code(ctrl.code, inst.edits, self.code_store)
            except (SyntaxError, ValueError) as error:
                ok, detail, why = False, {"tail": str(error)}, f"slot edits rejected: {error}"
            else:
                ok, detail = sandbox.check(version, self._known_failures)
                why = "official tests passed" if ok else f"official tests failed: {detail['new_failures']}"
            edit = inst.edits[0]
            if ok or attempt or edit.get("variant") != "custom" or not self.use_llm:
                return ok, why, inst
            fixed, tokens = self.agent.repair_slot(edit["slot"], edit["source"], detail.get("tail", why))
            self._repair_tokens += tokens
            if tokens:
                ctrl.ledger.charge(Cost(llm_tokens=tokens))
            if not fixed:
                return ok, why, inst
            inst.edits = [{**edit, "source": fixed, "hypothesis": (edit.get("hypothesis") or "") + " [repaired]"}]
        return ok, why, inst

    def _judge_probation(self, ctrl, probation: dict) -> None:
        rows = ctrl.state["health"]
        start, until, m = probation["start"], probation["until"], ctrl.policy.probation_windows

        def diffs(lo, hi):
            return [r["vs_static"]["diff"] for r in rows if lo <= r["window"] < hi and r.get("vs_static")]
        after, before = diffs(start, until), diffs(start - m, start)
        sv = self._state(ctrl)
        if not after:
            verdict = "inconclusive"
        else:
            def bands(lo, hi):
                return [r["vs_static"]["band"] for r in rows if lo <= r["window"] < hi and r.get("vs_static")]
            # the noise of both means being compared (a period served by the static model itself has band 0)
            b_after, b_before = bands(start, until), bands(start - m, start)
            tolerance = float(np.sqrt((np.mean(np.square(b_after)) / len(b_after) if b_after else 0.0) +
                                      (np.mean(np.square(b_before)) / len(b_before) if b_before else 0.0)))
            reference = float(np.mean(before)) if before else 0.0
            # worse than before the change by more than the noise of the probation windows → undo it
            verdict = "reverted" if float(np.mean(after)) < reference - tolerance else "confirmed"
        used = probation.get("experiences") or []
        if verdict == "confirmed" and probation.get("mode_change") and float(np.mean(after)) > 0:
            sv["mode_held"] = True          # the rhythm earned something: only a fall below static moves it
        if verdict == "reverted":
            sv.update(mode_since=ctrl.state["next_window"], mode_held=False)
            previous = probation["previous"]
            code = codebase.open_version(self.code_store, previous["code_version"]) \
                if previous["code_version"] != ctrl.state["code_version"] else None
            ctrl.apply_strategy(policy=PolicyConfig(**previous["policy"]), code=code,
                                reason=f"probation of round {probation['round']} failed", source="supervisor:revert",
                                round_index=probation["round"])
            if probation.get("refit") and ctrl.tree.serving == probation["refit"]:
                ctrl.tree.pointers.update(confirmed=probation["serving_before"], provisional=None, shadow=None)
            if probation.get("refit") and ctrl.state.get("floor_node") == probation["refit"]:
                ctrl.state["floor_node"] = probation["serving_before"]
            # The same experiences would be chosen again at once: keep them out for the rest of this run.
            sv["blocked_experiences"] = sorted(set(sv["blocked_experiences"]) | set(used))
        sv["outcomes"][verdict] += 1
        self.experiences.record(used, verdict, probation["round"], ctrl.task.task)
        self.memory.backfill(probation["round"], {"probation": verdict, "after_vs_static": after,
                                                  "before_vs_static": before, "experiences": used})


# ==================================================================== correction → operator calls

_SLOT_OPERATORS = {"loss_and_delta": "code.loss", "sample_weights": "code.sample_weight",
                   "regularizer_gradients": "code.regularizer", "learning_rate": "code.lr_schedule"}
_SIMPLE = {"z": ("gate.z", "value"), "min_gate_units": ("gate.min_units", "n"), "confirm_windows": ("gate.confirm", "n"),
           "cooldown_windows": ("gate.cooldown", "n"), "label_fraction": ("label.fraction", "fraction"),
           "max_epochs": ("train.max_epochs", "n"), "min_samples": ("data.min_samples", "n"),
           "lr_scale": ("train.lr_scale", "scale"), "branch_from": ("train.branch_from", "target")}
_CATALOGUE = {v.name for v in slot_rules.VARIANTS}


def calls_from(correction: dict) -> list[tuple[str, dict]]:
    """Operator calls in a correction. The manual's own format is {"operator", "params"}; older or
    sloppier {"change": {...}, "edits": [...]} forms are translated, so a well-meant correction is not
    lost to formatting (e.g. a "validation" object written inside "change")."""
    if correction.get("operator"):
        return [(str(correction["operator"]), dict(correction.get("params") or {}))]
    if isinstance(correction.get("operators"), list):
        return [(str(c.get("operator")), dict(c.get("params") or {})) for c in correction["operators"] if isinstance(c, dict)]
    calls = []
    change = {k: v for k, v in dict(correction.get("change") or {}).items() if k != "validation"}
    if "cap_per_group" in change or "cap_group_by" in change:
        n, group = change.pop("cap_per_group", None), change.pop("cap_group_by", None)
        if n is None:
            calls.append(("data.no_cap", {}))
        elif group == ["answer"]:
            calls.append(("data.class_balance", {"n": n}))
        else:
            calls.append(("data.cap_per_group", {"n": n, **({"group_by": group} if group else {})}))
    if "split_mode" in change:
        mode, group = change.pop("split_mode"), change.pop("split_group_by", None)
        calls.append((f"split.{mode}", {"group_by": group} if mode == "by_group" and group else {}))
    elif "split_group_by" in change:
        calls.append(("split.by_group", {"group_by": change.pop("split_group_by")}))
    if "mode" in change:
        calls.append((f"mode.{change.pop('mode')}", {}))
    for field_name, (op, param) in {"gate_mode": ("gate.mode", "mode"), "train_trigger": ("train.trigger", "trigger"),
                                    "label_sampling": ("label.sampling", "method")}.items():
        if field_name in change:
            calls.append((op, {param: change.pop(field_name)}))
    if "serve_mode" in change:
        calls.append((f"serve.{change.pop('serve_mode')}", {}))
    if "recent_windows_only" in change:
        n = change.pop("recent_windows_only")
        calls.append(("data.recent_only", {"n": n}) if n else ("data.all_history", {}))
    for field_name, (op, param) in _SIMPLE.items():
        if field_name in change:
            calls.append((op, {param: change.pop(field_name)}))
    meta = {k: change.pop(k) for k in ("probation_windows", "review_every", "health_window") if k in change}
    if meta:
        calls.append(("supervise.tune", meta))
    for field_name, value in change.items():
        calls.append((f"<field {field_name}>", {"value": value}))
    for edit in correction.get("edits") or []:
        op = _SLOT_OPERATORS.get(edit.get("slot"), f"<slot {edit.get('slot')}>")
        if edit.get("variant") in _CATALOGUE and not edit.get("source"):
            calls.append((op, {"variant": edit["variant"]}))
        elif edit.get("variant") in _CATALOGUE:
            calls.append((op, {"variant": edit["variant"]}))
        else:
            calls.append((op, {"source": edit.get("source"), "hypothesis": edit.get("hypothesis", "")}))
    return calls


def _backfill(plan: dict, rule_plan: dict) -> dict:
    """Add the rule plan's corrections whose operators the LLM plan does not already call."""
    used = {name for c in plan.get("corrections") or [] for name, _ in calls_from(c)}
    extra = []
    for c in rule_plan.get("corrections") or []:
        names = [name for name, _ in calls_from(c)]
        if names and not any(n in used for n in names):
            extra.append({**c, "id": f"R{len(extra) + 1}", "by": "rule"})
    if not extra:
        return plan
    return {**plan, "corrections": list(plan.get("corrections") or []) + extra}


def _layer_group(inst) -> str:
    """Improvements compete inside a group: training code vs everything that shapes data and training."""
    return "code" if inst.operator.layer == "code" else "data"


def _plain_params(params: dict) -> dict:
    return {k: (v if k != "source" else f"<{len(str(v))} chars of code>") for k, v in (params or {}).items()}


def _without_sources(plan: dict) -> dict:
    out = dict(plan)
    corrections = []
    for c in plan.get("corrections") or []:
        c = dict(c)
        if "params" in c:
            c["params"] = _plain_params(c["params"])
        if "edits" in c:
            c["edits"] = [{k: e.get(k) for k in ("slot", "variant", "hypothesis")} for e in c.get("edits") or []]
        corrections.append(c)
    out["corrections"] = corrections
    return out


def _template(r: int, trigger: str, window: int, plan: dict, applied: list, rejected: list, exps: list) -> dict:
    """Without an LLM: lessons from validated experiments, written as operator calls with the
    placeholders the correction used, so they transfer to other tasks."""
    requested = {}
    for c in plan.get("corrections") or []:
        for name, params in calls_from(c):
            requested.setdefault(name, params)
    lessons = []
    for e in exps:
        result = e["result"]
        if e.get("passed") and result.get("n"):
            call = {"operator": e.get("operator"), "params": _plain_params(requested.get(e.get("operator"), e.get("params")))}
            lessons.append({"cue": f"{TRIGGERS[trigger][0]}; {result.get('mode')} validation",
                            "lesson": f"{call['operator']}({json.dumps(call['params'], ensure_ascii=False)}) improved held-out utility",
                            "action": call, "confidence": "medium",
                            "evidence": f"paired difference {result['mean_diff']:+.4f} vs band {result['band']:.4f}, "
                                        f"n={result['n']}"})
    findings = "；".join(f.get("claim", "") for f in plan.get("findings") or []) or "无新发现"
    done = "；".join(f"{a['id']} {a['operator']}（{KIND_TEXT.get(a['kind'], a['kind'])}"
                    f"{'，依据经验 ' + a['experience'] if a.get('experience') else ''}）" for a in applied) or "无"
    refused = "；".join(f"{x['id']} {x['operator']}（{x['reason']}）" for x in rejected) or "无"
    when = "开局（第 0 批之前）" if window < 0 else f"第 {window} 批之后"
    paragraph = (f"第 {r} 轮由“{TRIGGERS[trigger][0]}”触发，时间：{when}。发现：{findings}。"
                 f"生效的修正：{done}。未采纳：{refused}。")
    return {"lessons": lessons, "next_time": "check whether the applied change held during probation",
            "report_paragraph": paragraph}


def _closing_template(stats: dict, rounds: list[dict]) -> dict:
    tiers = stats["tiers"]
    paragraph = (f"全程 {stats['windows']} 批：延续 {tiers['continue']} 次、复查 {tiers['review']} 次、更新 "
                 f"{tiers['update']} 次；策略版本 {stats['strategy_versions']} 个，观察期确认 "
                 f"{stats['outcomes']['confirmed']} 次、撤销 {stats['outcomes']['reverted']} 次；模型晋升 "
                 f"{stats['promotions']} 次、回退 {stats['rollbacks']} 次。")
    return {"lessons": [], "next_time": "", "report_paragraph": paragraph}
