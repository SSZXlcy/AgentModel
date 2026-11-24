"""Acceptance tests for the harness (设计方案 §10). Public data only; task A keeps them fast."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import agent_harness  # noqa: F401  (thread variables before NumPy)
import numpy as np

from agent_harness import codebase, gate, metrics, operators, sandbox, signature, structure
from agent_harness import policy as rules
from agent_harness import slots as slot_rules
from agent_harness.agent import Agent, local_proposal
from agent_harness.config import PolicyConfig, load_task
from agent_harness.contracts import Cost
from agent_harness.controller import Controller, Session, _cap
from agent_harness.data import Table
from agent_harness.evolution import edit_budget
from agent_harness.executor import Executor
from agent_harness.explorer import explore as explore_llm
from agent_harness.ledger import BudgetError, Ledger
from agent_harness.library import LeakError, Library
from agent_harness.memory import Memory
from agent_harness.recorder import Recorder, read_jsonl
from agent_harness.report import build as build_report
from agent_harness.supervisor import Supervisor, _backfill, calls_from
from agent_harness.tools import Toolbox
from agent_harness.tree import Tree

ROOT = Path(__file__).resolve().parents[1]


class Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="harness-test-"))
        cls.task = load_task(ROOT / "configs/tasks/A.yaml", ROOT)
        cls.v0 = codebase.base_version(cls.task.package, cls.tmp / "code")
        cls.session = Session(cls.task, cls.v0)
        cls.policy = PolicyConfig(bootstrap_reps=100)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class CodeVersionTests(Shared):
    def test_v0_checkpoint_is_byte_identical_to_official(self):
        official = codebase.CodeVersion("official-test", self.task.package, None).modules().train
        mine = self.v0.modules().train
        args = (self.task.train_path, self.task.stream_path)
        a = official.train_run("A", *args, self.tmp / "official", resume=self.task.checkpoint, epochs=2, stage_id="t")
        b = mine.train_run(str(self.v0.profile_path("A")), *args, self.tmp / "slotted", resume=self.task.checkpoint,
                           epochs=2, stage_id="t")
        self.assertEqual(a["checkpoint_sha256"], b["checkpoint_sha256"])
        self.assertIn("slots.py", b["source_sha256"])

    def test_split_training_equals_continuous_training(self):
        run = self.tmp / "split"
        tree, ledger = Tree(run), Ledger()
        root = tree.add_root(self.task.checkpoint, self.v0.version_id, self.v0.modules().model.Model.load(self.task.checkpoint).meta)
        executor = Executor(self.v0, "A", tree, ledger, Recorder(run), 1e-6)
        args = (self.task.train_path, self.task.stream_path, 445)
        one = executor.train(root, *args, epochs=1, learning_rate=0.003, window=0, recipe_id="r", spec="s")
        two = executor.train(one, *args, epochs=1, learning_rate=0.003, window=0, recipe_id="r", spec="s")
        both = executor.train(root, *args, epochs=2, learning_rate=0.003, window=0, recipe_id="r", spec="s")
        self.assertEqual(two.sha256, both.sha256)
        self.assertEqual(executor.row_epochs, 445 * 4)

    def test_slot_boundary(self):
        base = slot_rules.DEFAULT_SOURCE
        good = slot_rules.apply_edits(base, [{"slot": "learning_rate",
                                              "source": "def learning_rate(base_lr, step, epoch):\n    return base_lr * 0.5\n"}])
        self.assertEqual(slot_rules.check_module(good), [])
        self.assertEqual(slot_rules.changed_slots(base, good), ["learning_rate"])
        self.assertEqual(slot_rules.changed_slots(base, base.replace("return {}", "return  {}")), [])
        bad_import = slot_rules.apply_edits(base, [{"slot": "learning_rate",
                                                    "source": "def learning_rate(base_lr, step, epoch):\n    import os\n    return base_lr\n"}])
        self.assertTrue(slot_rules.check_module(bad_import))
        self.assertTrue(slot_rules.check_module(base + "\nprint('outside')\n"))
        self.assertTrue(slot_rules.check_module(base.replace("def learning_rate(base_lr, step, epoch)",
                                                             "def learning_rate(base_lr, step)")))
        self.assertTrue(slot_rules.check_module(base.replace("return base_lr", "return open('x')")))
        with self.assertRaises(ValueError):
            slot_rules.apply_edits(base, [{"slot": "learning_rate", "source": "def other(a):\n    return a\n"}])

    def test_catalogue_variants_are_valid(self):
        for variant in slot_rules.VARIANTS:
            source = slot_rules.apply_edits(slot_rules.DEFAULT_SOURCE, [variant.edit()])
            self.assertEqual(slot_rules.check_module(source), [], variant.name)
            self.assertEqual(slot_rules.changed_slots(slot_rules.DEFAULT_SOURCE, source), [variant.slot])

    def test_sandbox_rejects_a_slot_that_breaks_training(self):
        broken = slot_rules.apply_edits(slot_rules.DEFAULT_SOURCE, [{"slot": "regularizer_gradients", "source":
            "def regularizer_gradients(params):\n    return {'w1': params['w1'] * float('nan')}\n"}])
        version = codebase.derive_version(self.v0, broken, [], self.tmp / "code")
        ok, detail = sandbox.check(version, set())
        self.assertFalse(ok)
        self.assertTrue(detail["new_failures"])


class StatisticsTests(Shared):
    def test_metrics_match_official(self):
        model = self.v0.modules().model.Model.load(self.task.checkpoint)
        table = self.session.stream_table
        official = self.v0.modules().train.metrics(model, {"x": table.x, "y": table.y})
        mine = metrics.report("classification", table.y, model.predict(table.x))
        self.assertAlmostEqual(official["macro_f1"], mine["macro_f1"], places=12)
        self.assertAlmostEqual(official["macro_f1"], 0.7140297220788709, places=12)

    def test_paired_difference_and_gate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        groups = np.asarray([str(i) for i in range(300)])
        weak = np.eye(3)[np.where(rng.random(300) < 0.5, y, (y + 1) % 3)]
        strong = np.eye(3)[np.where(rng.random(300) < 0.95, y, (y + 1) % 3)]
        self.assertEqual(metrics.paired_difference("classification", y, weak, weak, groups, 50), (0.0, 0.0))
        self.assertFalse(gate.promotion_gate("classification", y, weak, weak, groups, 1.0, self.policy).accepted)
        self.assertTrue(gate.promotion_gate("classification", y, strong, weak, groups, 1.0, self.policy).accepted)
        self.assertEqual(gate.confirmation("classification", y, weak, strong, groups, self.policy)[0], "rollback")
        self.assertEqual(gate.confirmation("classification", y, strong, weak, groups, self.policy)[0], "hold")

    def test_correlated_rows_widen_the_band(self):
        rng = np.random.default_rng(1)
        y = np.repeat(rng.normal(0, 100, 3), 100)              # three entities, strongly correlated rows
        out = y + np.repeat(rng.normal(0, 50, 3), 100) + rng.normal(0, 5, 300)
        by_row = metrics.utility_sd("regression", y, out, np.asarray([str(i) for i in range(300)]), 200)
        by_entity = metrics.utility_sd("regression", y, out, np.repeat(np.asarray(["a", "b", "c"]), 100), 200)
        self.assertGreater(by_entity, 2 * by_row)


class BookkeepingTests(Shared):
    def test_ledger_refuses_before_execution(self):
        run = self.tmp / "budget"
        tree, ledger = Tree(run), Ledger(max_train_seconds=0.0)
        root = tree.add_root(self.task.checkpoint, self.v0.version_id, self.v0.modules().model.Model.load(self.task.checkpoint).meta)
        executor = Executor(self.v0, "A", tree, ledger, Recorder(run), 1e-3)
        with self.assertRaises(BudgetError):
            executor.train(root, self.task.train_path, self.task.stream_path, 445, epochs=1, learning_rate=0.003,
                           window=0, recipe_id="r", spec="s")
        self.assertFalse((run / "nodes" / "n0001").exists())
        self.assertEqual(ledger.spent, Cost())

    def test_tree_pointers(self):
        tree = Tree(self.tmp / "pointers")
        tree.add_root(self.task.checkpoint, "v", {"epochs": 1, "steps": 1, "learning_rate": 0.1})
        tree.promote_provisional("n0001")
        self.assertEqual(tree.serving, "n0001")
        tree.rollback()
        self.assertEqual(tree.serving, "n0000")
        tree.promote_provisional("n0002")
        tree.confirm()
        self.assertEqual((tree.pointers["confirmed"], tree.serving), ("n0002", "n0002"))
        tree.reset_to_static()
        self.assertEqual(tree.serving, "n0000")

    def test_recipes(self):
        policy = PolicyConfig(min_samples=300)
        pool = {0: 100, 1: 100, 2: 100, 3: 100}
        adaptive = rules.resolve_recipe("adaptive", 4, pool, 50, policy)
        self.assertEqual((adaptive.past_windows, adaptive.include_base), ((1, 2, 3), False))
        thin = rules.resolve_recipe("recent:1", 4, pool, 50, policy)
        self.assertEqual((thin.past_windows, thin.include_base), ((3,), True))
        self.assertEqual(rules.resolve_recipe("all", 4, pool, 50, policy).past_windows, (0, 1, 2, 3))
        self.assertEqual(adaptive.recipe_id, rules.resolve_recipe("adaptive", 4, pool, 50, policy).recipe_id)

    def test_stream_refuses_unpaid_training_rows(self):
        stream = self.session.new_stream()
        released, labels = stream.request_labels(0, np.array([0, 1, 2]))
        self.assertEqual((len(released), len(labels)), (3, 3))
        self.assertEqual(len(stream.labelled_lines(0, np.array([0, 1]))), 2)
        with self.assertRaises(PermissionError):
            stream.labelled_lines(0, np.array([3]))

    def test_library_admission_leaks_and_causal_prefix(self):
        library = Library(self.tmp / "library", ["data/raw"])
        good = {"cue": "c", "lesson": "l", "evidence": "e", "confidence": "high"}
        self.assertIsNone(library.add("x", 0, "v", "src", "why", {"cue": "c"}))
        self.assertIsNotNone(library.add("r0", 0, "v0", "src", "why", good))
        self.assertIsNotNone(library.add("r1", 1, "v1", "src", "why", good))
        with self.assertRaises(LeakError):
            library.add("r2", 2, "v2", "read data/raw/x", "why", good)
        library.retain(1)
        self.assertEqual([e["entry_id"] for e in library.entries()], ["r0"])
        self.assertIn("r0", library.index.read_text(encoding="utf-8"))

    def test_edit_budget_anneals(self):
        budgets = [edit_budget(t, 5, 1, 3) for t in range(5)]
        self.assertEqual(budgets[0], 3)
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertGreaterEqual(min(budgets), 1)


class AgentTests(unittest.TestCase):
    evidence = {"drift": {"shifted": True}, "performance": {"degraded": False}, "errors": {}}

    def test_fallbacks_equal_the_rule_table(self):
        class Broken:
            model = "broken"

            def generate_json(self, system, payload):
                raise OSError("offline")

        class OutOfBounds:
            model = "oob"

            def generate_json(self, system, payload):
                return {"drift_type": "covariate", "recipe": "everything", "confidence": 0.9}, 10

        class Valid:
            model = "valid"

            def generate_json(self, system, payload):
                return {"drift_type": "concept", "recipe": "recent:1", "confidence": 0.7, "rationale": "r"}, 10

        rule = rules.rule_judgment(self.evidence)
        for provider in (None, Broken()):
            judged = Agent(provider).judge_drift(self.evidence)
            self.assertEqual((judged.drift_type, judged.recipe, judged.source), (rule.drift_type, rule.recipe, "rule"))
        self.assertEqual(Agent(OutOfBounds()).judge_drift(self.evidence).source, "rule")
        valid = Agent(Valid()).judge_drift(self.evidence)
        self.assertEqual((valid.drift_type, valid.recipe, valid.source, valid.tokens), ("concept", "recent:1", "llm", 10))

    def test_local_proposal_never_repeats(self):
        first = local_proposal("PHM", 1, "regression", {"errors": {"over_estimate_ratio": 0.7}}, [])
        self.assertEqual(first[0]["variant"], "asymmetric_overestimate")
        tried = [{"edits": first}]
        second = local_proposal("PHM", 1, "regression", {"errors": {}}, tried)
        self.assertNotEqual(second[0]["variant"], first[0]["variant"])


class ControllerTests(Shared):
    def test_end_to_end_is_deterministic_and_settles_every_decision(self):
        a = Controller(self.task, self.policy, self.v0, self.tmp / "run_a", session=self.session).run()
        b = Controller(self.task, self.policy, self.v0, self.tmp / "run_b", session=self.session).run()
        self.assertEqual([w["utility"] for w in a["per_window"]], [w["utility"] for w in b["per_window"]])
        shas = [[n["sha256"] for n in json.loads((self.tmp / r / "tree.json").read_text())["nodes"]] for r in ("run_a", "run_b")]
        self.assertEqual(shas[0], shas[1])
        rows = read_jsonl(self.tmp / "run_a" / "transitions.jsonl")
        decisions = [r for r in rows if r["type"] == "decision"]
        settled = {r["decision_seq"] for r in rows if r["type"] == "settlement"}
        self.assertEqual(len(decisions), a["windows"])
        self.assertEqual({d["seq"] for d in decisions[:-1]}, settled)
        for d in decisions:
            self.assertIn("observe", d["action_mask"])
            self.assertEqual(set(d["cost"]), {"train_seconds", "labels", "llm_tokens"})
        self.assertEqual(a["errors"], 0)
        self.assertTrue((self.tmp / "run_a" / "manifest.json").is_file())

    def test_resume_reproduces_an_uninterrupted_run(self):
        full = Controller(self.task, self.policy, self.v0, self.tmp / "full", session=self.session).run()
        Controller(self.task, self.policy, self.v0, self.tmp / "resumed", session=self.session,
                   windows=[0, 1]).run()                            # "crash" after two windows
        resumed = Controller(self.task, self.policy, self.v0, self.tmp / "resumed", session=self.session).run(resume=True)
        self.assertEqual([w["utility"] for w in full["per_window"]], [w["utility"] for w in resumed["per_window"]])
        self.assertEqual(full["pointers"], resumed["pointers"])


def _synthetic_table(meta: list[dict], y: np.ndarray) -> Table:
    n = len(meta)
    ids = np.asarray([f"r{i}" for i in range(n)])
    return Table(ids, ["{}"] * n, np.zeros((n, 2)), np.asarray(y, dtype=float), ids.copy(), meta)


class StructureTests(unittest.TestCase):
    """The discovery must not depend on any field name: these tables use invented names."""

    def test_entity_order_and_countdown_with_unknown_names(self):
        meta, y = [], []
        for machine, life in (("M-a", 30), ("M-b", 120), ("M-c", 60), ("M-d", 45)):
            for tick in range(1, life + 1):
                meta.append({"machine_tag": machine, "tick": tick, "batch_note": "x"})
                y.append(life - tick)
        found = structure.detect(_synthetic_table(meta, np.asarray(y)), "regression")
        self.assertEqual((found["entity"], found["order"], found["countdown"]), (["machine_tag"], "tick", True))

    def test_composite_entity_key_is_found(self):
        meta, y = [], []
        for plant in ("P1", "P2"):
            for unit in (1, 2, 3):                                   # unit numbers repeat across plants
                for step in range(1, 21 + unit):
                    meta.append({"plant": plant, "unit_no": unit, "t": step})
                    y.append(20 + unit - step)
        found = structure.detect(_synthetic_table(meta, np.asarray(y)), "regression")
        self.assertEqual(sorted(found["entity"]), ["plant", "unit_no"])
        self.assertTrue(found["countdown"])

    def test_independent_rows(self):
        meta = [{"origin": "lab", "row_index": i} for i in range(200)]
        found = structure.detect(_synthetic_table(meta, np.zeros(200)), "classification")
        self.assertIsNone(found["entity"])

    def test_public_tasks(self):
        cases = {"A": None, "B": ["bearing"], "C": ["source_subset", "unit"]}
        for name, entity in cases.items():
            task = load_task(ROOT / f"configs/tasks/{name}.yaml", ROOT)
            with tempfile.TemporaryDirectory() as tmp:
                code = codebase.base_version(task.package, Path(tmp))
                session = Session(task, code)
                found = session.structure
            self.assertEqual(found["entity"] and sorted(found["entity"]), entity and sorted(entity), name)


class ProtocolAndSplitTests(Shared):
    def test_labels_released_only_when_the_entity_completes(self):
        task = dataclasses.replace(load_task(ROOT / "configs/tasks/B.yaml", ROOT), label_protocol="on_group_complete")
        with tempfile.TemporaryDirectory() as tmp:
            code = codebase.base_version(task.package, Path(tmp))
            stream = Session(task, code).new_stream()
            released, _ = stream.request_labels(1, np.array([0, 1, 2]))      # an entity still running
            self.assertEqual(len(released), 0)
            with self.assertRaises(PermissionError):
                stream.labels(1, np.array([0]))
            late = {}
            for k in range(2, len(stream)):
                late.update(stream.deliver(k))
            self.assertEqual(late[1].tolist(), [0, 1, 2])                    # delivered when it finished
            self.assertEqual(len(stream.labels(1, np.array([0, 1, 2]))), 3)

    def test_split_modes_keep_entities_and_time_apart(self):
        metas = [{"e": key} for key in "aaabbbcccddd"]
        policy = PolicyConfig(split_mode="by_group", split_group_by=["e"], gate_fraction=0.5)
        fit, gate_rows = Session.split(policy, 0, 0, metas)
        self.assertFalse({metas[i]["e"] for i in fit} & {metas[i]["e"] for i in gate_rows})
        fit, gate_rows = Session.split(policy.replace(split_mode="by_order"), 0, 0, metas)
        self.assertEqual(gate_rows.tolist(), list(range(6, 12)))
        fit, gate_rows = Session.split(policy, 0, 0, [{"e": "only"}] * 10)   # one entity → later rows
        self.assertEqual(gate_rows.tolist(), list(range(5, 10)))

    def test_cap_per_group_spreads_rows_in_time(self):
        items = [(f"line{i}", {"g": "big" if i < 100 else "small"}, 0) for i in range(110)]
        kept = _cap(items, 10, ["g"], [])
        big = [line for line, meta, _ in kept if meta["g"] == "big"]
        self.assertEqual(len(big), 10)
        self.assertEqual((big[0], big[-1]), ("line0", "line99"))
        self.assertEqual(len([1 for _, meta, _ in kept if meta["g"] == "small"]), 10)

    def test_small_samples_widen_the_band_and_need_enough_units(self):
        self.assertGreater(metrics.small_sample_factor(np.asarray(list("abcde"))), 1.3)
        self.assertEqual(metrics.small_sample_factor(np.asarray([str(i) for i in range(500)])), 1.0)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 90)
        strong, weak = np.eye(3)[y], np.eye(3)[(y + 1) % 3]
        few = np.repeat(np.asarray(["u1", "u2", "u3"]), 30)
        result = gate.promotion_gate("classification", y, strong, weak, few, 1.0, self.policy)
        self.assertFalse(result.accepted)
        self.assertIn("too few", result.reason)


class SupervisorTests(Shared):
    def _ctrl(self, health_rows, **policy):
        tmp = Path(tempfile.mkdtemp(dir=self.tmp))
        return SimpleNamespace(policy=self.policy.replace(**policy), run_dir=tmp, task=self.task,
                               state={"health": health_rows, "rows": [{}] * len(health_rows), "next_window": len(health_rows)},
                               window_ids=list(range(20)), ledger=Ledger(), supervisor=None)

    def test_triggers(self):
        sv = Supervisor(None, self.tmp / "mem", self.tmp / "code", use_llm=False)
        ok = {"vs_static": {"diff": 0.0, "band": 0.1, "worse": False}}
        healthy = [dict(ok, window=i) for i in range(4)]
        self.assertEqual(sv.triggers(self._ctrl(healthy), sv.health(self._ctrl(healthy)), 3), [])
        losing = healthy[:2] + [{"window": i, "vs_static": {"diff": -1.0, "band": 0.2, "worse": True}} for i in (2, 3)]
        ctrl = self._ctrl(losing)
        self.assertIn("T2", sv.triggers(ctrl, sv.health(ctrl), 3))
        unreliable = [{"window": i, "promoted": True, "rollback": i % 2 == 1, **ok} for i in range(4)]
        ctrl = self._ctrl(unreliable)
        self.assertIn("T3", sv.triggers(ctrl, sv.health(ctrl), 3))
        ctrl = self._ctrl(healthy, review_every=2)
        self.assertEqual(sv.triggers(ctrl, sv.health(ctrl), 3), ["T8"])

    def test_operator_manual_rules(self):
        p = PolicyConfig()
        sig = {"task_type": "regression", "has_entity": True, "entity": ["machine_tag"], "median_group_size": 40}
        fields = {"machine_tag", "tick"}

        def kind(name, params, policy=p):
            return operators.instantiate(name, params, sig, policy, fields).kind
        self.assertEqual(kind("split.by_order", {}), "tightening")
        self.assertEqual(kind("serve.shadow", {}), "tightening")
        self.assertEqual(kind("gate.z", {"value": 2.5}), "tightening")
        self.assertEqual(kind("gate.z", {"value": 1.5}), "improvement")
        self.assertEqual(kind("serve.trial", {}, p.replace(serve_mode="freeze")), "improvement")
        self.assertEqual(kind("label.fraction", {"fraction": 0.2}), "tightening")
        inst = operators.instantiate("data.cap_per_group", {}, sig, p, fields)     # placeholders filled per task
        self.assertEqual((inst.kind, inst.changes), ("improvement", {"cap_per_group": 40, "cap_group_by": ["machine_tag"]}))
        for name, params, message in (("split.by_group", {"group_by": ["bearing"]}, "do not exist"),
                                      ("gate.z", {"value": 9.0}, "outside"),
                                      ("data.class_balance", {}, "requires"),
                                      ("budget.tokens", {}, "unknown operator"),
                                      ("gate.z", {"value": 2.5, "colour": 1}, "unknown parameters"),
                                      ("code.loss", {"variant": "label_smoothing"}, "does not apply")):
            with self.assertRaises(operators.OperatorError) as raised:
                operators.instantiate(name, params, sig, p, fields)
            self.assertIn(message, str(raised.exception), name)
        edit = operators.instantiate("code.loss", {"variant": "huber"}, sig, p, fields).edits[0]
        self.assertEqual((edit["slot"], edit["variant"]), ("loss_and_delta", "huber"))
        self.assertIn("data.cap_per_group", operators.manual_markdown())

    def test_corrections_are_translated_into_operator_calls(self):
        legacy = {"change": {"split_mode": "by_group", "split_group_by": ["e"], "cap_per_group": 50,
                             "cap_group_by": ["e"], "z": 2.5, "validation": {"holdout": "x"}},
                  "edits": [{"slot": "sample_weights", "source": "def sample_weights(x, target, task, n_classes):\n    return None\n"}]}
        names = [name for name, _ in calls_from(legacy)]
        self.assertEqual(sorted(names), sorted(["data.cap_per_group", "split.by_group", "gate.z", "code.sample_weight"]))
        self.assertEqual(calls_from({"operator": "serve.shadow"}), [("serve.shadow", {})])
        llm = {"corrections": [{"id": "C1", "operator": "split.by_group", "params": {}}]}
        rule = {"corrections": [{"id": "C1", "operator": "split.by_group", "params": {}},
                                {"id": "C2", "operator": "data.cap_per_group", "params": {}}]}
        merged = _backfill(llm, rule)["corrections"]
        self.assertEqual([(c["operator"], c.get("by")) for c in merged],
                         [("split.by_group", None), ("data.cap_per_group", "rule")])

    def test_supervised_run_discovers_structure_and_records_everything(self):
        task = dataclasses.replace(load_task(ROOT / "configs/tasks/B.yaml", ROOT), label_protocol="on_group_complete")
        code = codebase.base_version(task.package, self.tmp / "code_b")
        sv = Supervisor(None, self.tmp / "memory_b", self.tmp / "code_b", use_llm=False)
        summary = Controller(task, self.policy, code, self.tmp / "run_b", supervisor=sv).run()
        self.assertEqual(summary["errors"], 0)
        updates = [r for r in read_jsonl(self.tmp / "run_b" / "transitions.jsonl") if r["type"] == "strategy_update"]
        self.assertTrue(updates)
        first = updates[0]
        self.assertEqual(first["effective_from_window"], 0)
        self.assertEqual(first["policy_changes"].get("split_group_by"), ["bearing"])   # discovered, not configured
        journal = read_jsonl(self.tmp / "run_b" / "journal.jsonl")
        self.assertEqual([r["trigger"] for r in journal if r.get("type") == "round"][0], "T1")
        self.assertTrue(all((r.get("summary") or {}).get("report_paragraph") for r in journal if r.get("type") == "round"))
        self.assertGreaterEqual(summary["supervisor"]["tiers"]["continue"], 1)
        self.assertTrue(build_report(self.tmp / "run_b").is_file())

    def test_memory_is_causal(self):
        memory = Memory(self.tmp / "mem_run", self.tmp / "mem_shared", ["data/sealed"], "run1", "X")
        lesson = {"cue": "c", "lesson": "l", "evidence": "e", "confidence": "low"}
        for r in range(3):
            memory.append({"round": r, "trigger": "T8", "summary": {"report_paragraph": f"p{r}", "lessons": [lesson]}})
            memory.add_lessons(r, [lesson], "v")
        view = memory.prompt_view(before_round=1)
        self.assertEqual([j["round"] for j in view["journal"]], [0])
        self.assertIn("run1-r00", view["library"])
        self.assertNotIn("run1-r02", view["library"])
        with self.assertRaises(LeakError):
            memory.append({"round": 3, "note": "read data/sealed/A"})


class ExplorerTests(Shared):
    def test_llm_explorer_loop_and_fallback(self):
        class Box:
            calls = []

            def call(self, name, args):
                self.calls.append(name)
                return {"echo": name}

        class Scripted:
            model = "scripted"

            def __init__(self, replies):
                self.replies = list(replies)

            def chat_json(self, messages):
                reply = self.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply, 10

        good = Scripted([{"tool": "schema", "args": {}, "why": "look"}, {"plan": {"findings": [], "corrections": []}}])
        plan, trace, tokens, note = explore_llm(Agent(good), Box(), {}, 5, 10_000)
        self.assertEqual((note, [t["tool"] for t in trace], tokens), ("ok", ["schema"], 20))
        broken = Scripted([OSError("offline")])
        self.assertIsNone(explore_llm(Agent(broken), Box(), {}, 5, 10_000)[0])
        nonsense = Scripted([{"hello": 1}, {"hello": 2}, {"hello": 3}])
        self.assertIsNone(explore_llm(Agent(nonsense), Box(), {}, 5, 10_000)[0])

    def test_fixed_protocol_experiment_sees_model_changes_and_no_empty_versions(self):
        ctrl = Controller(self.task, self.policy, self.v0, self.tmp / "fixed_exp", session=self.session, windows=[0, 1, 2, 3])
        ctrl.run()
        ctrl.state["next_window"] = 4
        box = Toolbox(ctrl, None, self.tmp / "fixed_scratch", self.tmp / "code")
        sig = signature.compute(ctrl)
        inst = operators.instantiate("train.lr_scale", {"scale": 0.1}, sig, ctrl.policy)
        result = box.validate(inst)
        self.assertEqual(result["mode"], "past_windows_fixed")
        self.assertGreaterEqual(result["n"], 2)
        self.assertNotEqual(result["baseline_mean"], result["candidate_mean"])   # the gate no longer hides the change
        version = ctrl.state["strategy_version"]
        self.assertEqual(ctrl.apply_strategy(policy=ctrl.policy, reason="same", source="test"), version)

    def test_toolbox_refuses_the_future(self):
        ctrl = Controller(self.task, self.policy, self.v0, self.tmp / "toolbox", session=self.session, windows=[0])
        ctrl.run()
        box = Toolbox(ctrl, None, self.tmp / "toolbox_scratch", self.tmp / "code")
        self.assertIn("not allowed", box.call("compare_subsets", {"a": "base", "b": "window:3"})["error"])
        self.assertIn("unknown tool", box.call("open_file", {})["error"])
        self.assertIn("entity", box.call("find_structure", {}))

    def test_helpers_are_allowed_in_slots(self):
        source = slot_rules.apply_edits(slot_rules.DEFAULT_SOURCE, [{"slot": "learning_rate", "source":
            "_FLOOR = 0.1\n\ndef _decay(step):\n    return 1.0 / (1.0 + step / 1000.0)\n\n"
            "def learning_rate(base_lr, step, epoch):\n    return base_lr * max(_FLOOR, _decay(step))\n"}])
        self.assertEqual(slot_rules.check_module(source), [])
        self.assertEqual(slot_rules.changed_slots(slot_rules.DEFAULT_SOURCE, source), ["learning_rate"])


class RhythmTests(Shared):
    """Three rhythms (adaptive / event / conserve), chosen from experiences by the situation signature."""

    def test_spread_sampling_covers_the_window_and_random_is_unchanged(self):
        spread = Session.label_positions(0, 3, 200, 0.25, "spread")
        self.assertEqual(len(spread), 50)
        self.assertTrue(set(np.diff(spread).tolist()) <= {3, 4, 5})
        legacy = np.sort(np.random.default_rng([0, 3, 1]).choice(200, size=50, replace=False))
        self.assertEqual(Session.label_positions(0, 3, 200, 0.25).tolist(), legacy.tolist())

    def test_rhythm_is_chosen_from_experiences_without_task_names(self):
        from agent_harness.experiences import Experiences, mode_of
        store = Experiences(self.tmp / "exp_shared", "run-x")
        base = {"task_type": "regression", "has_entity": True, "entity": ["tag"], "entity_imbalance": 1.2,
                "countdown": True, "median_group_size": 10, "mode": "standard", "stream_format": "sequence"}
        cases = [({"label_timeliness": None, "drift_exploitable": None}, "event"),     # countdown prior
                 ({"label_timeliness": 0.2, "drift_exploitable": None}, "event"),      # answers arrive late
                 ({"label_timeliness": 1.0, "drift_exploitable": None}, "standard"),   # not known yet: cautious
                 # a running entity's fresh answers mostly teach that entity: never the adaptive rhythm
                 ({"label_timeliness": 1.0, "drift_exploitable": True}, "standard"),
                 ({"label_timeliness": 1.0, "drift_exploitable": False}, "conserve"),
                 # a snapshot stream shows each unit once with its answer
                 ({"label_timeliness": None, "drift_exploitable": None, "stream_format": "snapshot"}, "standard"),
                 ({"label_timeliness": 1.0, "drift_exploitable": True, "stream_format": "snapshot"}, "adaptive")]
        for extra, expected in cases:
            rhythm, _ = store.choose({**base, **extra})
            self.assertEqual(mode_of(rhythm), expected, extra)
        independent = {"task_type": "classification", "has_entity": False, "countdown": False, "class_imbalance": 5.0,
                       "label_timeliness": None, "drift_exploitable": None, "mode": "standard",
                       "stream_format": "independent"}
        rhythm, others = store.choose(independent)
        self.assertEqual(mode_of(rhythm), "adaptive")                # independent rows follow the stream at once
        self.assertEqual([e.id for e in others], ["E-class-balance", "E-rare-class-weight"])
        self.assertEqual(mode_of(store.choose({**independent, "drift_exploitable": False})[0]), "conserve")
        # hysteresis: undecided evidence does not reopen a rhythm that saving chose
        self.assertIsNone(store.choose({**base, "label_timeliness": 1.0, "drift_exploitable": None,
                                        "mode": "conserve"})[0])
        self.assertIsNone(store.choose({**independent, "mode": "conserve"})[0])
        # countdown regression: the over-estimate experiences (training code) apply, not the class ones
        _, regression = store.choose({**base, "label_timeliness": None, "drift_exploitable": None})
        self.assertEqual({e.id for e in regression}, {"E-split-entity", "E-overestimate", "E-relative-error"})
        # verdicts move credibility; the running run sees only its earlier rounds
        before = next(e for e in store.all() if e.id == "E-conserve").credibility
        store.record(["E-conserve"], "reverted", 2, "X")
        self.assertLess(next(e for e in store.all() if e.id == "E-conserve").credibility, before)
        self.assertEqual(next(e for e in store.all(before_round=2) if e.id == "E-conserve").credibility, before)
        self.assertTrue(signature.holds({"all": [{"any": [{"a": 1}, {"b": 2}]}, {"c": {"!=": False}}]},
                                        {"a": 0, "b": 2, "c": None}))

    def test_mode_operators(self):
        p = PolicyConfig()
        sig = {"task_type": "regression", "has_entity": True, "entity": ["tag"]}
        inst = operators.instantiate("mode.event", {}, sig, p)
        self.assertEqual(inst.kind, "mode")
        self.assertEqual((inst.changes["train_trigger"], inst.changes["label_sampling"], inst.changes["label_fraction"]),
                         ("on_completion", "spread", 0.25))
        with self.assertRaises(operators.OperatorError):
            operators.instantiate("mode.event", {}, {"task_type": "classification", "has_entity": False}, p)
        self.assertEqual(operators.instantiate("mode.adaptive", {}, sig, p).changes["gate_mode"], "next_window")
        self.assertEqual([n for n, _ in calls_from({"change": {"gate_mode": "next_window", "mode": "conserve"}})],
                         ["mode.conserve", "gate.mode"])

    def test_next_window_trains_one_candidate_on_every_fresh_label(self):
        policy = self.policy.replace(gate_mode="next_window", mode="adaptive")
        summary = Controller(self.task, policy, self.v0, self.tmp / "direct", session=self.session,
                             windows=[0, 1]).run()
        recipes = [e for e in read_jsonl(self.tmp / "direct" / "events.jsonl") if e["kind"] == "recipe"]
        self.assertEqual([r["purpose"] for r in recipes], ["direct", "direct"])
        fresh0 = summary["per_window"][0]["labels_fresh"]
        self.assertEqual(recipes[0]["rows"], len(self.session.base_lines) + fresh0)   # nothing held back for a gate
        self.assertEqual(summary["promotions"], 2)
        self.assertEqual(summary["errors"], 0)

    def test_on_completion_judges_only_on_finished_entities(self):
        task = dataclasses.replace(load_task(ROOT / "configs/tasks/B.yaml", ROOT), label_protocol="on_group_complete")
        code = codebase.base_version(task.package, self.tmp / "code_rhythm_b")
        policy = self.policy.replace(train_trigger="on_completion", label_sampling="spread", label_fraction=0.25,
                                     mode="event")
        run = self.tmp / "event_b"
        ctrl = Controller(task, policy, code, run, keep_data=True, windows=[0, 1, 2])
        summary = ctrl.run()
        actions = [w["action"] for w in summary["per_window"]]
        self.assertEqual(actions[0], "observe")                      # nothing finished before window 1
        done = [e for e in read_jsonl(run / "events.jsonl") if e["kind"] == "completion"]
        self.assertEqual(len(done), 1)
        finished = set(done[0]["entities"])
        trained = [json.loads(line)["meta"]["bearing"] for line in
                   (run / "data" / "w001" / "event-check.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertFalse(finished & set(trained))                    # the judges were never trained on
        sig = signature.compute(ctrl)
        self.assertTrue(0 < sig["label_timeliness"] < 1)             # some answers arrive only later

    def test_drift_exploitable(self):
        """Judged by what changes of the serving model earned on the next window, never by input movement."""
        def rows(degraded=False, gains=(), trained=False, promoted=False, band=0.01):
            out = [{"drift": {"shifted": True, "degraded": degraded}, "trained": trained, "promoted": promoted,
                    "vs_static": {"diff": 0.0, "band": band, "worse": False}} for _ in range(3)]
            for row, gain in zip(out, gains):
                row["settled"] = {"gain": gain, "changed": True}
            return out
        self.assertIsNone(signature.drift_exploitable(rows(), 3))                     # moving inputs prove nothing
        self.assertTrue(signature.drift_exploitable(rows(gains=(0.1, 0.1)), 3))
        self.assertIsNone(signature.drift_exploitable(rows(gains=(0.01, 0.005), band=0.1), 3))   # within noise
        self.assertFalse(signature.drift_exploitable(rows(gains=(-0.1, 0.05)), 3))
        self.assertFalse(signature.drift_exploitable(rows(trained=True), 3))          # trained, nothing promoted
        self.assertIsNone(signature.drift_exploitable(rows(trained=True, degraded=True), 3))
        self.assertIsNone(signature.drift_exploitable(rows(gains=(0.2,)), 3))         # one settled change only

    def test_supervisor_switches_rhythm_and_no_stall_freezes(self):
        from agent_harness import fallback_explorer
        sv = Supervisor(None, self.tmp / "mem_rhythm", self.tmp / "code", use_llm=False)
        ctrl = Controller(self.task, self.policy, self.v0, self.tmp / "rhythm_ctrl", session=self.session,
                          windows=[0, 1, 2], supervisor=None)
        ctrl.run()
        ctrl.supervisor = sv
        ctrl.policy = ctrl.policy.replace(mode="event")
        self.assertTrue(sv._mode_mismatch(ctrl))                     # independent rows: the adaptive rhythm fits
        sv._state(ctrl)["mode_since"] = 1
        self.assertFalse(sv._mode_mismatch(ctrl))                    # a rhythm gets probation_windows windows first
        sv._state(ctrl).update(mode_since=0, mode_held=True)
        self.assertFalse(sv._mode_mismatch(ctrl))                    # a confirmed rhythm moves only on a loss
        self.assertTrue(sv._mode_mismatch(ctrl, emergency=True))
        sv._state(ctrl)["mode_held"] = False
        ctrl.policy = ctrl.policy.replace(mode="adaptive")
        self.assertFalse(sv._mode_mismatch(ctrl))
        box = Toolbox(ctrl, sv.memory, self.tmp / "rhythm_box", self.tmp / "code")
        stall = fallback_explorer.explore(box, "T6", ctrl)
        names = [c["operator"] for c in stall["corrections"]]
        self.assertNotIn("serve.freeze", names)
        self.assertEqual(names[0], "mode.adaptive")
        rhythm, _ = sv.choose(ctrl, signature.compute(ctrl))
        ctrl.policy = ctrl.policy.replace(mode="event")
        switch = fallback_explorer.explore(box, "T10", ctrl, rhythm=rhythm, fired=["T10"])
        self.assertEqual([(c["operator"], c["experience"]) for c in switch["corrections"]],
                         [("mode.adaptive", "E-adaptive")])


class PipelineTests(Shared):
    """The unified pipeline: generic structure, evidence-gated rhythm, cumulative safety net, paid opening
    experiments, rules before the LLM."""

    @staticmethod
    def _view(session):
        """What signature.compute needs before the stream starts."""
        return SimpleNamespace(session=session, stream=session.new_stream(), window_ids=[0], state={},
                               policy=PolicyConfig())

    def test_entity_key_and_stream_format_are_discovered(self):
        task_b = load_task(ROOT / "configs/tasks/B.yaml", ROOT)
        self.assertEqual(task_b.group_key, [])                       # nothing task-specific is configured
        session_b = Session(task_b, codebase.base_version(task_b.package, self.tmp / "code_b_pipe"))
        self.assertEqual(session_b.group_key, ["bearing"])
        self.assertEqual(signature.stream_format(self._view(session_b), session_b.entity), "sequence")
        self.assertEqual(signature.stream_format(self._view(self.session), self.session.entity), "independent")
        snapshot = ROOT / "data/dev/C_validation_snapshot.jsonl"
        if not snapshot.is_file():
            import runpy
            runpy.run_path(str(ROOT / "scripts/make_public_snapshot.py"), run_name="__main__")
        task_c = load_task(ROOT / "configs/tasks/C_snapshot.yaml", ROOT)
        session_c = Session(task_c, codebase.base_version(task_c.package, self.tmp / "code_c_pipe"))
        self.assertEqual(set(session_c.group_key), {"source_subset", "unit"})
        self.assertEqual(len(session_c.base_rows), 20_000)          # the generic budget rule, no per-task cap
        self.assertEqual(signature.stream_format(self._view(session_c), session_c.entity), "snapshot")

    def test_renamed_fields_lead_to_the_same_decisions(self):
        """Rename every meta field of a public task: structure, signature and chosen experiences must not change."""
        from agent_harness.experiences import Experiences, mode_of
        task_b = load_task(ROOT / "configs/tasks/B.yaml", ROOT)
        renamed = self.tmp / "pkg_renamed"
        shutil.copytree(task_b.package, renamed, ignore=shutil.ignore_patterns("public"))
        (renamed / "public").mkdir()
        for split in ("train", "validation"):
            lines = []
            for line in (task_b.package / "public" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["meta"] = {f"field_{i}": v for i, v in enumerate(row["meta"].values())}
                lines.append(json.dumps(row, ensure_ascii=False))
            (renamed / "public" / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store = Experiences(self.tmp / "exp_renamed", "run-r")
        out = []
        for package in (task_b.package, renamed):
            task = dataclasses.replace(task_b, package=package)
            session = Session(task, codebase.base_version(package, self.tmp / f"code_{package.name}"))
            sig = signature.compute(self._view(session))
            rhythm, others = store.choose(sig)
            out.append(({k: v for k, v in sig.items() if k not in ("entity", "order")}, mode_of(rhythm),
                        [e.id for e in others], len(session.entity)))
        self.assertEqual(out[0], out[1])
        self.assertEqual(out[0][1], "event")

    def test_label_timeliness_needs_evidence(self):
        one = Controller(self.task, self.policy, self.v0, self.tmp / "timely1", session=self.session, windows=[0])
        one.run()
        self.assertIsNone(signature.compute(one)["label_timeliness"])     # one window can mislead
        three = Controller(self.task, self.policy, self.v0, self.tmp / "timely3", session=self.session,
                           windows=[0, 1, 2])
        three.run()
        self.assertEqual(signature.compute(three)["label_timeliness"], 1.0)

    def test_cumulative_checks_catch_small_repeated_losses(self):
        self.assertFalse(gate.cumulative_worse([[-1.0, 2.0]]))
        self.assertFalse(gate.cumulative_worse([[-1.0, 2.0], [1.5, 2.0]]))
        self.assertTrue(gate.cumulative_worse([[-1.5, 2.0], [-1.5, 2.0], [-1.5, 2.0]]))   # -4.5 < -3.46

    def test_fall_back_returns_to_the_opening_model(self):
        policy = self.policy.replace(gate_mode="next_window", mode="adaptive")
        ctrl = Controller(self.task, policy, self.v0, self.tmp / "fallback", session=self.session, windows=[0, 1])
        ctrl.run()
        nodes = sorted(n for n in ctrl.tree.nodes if n != "n0000")
        ctrl.state["floor_node"] = nodes[0]
        ctrl.tree.pointers.update(confirmed=nodes[-1], provisional=None)
        refs = [(1, q) for q in ctrl.state["pool"]["1"]]
        lx, ly, _, lg = ctrl._arrays(refs)
        ctrl._fall_back(1, ly, lx, lg, -1.0, 0.1)
        self.assertIn(ctrl.tree.serving, (nodes[0], "n0000"))
        events = [e["kind"] for e in read_jsonl(self.tmp / "fallback" / "events.jsonl")]
        self.assertTrue({"reset_to_opening", "reset_to_static"} & set(events))

    def test_opening_experiments_are_budgeted_paid_and_share_the_baseline(self):
        ctrl = Controller(self.task, self.policy.replace(opening_rows=300), self.v0, self.tmp / "opening_cost",
                          session=self.session, windows=[0])
        ctrl._setup(False)
        box = Toolbox(ctrl, None, self.tmp / "opening_scratch", self.tmp / "code")
        sig = signature.compute(ctrl)
        first = box.validate(operators.instantiate("data.class_balance", {"n": 60}, sig, ctrl.policy))
        self.assertEqual(first["mode"], "group_kfold")
        self.assertEqual(first["train_rows_per_fold"], 300)
        self.assertEqual(ctrl.executor.row_epochs, first["spent_row_epochs"])     # counted in the run's cost
        second = box.validate(operators.instantiate("code.sample_weight", {"variant": "sqrt_inverse_class_frequency"},
                                                    sig, ctrl.policy))
        self.assertLess(second["spent_row_epochs"], first["spent_row_epochs"])   # baseline folds reused
        self.assertEqual(ctrl.state["experiment_row_epochs"], first["spent_row_epochs"] + second["spent_row_epochs"])

    def test_cut_point_scoring_keeps_a_few_rows_per_entity(self):
        from agent_harness.experiments import _cut_points
        meta = [{"unit": u, "cycle": c} for u in range(4) for c in range(10)]
        table = _synthetic_table(meta, np.zeros(len(meta)))
        rows = _cut_points(table, np.arange(len(meta)), ["unit"], seed=0)
        self.assertEqual(len(rows), 12)
        self.assertEqual(sorted({meta[i]["unit"] for i in rows}), [0, 1, 2, 3])

    def test_opening_may_adopt_one_data_and_one_code_change_only_together(self):
        sig = {"task_type": "regression", "has_entity": True, "entity": ["tag"], "median_group_size": 10}
        data = operators.instantiate("data.cap_per_group", {"group_by": ["tag"], "n": 10}, sig, PolicyConfig(),
                                     {"tag"})
        code = operators.instantiate("code.loss", {"variant": "asymmetric_overestimate_x4"}, sig, PolicyConfig())
        items = [(0.10, {"id": "C1"}, data, {"mean_diff": 1.0, "band": 0.5}),
                 (0.05, {"id": "C2"}, code, {"mean_diff": 0.5, "band": 0.2})]
        sv = Supervisor(None, self.tmp / "mem_combo", self.tmp / "code", use_llm=False)
        good = SimpleNamespace(validate=lambda inst, spec, tag="": {"passed": True, "mean_diff": 1.2, "band": 0.4})
        bad = SimpleNamespace(validate=lambda inst, spec, tag="": {"passed": True, "mean_diff": 0.8, "band": 0.4})
        self.assertEqual([c[1]["id"] for c in sv._choose_improvements(good, items, "T1")], ["C1", "C2"])
        self.assertEqual([c[1]["id"] for c in sv._choose_improvements(bad, items, "T1")], ["C1"])   # not better together
        self.assertEqual([c[1]["id"] for c in sv._choose_improvements(good, items, "T6")], ["C1"])  # in-run: one change

    def test_llm_only_where_the_rules_do_not_reach(self):
        class Silent:
            model = "silent"

            def chat_json(self, messages):
                raise AssertionError("must not be called")
        ctrl = Controller(self.task, self.policy, self.v0, self.tmp / "llm_gate", session=self.session, windows=[0])
        ctrl.run()
        sv = Supervisor(Agent(Silent()), self.tmp / "mem_llm_gate", self.tmp / "code", use_llm=True)
        rhythm, _ = sv.choose(ctrl, signature.compute(ctrl))
        self.assertFalse(sv._llm_wanted(ctrl, "T10", rhythm)[0])            # a rhythm switch is computed
        self.assertFalse(sv._llm_wanted(ctrl, "T1", rhythm)[0])             # a credible experience fits
        self.assertTrue(sv._llm_wanted(ctrl, "T1", None)[0])
        self.assertFalse(sv._llm_wanted(ctrl, "T2", rhythm)[0])
        sv._state(ctrl)["outcomes"]["reverted"] = 1
        self.assertTrue(sv._llm_wanted(ctrl, "T2", rhythm)[0])              # the rules already failed
        self.assertTrue(sv._llm_allowed(ctrl, 12_000))
        ctrl.ledger.charge(Cost(llm_tokens=25_000))
        self.assertFalse(sv._llm_allowed(ctrl, 12_000))                     # the run cap holds

    def test_new_variants_pass_the_official_tests(self):
        for name in ("asymmetric_overestimate_x4", "relative_error"):
            variant = next(v for v in slot_rules.VARIANTS if v.name == name)
            source = slot_rules.apply_edits(slot_rules.DEFAULT_SOURCE, [variant.edit()])
            version = codebase.derive_version(self.v0, source, [variant.edit()], self.tmp / "code")
            ok, detail = sandbox.check(version, sandbox.reference_failures(self.v0, self.tmp / "sandbox_ref.json"))
            self.assertTrue(ok, (name, detail))


if __name__ == "__main__":
    unittest.main()
