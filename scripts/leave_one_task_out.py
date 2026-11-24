"""Leave-one-task-out check of the generic harness (public data only).

For each target task: the full system (deterministic explorer, no LLM) first runs on the two other tasks with
one shared memory, so their probation verdicts move the credibility of the shared experiences; then the target
runs twice on the same seed, once with that memory ("transferred") and once with an empty one ("cold"). Same
code, same configuration rules, no task-specific setting anywhere.

    python scripts/leave_one_task_out.py [--root runs/loto]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent_harness  # noqa: E402,F401  (thread variables before NumPy)
from agent_harness import codebase  # noqa: E402
from agent_harness.config import load_policy, load_task  # noqa: E402
from agent_harness.controller import Controller  # noqa: E402
from agent_harness.experiences import Experiences  # noqa: E402
from agent_harness.supervisor import Supervisor  # noqa: E402

TASKS = {"A_long": ("configs/tasks/A_long.yaml", "immediate"), "B": ("configs/tasks/B.yaml", "on_group_complete"),
         "C": ("configs/tasks/C.yaml", "immediate")}


def run(name: str, memory: Path, out: Path, code_root: Path) -> dict:
    path, protocol = TASKS[name]
    import dataclasses
    task = dataclasses.replace(load_task(ROOT / path, ROOT), label_protocol=protocol)
    code = codebase.base_version(task.package, code_root / name)
    supervisor = Supervisor(None, memory, code_root / name, use_llm=False)
    summary = Controller(task, load_policy(ROOT / "configs/policies/default.yaml"), code, out,
                         supervisor=supervisor).run()
    modes = []
    for item in sorted((out / "strategies").glob("s*.json")):
        mode = json.loads(item.read_text(encoding="utf-8"))["policy"].get("mode")
        if not modes or modes[-1][0] != mode:
            modes.append((mode, json.loads(item.read_text(encoding="utf-8"))["window"]))
    return {"gain_vs_static": round(summary["gain_vs_static"], 4), "row_epochs": summary["train_row_epochs"],
            "labels": summary["cost"]["labels"], "modes": modes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/loto")
    root = ROOT / parser.parse_args().root
    shutil.rmtree(root, ignore_errors=True)
    report = {}
    for target in TASKS:
        others = [t for t in TASKS if t != target]
        shared = root / target / "memory"
        for other in others:
            run(other, shared, root / target / f"warmup_{other}", root / "code")
        credibility = {e.id: round(e.credibility, 3) for e in Experiences(shared, "report").all()}
        report[target] = {"warmed_on": others, "credibility_after_warmup": credibility,
                          "transferred": run(target, shared, root / target / "transferred", root / "code"),
                          "cold": run(target, root / target / "cold_memory", root / target / "cold", root / "code")}
        print(target, json.dumps(report[target], ensure_ascii=False))
    (root / "loto.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
