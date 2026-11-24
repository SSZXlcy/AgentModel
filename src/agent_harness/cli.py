"""Command line.

  run            one pass over the (simulated) stream: inner loop + supervisor
  compare        static / naive / inner / system (/ system with LLM) on the same seeds
  report         Markdown report of a run
  llm-check      one tiny call to the configured LLM (no project data)
  evolve-code    outer layer offline: evolve slots.py
  evolve-policy  outer layer offline: evolve π
  inspect        print a run summary
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from .agent import Agent, OpenAICompatibleProvider
from .config import load_policy, load_task
from .controller import Controller
from .evolution import CodeEvolver, PolicyEvolver, promoted_code, promoted_policy
from .supervisor import Supervisor
from .tracking import NullTracker, SwanLabTracker


def _agent(args) -> Agent:
    return Agent(OpenAICompatibleProvider() if getattr(args, "llm", False) else None)


def _task(args):
    """--task-config names any task file (e.g. a longer public stream); --task the standard one."""
    if getattr(args, "task_config", None):
        task = load_task(args.task_config)
        args.task = task.task
    elif args.task:
        task = load_task(Path(args.configs) / "tasks" / f"{args.task}.yaml")
    else:
        raise SystemExit("give --task or --task-config")
    protocol = getattr(args, "label_protocol", None)
    return dataclasses.replace(task, label_protocol=protocol) if protocol else task


def _label(args) -> str:
    return Path(args.task_config).stem if getattr(args, "task_config", None) else args.task


def _artifacts(args) -> Path:
    return Path(args.artifacts) / args.task


def _compact(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in ("per_window", "rounds", "trials")}


def cmd_run(args) -> None:
    task = _task(args)
    policy = promoted_policy(_artifacts(args), load_policy(args.policy))
    code = promoted_code(task, _artifacts(args))
    run_dir = Path(args.resume) if args.resume else (
        Path(args.run_root) / _label(args) / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    tracker = NullTracker()
    if args.swanlab:
        try:
            tracker = SwanLabTracker(args.swanlab_project, args.swanlab_experiment or run_dir.name,
                                     {"task": task.to_dict(), "policy": policy.to_dict(),
                                      "code_version": code.version_id}, mode=args.swanlab_mode)
        except Exception as error:   # tracking must never stop a run
            print(f"SwanLab unavailable, continuing with local records only: {error}", file=sys.stderr)
    agent = _agent(args)
    supervisor = None if args.no_supervise else Supervisor(
        agent, Path(args.artifacts) / "memory", _artifacts(args) / "code", use_llm=args.llm)
    summary = Controller(task, policy, code, run_dir, agent=agent, label_seed=args.label_seed, tracker=tracker,
                         keep_data=args.keep_data, supervisor=supervisor).run(resume=bool(args.resume))
    if supervisor is not None:
        from .report import build
        summary["report"] = build(run_dir).as_posix()
    print(json.dumps(_compact(summary), ensure_ascii=False, indent=2))


def cmd_compare(args) -> None:
    from .compare import compare
    task = _task(args)
    provider = OpenAICompatibleProvider() if args.llm else None
    root = Path(args.run_root) / "compare" / f"{_label(args)}_{task.label_protocol}"
    result = compare(task, load_policy(args.policy), promoted_code(task, _artifacts(args)), args.seeds, root,
                     _artifacts(args) / "code", provider=provider, arms=args.arms)
    (root / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_report(args) -> None:
    from .report import build
    print(build(Path(args.run_dir)).as_posix())


def cmd_manual(args) -> None:
    from .operators import manual_markdown
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manual_markdown(), encoding="utf-8")
    print(path.as_posix())


def cmd_llm_check(args) -> None:
    import time
    provider = OpenAICompatibleProvider()
    started = time.monotonic()
    try:
        answer, tokens = provider.generate_json("Reply with one JSON object only.",
                                                {"request": "Return {\"ok\": true, \"model\": \"<your model name>\"}"})
        print(json.dumps({"ok": True, "model": provider.model, "answer": answer, "tokens": tokens,
                          "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "model": provider.model, "error": f"{type(error).__name__}: {str(error)[:300]}"},
                         ensure_ascii=False))


def cmd_evolve_code(args) -> None:
    task = _task(args)
    summary = CodeEvolver(task, load_policy(args.policy), _artifacts(args), agent=_agent(args),
                          seeds=args.seeds, rounds=args.rounds).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_evolve_policy(args) -> None:
    task = _task(args)
    summary = PolicyEvolver(task, load_policy(args.policy), _artifacts(args), seeds=args.seeds).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_inspect(args) -> None:
    summary = json.loads((Path(args.run_dir) / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(_compact(summary), ensure_ascii=False, indent=2))
    for row in summary["per_window"]:
        print(json.dumps(row, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="agent-harness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--artifacts", default="artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, llm=True):
        p.add_argument("--task", choices=["A", "B", "C"])
        p.add_argument("--task-config", help="any task YAML (overrides --task), e.g. configs/tasks/A_long.yaml")
        p.add_argument("--policy", default="configs/policies/default.yaml")
        if llm:
            p.add_argument("--llm", action="store_true", help="use the OpenAI-compatible provider (LLM_API_KEY)")

    def protocol(p):
        p.add_argument("--label-protocol", choices=["immediate", "on_group_complete"],
                       help="simulated platform: when labels exist (default from the task config)")

    run = sub.add_parser("run")
    common(run)
    protocol(run)
    run.add_argument("--no-supervise", action="store_true", help="inner loop only, no strategy updates")
    run.add_argument("--run-root", default="runs")
    run.add_argument("--resume", help="continue an interrupted run directory")
    run.add_argument("--label-seed", type=int, default=0)
    run.add_argument("--keep-data", action="store_true")
    run.add_argument("--swanlab", action="store_true")
    run.add_argument("--swanlab-project", default="agent-harness")
    run.add_argument("--swanlab-experiment")
    run.add_argument("--swanlab-mode", default="cloud", choices=["cloud", "offline", "local", "disabled"])
    run.set_defaults(func=cmd_run)

    code = sub.add_parser("evolve-code")
    common(code)
    code.add_argument("--rounds", type=int, default=3)
    code.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    code.set_defaults(func=cmd_evolve_code)

    pol = sub.add_parser("evolve-policy")
    common(pol, llm=False)
    pol.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    pol.set_defaults(func=cmd_evolve_policy)

    cmp = sub.add_parser("compare")
    common(cmp)
    protocol(cmp)
    cmp.add_argument("--run-root", default="runs")
    cmp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cmp.add_argument("--arms", nargs="+", choices=["naive", "inner", "system", "system_llm"],
                     help="re-run only these arms and merge them into the stored compare.json")
    cmp.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    check = sub.add_parser("llm-check")
    check.set_defaults(func=cmd_llm_check)

    manual = sub.add_parser("manual", help="write the operator manual (修改说明书) as Markdown")
    manual.add_argument("--output", default="docs/修改说明书.md")
    manual.set_defaults(func=cmd_manual)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("run_dir")
    inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    args.func(args)
