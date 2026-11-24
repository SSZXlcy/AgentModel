"""Publish the experiment records under runs/ to a SwanLab cloud project.

Every run directory with a summary.json becomes one SwanLab experiment:
  config   the run's manifest (task, policy, code version, data digests, environment), arm, seed,
           source directory and the time the run started and finished (first / last event time stamp)
  metrics  per window: utility, static utility, gain over static, labels, promotions, strategy version
           per training call: rows, training seconds, validation after training
           summary: gain over static, worst window, cost (row-epochs, labels, tokens), promotions, rollbacks
Every compare.json / loto.json becomes one aggregate experiment with each arm's scores and costs.

All streams under runs/ are the public validation projection (or a snapshot cut from it); the manifest
only *lists* data/raw and data/sealed as forbidden paths. The script refuses any run whose stream is not
the public validation projection. The key is read from SWANLAB_API_KEY or ~/.agent_harness/swanlab_key.

    python scripts/swanlab_upload.py --dry-run
    python scripts/swanlab_upload.py --project AgentModel-zhixun-harness
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
KEY_FILE = Path.home() / ".agent_harness" / "swanlab_key"
LEDGER = RUNS / "swanlab_uploaded.json"      # names already published; a rerun skips them
GROUPS = {"compare_before_rhythm": "第二版对照（节奏模式之前）", "compare": "第三版对照（三种训练节奏）",
          # runs/v3 holds the fourth version's comparison (the third round of compare runs, hence the name)
          "v3": "第四版对照（统一 pipeline）", "loto": "第四版·留一任务验证", "e2e": "端到端验收",
          "e2e_llm_offline": "端到端验收（27B 离线回放）"}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _beijing(stamp: str | None) -> str:
    if not stamp:
        return "未知"
    from datetime import datetime, timedelta, timezone
    return datetime.fromisoformat(stamp).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")


def _number(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def runs() -> list[Path]:
    found = [p.parent for p in RUNS.rglob("summary.json") if "swanlab" not in p.parts]
    return sorted(found, key=lambda d: (_lines(d / "events.jsonl")[:1] or [{"ts": ""}])[0].get("ts", ""))


def aggregates() -> list[Path]:
    return sorted(p for p in RUNS.rglob("*.json") if p.name in ("compare.json", "loto.json") and "swanlab" not in p.parts)


def describe(run: Path) -> dict:
    manifest = _json(run / "manifest.json") if (run / "manifest.json").is_file() else {}
    summary = _json(run / "summary.json")
    events = _lines(run / "events.jsonl")
    transitions = _lines(run / "transitions.jsonl")
    data = manifest.get("data") or {}
    if data.get("stream_split") not in (None, "validation"):
        raise ValueError(f"{run}: stream is {data.get('stream_split')!r}, not the public validation projection")
    rel = run.relative_to(RUNS)
    parts = rel.parts
    group = GROUPS.get(parts[0], parts[0])
    arm = next((p for p in parts if p in ("static", "naive", "inner", "system", "system_llm")), None)
    seed = next((p for p in parts if p.startswith("seed")), None)
    started = events[0]["ts"] if events else None
    finished = events[-1]["ts"] if events else None
    task = summary.get("task")
    config = {"source": rel.as_posix(), "group": group, "arm": arm, "seed": seed, "task": task,
              "label_protocol": summary.get("label_protocol"), "run_started_at": started, "run_finished_at": finished,
              "code_version": summary.get("code_version"), "policy_id": summary.get("policy_id"),
              "manifest": manifest,
              "strategy_updates": [{"version": t.get("version"), "window": t.get("effective_from_window"),
                                    "source": t.get("source"), "reason": str(t.get("reason", ""))[:400]}
                                   for t in transitions if t.get("type") == "strategy_update"]}
    return {"run": run, "name": " | ".join(parts), "group": group, "tags": [x for x in (task, arm, parts[0]) if x],
            "config": config, "summary": summary, "events": events}


def log_run(swanlab, item: dict) -> None:
    summary = item["summary"]
    for row in summary.get("per_window") or []:
        step = int(row.get("window", 0))
        metrics = {f"window/{k}": v for k, v in ((k, _number(v)) for k, v in row.items() if k != "window") if v is not None}
        if "utility" in row and "static_utility" in row:
            metrics["window/gain_vs_static"] = row["utility"] - row["static_utility"]
        swanlab.log(metrics, step=step)
    trains = [e for e in item["events"] if e.get("kind") == "train"]
    for i, e in enumerate(trains):
        metrics = {"train/rows": e.get("rows"), "train/training_seconds": e.get("training_seconds"),
                   "train/window": e.get("window"), "train/from_scratch": int(bool(e.get("from_scratch")))}
        for k, v in (e.get("validation_after") or {}).items():
            if _number(v) is not None:
                metrics[f"train/validation_after_{k}"] = v
        swanlab.log({k: v for k, v in metrics.items() if v is not None}, step=i)
    cost = summary.get("cost") or {}
    final = {"summary/gain_vs_static": summary.get("gain_vs_static"), "summary/mean_utility": summary.get("mean_utility"),
             "summary/static_mean_utility": summary.get("static_mean_utility"),
             "summary/worst_utility": summary.get("worst_utility"), "summary/final_utility": summary.get("final_utility"),
             "summary/promotions": summary.get("promotions"), "summary/rollbacks": summary.get("rollbacks"),
             "summary/resets_to_static": summary.get("resets_to_static"), "summary/errors": summary.get("errors"),
             "summary/train_row_epochs": summary.get("train_row_epochs"), "summary/labels": cost.get("labels"),
             "summary/llm_tokens": cost.get("llm_tokens"), "summary/train_seconds": cost.get("train_seconds"),
             "summary/strategy_versions": summary.get("strategy_versions")}
    swanlab.log({k: _number(v) for k, v in final.items() if _number(v) is not None}, step=0)


def describe_aggregate(path: Path) -> dict:
    data = _json(path)
    rel = path.relative_to(RUNS)
    group = GROUPS.get(rel.parts[0], rel.parts[0])
    return {"name": "汇总 | " + " | ".join(rel.parts[:-1]), "group": group, "tags": ["aggregate", rel.parts[0]],
            "config": {"source": rel.as_posix(), "group": group, "content": data}, "data": data}


def log_aggregate(swanlab, item: dict) -> None:
    metrics = {}
    for arm, values in (item["data"].get("arms") or {}).items():
        if isinstance(values, dict):
            for k, v in values.items():
                if _number(v) is not None:
                    metrics[f"{arm}/{k}"] = v
    if metrics:
        swanlab.log(metrics, step=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="AgentModel-zhixun-harness")
    parser.add_argument("--workspace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = [("run", describe(r)) for r in runs()] + [("aggregate", describe_aggregate(a)) for a in aggregates()]
    if args.limit:
        items = items[:args.limit]
    if args.dry_run:
        for kind, item in items:
            c = item["config"]
            print(kind, "|", item["name"], "|", c.get("run_started_at", ""), "|", item["group"])
        print(len(items), "experiments")
        return
    import swanlab
    key = os.environ.get("SWANLAB_API_KEY") or (KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.is_file() else "")
    if not key:
        raise SystemExit("no SwanLab key: set SWANLAB_API_KEY or create ~/.agent_harness/swanlab_key")
    swanlab.login(api_key=key)
    published = set(json.loads(LEDGER.read_text(encoding="utf-8"))) if LEDGER.is_file() else set()
    done = 0
    for kind, item in items:
        if item["name"] in published:
            continue
        run = swanlab.init(project=args.project, workspace=args.workspace, name=item["name"][:100],
                           group=item["group"], tags=item["tags"], config=item["config"], mode="online",
                           description=f"来源：runs/{item['config']['source']}" +
                                       (f"；运行时间 {_beijing(item['config'].get('run_started_at'))}（北京时间）" if kind == "run" else ""))
        (log_run if kind == "run" else log_aggregate)(swanlab, item)
        swanlab.finish()
        published.add(item["name"])
        LEDGER.write_text(json.dumps(sorted(published), ensure_ascii=False, indent=0), encoding="utf-8")
        done += 1
        print(f"[{done}/{len(items)}] {item['name']}", flush=True)
        time.sleep(0.2)
    print("project:", args.project)


if __name__ == "__main__":
    main()
