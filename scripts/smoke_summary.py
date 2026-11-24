"""Summarise an sdk_smoke.py log: one line per round, then the prediction checks and model calls."""
import json
import re
import sys

rows, calls, served, signatures, finished = [], [], [], {}, {}
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith('{"dataset"'):
        r = json.loads(line)
        (rows if "round" in r else served).append(r)
    elif line.startswith('{"live_call"'):
        calls.append(json.loads(line))
    elif line.startswith("[harness] round ") and " dataset " in line:
        m = re.match(r"\[harness\] round (\d+) dataset (\w): (.*)$", line.strip())
        if m:
            signatures[(m.group(2), int(m.group(1)))] = json.loads(m.group(3))
    elif "round finished" in line:
        finished.setdefault("lines", []).append(line.strip())
for r in sorted(rows, key=lambda r: (r["dataset"], r["round"])):
    sig = signatures.get((r["dataset"], r["round"]), {})
    if "failed" in r:
        print(r["dataset"], r["round"], "FAILED", r["failed"])
        continue
    c = r["config"]
    gains = {k: f"{v['gain']:+.4f}/sd {v['sd']:.4f}" for k, v in (c.get("paired_vs_parent") or {}).items()}
    print(f"{r['dataset']} r{r['round']:<2} {r['seconds']:>6.1f}s  {c.get('rhythm', ''):<9} -> {c.get('selected_arm', ''):<8}"
          f" entity={sig.get('entity', '(skip)')} src={sig.get('entity_source', '')} fmt={sig.get('stream_format', '')} {gains}")
for s in served:
    print("served", s)
for c in calls:
    u = c.get("usage") or {}
    print("call", c.get("tag"), c.get("seconds"), c.get("failed", ""), u.get("prompt_tokens"), u.get("completion_tokens"))
