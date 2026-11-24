"""Build a public stream in the official C test format: each unit shown once, cut at a random cycle.

Only the PUBLIC validation split of the C package is read. Every engine is cut at a random cycle (its answer
is then the remaining life at that cycle, exactly what the official test file gives for the last observed
cycle). Three passes over the engines, each with a fresh cut and a fresh engine order, give a stream long
enough for the rhythm to act; inside a pass every engine appears once.

    python scripts/make_public_snapshot.py   ->  data/dev/C_validation_snapshot.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpu-c-guide-20261110-v1" / "public" / "validation.jsonl"
TARGET = ROOT / "data" / "dev" / "C_validation_snapshot.jsonl"
PASSES = 3
SEED = 20261115


def main() -> None:
    engines: dict[str, list[str]] = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            meta = json.loads(line)["meta"]
            engines.setdefault(f"{meta['source_subset']}|{meta['unit']}", []).append(line)
    rng = np.random.default_rng(SEED)
    keys = sorted(engines)
    # distinct cut cycles per engine across passes (ids stay unique); never the first 10% of a life
    cuts = {k: rng.choice(np.arange(int(0.1 * len(engines[k])), len(engines[k])), size=PASSES, replace=False)
            for k in keys}
    out = []
    for p in range(PASSES):
        for i in rng.permutation(len(keys)):
            out.append(engines[keys[i]][int(cuts[keys[i]][p])])
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{TARGET.relative_to(ROOT).as_posix()}: {len(out)} rows, {len(keys)} engines x {PASSES} passes")


if __name__ == "__main__":
    main()
