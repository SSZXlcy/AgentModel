"""Gate ② of the outer layer: a candidate code version must pass the official test-suite in a
separate process, with a timeout, before it is ever imported into the harness process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .codebase import CodeVersion

_LINE = re.compile(r"^(test_\w+) \([^)]*\) \.\.\. (ok|FAIL|ERROR|skipped.*)$", re.M)


def official_tests(version: CodeVersion, timeout: int = 900) -> dict:
    env = {**os.environ, "BASELINE_CPU_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "baselines/tests", "-v"],
                              cwd=version.root, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "passed": [], "failed": ["<timeout>"], "seconds": timeout, "tail": "timeout"}
    output = done.stdout + done.stderr
    results = _LINE.findall(output)
    return {"returncode": done.returncode, "seconds": time.monotonic() - started,
            "passed": sorted(name for name, status in results if status == "ok"),
            "failed": sorted(name for name, status in results if status in ("FAIL", "ERROR")),
            "tail": output[-1500:]}


def reference_failures(base: CodeVersion, cache: Path) -> set[str]:
    """Tests that already fail on v0 (the B package ships tests needing the platform's unshipped
    `scripts/` module). A candidate is judged only on failures it introduces."""
    if cache.is_file():
        return set(json.loads(cache.read_text(encoding="utf-8"))["failed"])
    result = official_tests(base)
    if not result["passed"]:
        raise RuntimeError("official tests did not run on the base version:\n" + result["tail"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"version": base.version_id, **result}, indent=2), encoding="utf-8")
    return set(result["failed"])


def step_benchmark(version: CodeVersion, checkpoint: Path, x, y, repeats: int = 5) -> float:
    """Median seconds of one fit() epoch on fixed rows. Comparing two versions this way measures
    what a slot change costs per step, free of the noise of whole replays."""
    Model = version.modules().model.Model
    times = []
    for _ in range(repeats):
        model = Model.load(checkpoint)
        started = time.perf_counter()
        model.fit(x, y, 1)
        times.append(time.perf_counter() - started)
    return float(sorted(times)[len(times) // 2])


def check(version: CodeVersion, known_failures: set[str]) -> tuple[bool, dict]:
    result = official_tests(version)
    new = sorted(set(result["failed"]) - known_failures)
    ok = bool(result["passed"]) and not new
    return ok, {"passed": len(result["passed"]), "new_failures": new, "seconds": result["seconds"],
                "tail": "" if ok else result["tail"]}
