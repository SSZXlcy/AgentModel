"""Local end-to-end check of the platform adapter with a JSONL host stand-in (no real model, no network).

For each dataset it runs two training rounds through `/app/harness.py train` exactly as the platform
would (start frame on stdin, RPC frames on stdout), carrying the selected artifact and the saved state
into the second round, then serves the selected artifact through `harness.py predict` on the public
validation inputs and checks the scores against the adapter's own local measurement.

    python scripts/sdk_smoke.py --sdk <contest-harness-sdk dir> [--datasets A B C] [--docker IMAGE]

With --docker the rounds run inside the image as the platform runs it: no network, read-only root,
/data, /previous, /state and /model mounted read-only, only /workspace writable, 2 CPUs and 4 GiB.

With --live the host stand-in forwards every model request to a real chat-completions endpoint, the way
the platform gateway does: the official prompt (here a stand-in) is put in front of the first system
message, and the request options (max_tokens, temperature, enable_thinking) pass through unchanged.
The key is read from PYTRIO_API_KEY / LLM_API_KEY or ~/.agent_harness/llm_key and is never printed.
    --live                                   LLM_BASE_URL / LLM_MODEL_ID (default: the Qwen3.8-27B endpoint)
    --live --live-base-url https://pytrio.com/api/openai/v1 --live-model <PyTRiO model id>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}
TASK = {"A": "classification", "B": "regression", "C": "regression"}
DOCKER: str | None = None
STAGE_EVERY = 0             # >0: the public projection changes every N rounds, as the platform's stages do


def stage_data(key: str, index: int) -> Path:
    """The `public/` directory a round sees. Stage s keeps the training set and drops the first s tenths of the
    validation rows (contiguous, so units stay whole apart from the first)."""
    package = ROOT / PACKAGES[key]
    stage = (index - 1) // STAGE_EVERY if STAGE_EVERY else 0
    if stage == 0:
        return package / "public"
    target = Path(tempfile.gettempdir()) / "smoke-stages" / f"{Path(PACKAGES[key]).name}-s{stage}" / "public"
    if not (target / "validation.jsonl").is_file():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / "public" / "train.jsonl", target / "train.jsonl")
        lines = [l for l in (package / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        keep = lines[int(len(lines) * min(stage, 5) / 10):]
        (target / "validation.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
    return target


def docker(image: str, command: str, mounts: dict[str, Path], env: dict[str, str]) -> list[str]:
    argv = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp",
            "--cpus", "2", "--memory", "4g"]
    for target, source in mounts.items():
        argv += ["-v", f"{Path(source).resolve().as_posix()}:{target}" + ("" if target == "/workspace" else ":ro")]
    for name, value in env.items():
        argv += ["-e", f"{name}={value}"]
    return argv + [image, command]


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(artifact_id: str, root: Path, checkpoint_paths, code_paths) -> dict:
    files = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(root.rglob("*")) if p.is_file()]
    return {"artifact_id": artifact_id, "files": files,
            "artifact": {"checkpoint_paths": list(checkpoint_paths), "code_paths": list(code_paths),
                         "predict_argv": ["python3", "/app/harness.py", "predict"]}}


def fake_model(messages: list, options: dict) -> dict:
    """Picks the choice with the largest gain_vs_parent, like a sensible decider would."""
    payload = json.loads(messages[-1]["content"])
    best = max(payload["choices"], key=lambda c: c.get("gain_vs_parent", 0.0))
    text = json.dumps({"select": best["artifact_id"], "reason": "收益最大（本地替身模型）"}, ensure_ascii=False)
    with LOCK:
        fake_model.calls += 1
        number = fake_model.calls
    return {"request_id": f"call-{number}", "choices": [{"message": {"role": "assistant", "content": text}}]}


fake_model.calls = 0

LIVE: dict | None = None
CALLS: list[dict] = []
LOCK = threading.Lock()
PROMPT = "官方固定任务提示（本地替身）"


def bind_task_prompt(messages: list, prompt: str) -> list:
    """What the host does: the official prompt goes in front of the first system message, once."""
    if messages and messages[0].get("role") == "system":
        return [{**messages[0], "content": prompt + "\n\n" + messages[0]["content"]}, *messages[1:]]
    return [{"role": "system", "content": prompt}, *messages]


def live_model(messages: list, options: dict, tag: str = "") -> dict:
    import urllib.error
    import urllib.request
    body = {"model": LIVE["model"], "messages": messages, "stream": False,
            **{k: options[k] for k in ("max_tokens", "temperature", "top_p", "enable_thinking", "seed") if k in options}}
    request = urllib.request.Request(LIVE["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Authorization": "Bearer " + LIVE["key"], "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        code = getattr(error, "code", None)
        with LOCK:
            CALLS.append({"tag": tag, "seconds": round(time.monotonic() - started, 1), "failed": f"HTTP {code}" if code else type(error).__name__,
                          "prompt_chars": sum(len(m.get("content") or "") for m in messages)})
        raise RuntimeError(f"endpoint failed: {code or type(error).__name__}") from None
    content = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    with LOCK:
        CALLS.append({"tag": tag, "seconds": round(time.monotonic() - started, 1), "usage": payload.get("usage"),
                  "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
                  "request_options": {k: v for k, v in body.items() if k not in ("messages", "model")},
                  "prompt_chars": sum(len(m.get("content") or "") for m in messages), "reply": content})
        payload["request_id"] = "live-" + str(payload.get("id") or len(CALLS))
    return payload


class Host:
    def __init__(self, start: dict, store: Path):
        self.start, self.store = start, store
        self.record = None
        self.artifacts: dict[str, Path] = {}
        self.state_dir: Path | None = None
        self.failure = None
        self.candidates = 0

    def serve(self, process):
        uploads: dict[str, dict] = {}
        pending: dict[str, dict] = {}
        process.stdin.write(json.dumps(self.start, ensure_ascii=False) + "\n")
        process.stdin.flush()
        sequence = 0
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("participant closed stdout")
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                raise RuntimeError("participant failed: " + str(frame.get("error")))
            if kind == "round_complete":
                self.record = frame["record"]
                process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                return
            if frame.get("id") != sequence:
                raise RuntimeError("rpc sequence mismatch")
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": self.start.get("_history", [])}
            elif kind == "model_request":
                if frame.get("max_tokens", 0) > self.start["max_output_tokens"]:
                    raise RuntimeError("max_tokens above limit")
                options = {k: v for k, v in frame.items() if k not in ("type", "id", "messages")}
                if LIVE:
                    response = live_model(bind_task_prompt(frame["messages"], self.start["prompt"]), options,
                                          f"{self.start['dataset_key']}-r{self.start['round_index']}")
                else:
                    response = fake_model(frame["messages"], options)
                reply = {"type": "model_response", "id": frame["id"], "call_id": response.pop("request_id"),
                         "response": response}
            elif kind == "upload_begin_request":
                upload_id = f"u{len(uploads)}"
                uploads[upload_id] = {"kind": frame["kind"], "files": {f["path"]: bytearray() for f in frame["files"]}}
                reply = {"type": "upload_begin_response", "id": frame["id"], "upload_id": upload_id}
            elif kind == "upload_chunk_request":
                data = uploads[frame["upload_id"]]["files"][frame["path"]]
                if frame["offset"] != len(data):
                    raise RuntimeError("upload offset mismatch")
                data.extend(base64.b64decode(frame["data_base64"], validate=True))
                reply = {"type": "upload_chunk_response", "id": frame["id"], "offset": len(data)}
            elif kind == "upload_finish_request":
                pending[frame["upload_id"]] = uploads[frame["upload_id"]]
                reply = {"type": "upload_finish_response", "id": frame["id"], "upload_id": frame["upload_id"]}
            elif kind == "state_request":
                upload = pending[frame["upload_id"]]
                self.state_dir = self.store / f"state-{self.start['round_index']}"
                shutil.rmtree(self.state_dir, ignore_errors=True)
                self.state_dir.mkdir(parents=True)
                for name, data in upload["files"].items():
                    (self.state_dir / name).parent.mkdir(parents=True, exist_ok=True)
                    (self.state_dir / name).write_bytes(bytes(data))
                reply = {"type": "state_response", "id": frame["id"], "state_sha256": "sha256:" + "11" * 32}
            elif kind == "candidate_request":
                if frame.get("parent_checkpoint_sha256") != self.start["parent_checkpoint_sha256"]:
                    raise RuntimeError("parent binding changed")
                self.candidates += 1
                artifact_id = f"art-r{self.start['round_index']}-{self.candidates}"
                folder = self.store / artifact_id
                for name, data in pending[frame["upload_id"]]["files"].items():
                    (folder / name).parent.mkdir(parents=True, exist_ok=True)
                    (folder / name).write_bytes(bytes(data))
                for name in frame["artifact"]["checkpoint_paths"] + frame["artifact"]["code_paths"]:
                    if not (folder / name).is_file():
                        raise RuntimeError(f"declared file missing: {name}")
                self.artifacts[artifact_id] = folder
                reply = {"type": "candidate_response", "id": frame["id"], "artifact_id": artifact_id,
                         "summary": {"metrics": {}}, "_artifact": frame["artifact"]}
                reply.pop("_artifact")
            else:
                raise RuntimeError("unexpected frame " + str(kind))
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()


def environment(sdk: Path, work: Path, previous: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(sdk), str(ROOT / "src"), str(ROOT / "submission")])
    env.update(CONTEST_WORK=str(work), CONTEST_PREVIOUS=str(previous), PYTHONIOENCODING="utf-8",
               AGENT_HARNESS_BASELINE=str(ROOT / "vendor" / "cpu-baseline"))
    return env


def run_round(sdk: Path, key: str, index: int, parent_id: str, parent_root: Path, state_dir: Path | None,
              store: Path, history: list) -> Host:
    package = ROOT / PACKAGES[key]          # an absolute override (--strip-meta) replaces ROOT
    public = stage_data(key, index)
    parent = receipt(parent_id, parent_root, ["checkpoint.npz"],
                     ["baselines/model.py"] if (parent_root / "baselines" / "model.py").is_file() else [])
    train_path, validation_path = public / "train.jsonl", public / "validation.jsonl"
    if DOCKER:
        train_path, validation_path = "/data/train.jsonl", "/data/validation.jsonl"
    start = {"type": "start", "protocol": "contest.harness.workspace.v2", "prompt": PROMPT,
             "prompt_version": "industrial-workspace-flexible-v3", "dataset_key": key, "round_index": index,
             "stage_id": f"stage-{(index - 1) // STAGE_EVERY if STAGE_EVERY else 0}", "run_id": "local-smoke", "model_name": "Qwen3.8-27B",
             "max_output_tokens": 2048, "wall_seconds": 900,
             "task_contract": {"task_type": TASK[key], "public_projection": {
                 "train_path": str(train_path), "validation_path": str(validation_path)}},
             "parent_checkpoint_sha256": sha(parent_root / "checkpoint.npz"), "parent_artifact": parent,
             "artifacts": [parent], "state_path": ("/state" if DOCKER else str(state_dir)) if state_dir else None,
             "resource_limits": {"artifact_upload_protocol": "chunked-v1"}, "_history": history}
    work = store / f"work-{index}"
    work.mkdir(parents=True)
    host = Host(start, store)
    if DOCKER:
        mounts = {"/data": public, "/previous": parent_root, "/workspace": work}
        if state_dir:
            mounts["/state"] = state_dir
        argv, env = docker(DOCKER, "train", mounts, {}), None
    else:
        argv, env = [sys.executable, str(sdk / "template" / "harness.py"), "train"], environment(sdk, work, parent_root)
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=env)
    errors: list[str] = []
    reader = threading.Thread(target=lambda: errors.extend(process.stderr))
    reader.start()
    try:
        host.serve(process)
    except BaseException:
        process.kill()          # the platform ends a round whose RPC failed; do not leave it waiting
        raise
    finally:
        process.wait(timeout=1200)
        reader.join()
    sys.stderr.write("".join(line for line in errors if line.startswith("[harness]")))
    if process.returncode != 0 or host.record is None:
        sys.stderr.write("".join(errors))
        raise RuntimeError(f"{key} round {index} failed with exit code {process.returncode}")
    return host


def serve_predictions(sdk: Path, key: str, artifact: Path) -> list[dict]:
    package = ROOT / PACKAGES[key]
    rows = [json.loads(line) for line in (package / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    env = environment(sdk, artifact, artifact)
    context = {"protocol": "contest.predict-context.v1", "dataset_key": key, "task": TASK[key],
               "artifact": {}, "artifact_id": artifact.name}
    env.update(CONTEST_ARTIFACT_PATH=str(artifact / "checkpoint.npz"), CONTEST_DATASET_KEY=key, CONTEST_TASK=TASK[key],
               CONTEST_PREDICTION_CONTEXT=json.dumps(context))
    feed = "".join(json.dumps({"task": TASK[key], "inputs": r["inputs"]}) + "\n" for r in rows)
    done = subprocess.run([sys.executable, str(sdk / "template" / "harness.py"), "predict"], input=feed,
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=1200)
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    outputs = [json.loads(line)["outputs"] for line in done.stdout.splitlines() if line.strip()]
    if len(outputs) != len(rows):
        raise RuntimeError(f"{len(outputs)} predictions for {len(rows)} rows")
    if TASK[key] == "classification":
        hits = [max(o, key=lambda item: item["score"])["label"] == r["answer"] for o, r in zip(outputs, rows)]
        return [{"accuracy": sum(hits) / len(hits)}]
    error = [(o["value"] - r["answer"]) ** 2 for o, r in zip(outputs, rows)]
    return [{"rmse": (sum(error) / len(error)) ** 0.5}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--keep", type=Path, help="keep artifacts here instead of a temporary directory")
    parser.add_argument("--docker", help="run every round inside this image")
    parser.add_argument("--live", action="store_true", help="forward model requests to a real endpoint")
    parser.add_argument("--live-base-url", help="OpenAI-compatible base URL (…/v1); default LLM_BASE_URL")
    parser.add_argument("--live-model", help="model id at that endpoint; default LLM_MODEL_ID")
    parser.add_argument("--strip-meta", action="store_true",
                        help="serve rows as the platform does: only id, inputs and answer (no meta)")
    parser.add_argument("--stage-every", type=int, default=0,
                        help="change the validation projection every N rounds (simulated platform stages)")
    parser.add_argument("--rounds", type=int, default=2, help="checkpoint rounds per dataset (the platform runs 10)")
    parser.add_argument("--parallel", action="store_true", help="run the datasets at the same time, as the platform does")
    parser.add_argument("--prompt-bytes", type=int, default=0,
                        help="pad the stand-in official prompt to about this many UTF-8 bytes (the platform's is up to 8000)")
    args = parser.parse_args()
    global DOCKER, LIVE, PROMPT, STAGE_EVERY
    DOCKER = args.docker
    STAGE_EVERY = args.stage_every
    if args.strip_meta:
        stripped = Path(tempfile.mkdtemp(prefix="stripped-"))
        for key in args.datasets:
            source = ROOT / PACKAGES[key]
            target = stripped / PACKAGES[key]
            (target / "public").mkdir(parents=True)
            shutil.copyfile(source / "checkpoint.npz", target / "checkpoint.npz")
            for name in ("train.jsonl", "validation.jsonl"):
                with (source / "public" / name).open(encoding="utf-8") as src, (target / "public" / name).open("w", encoding="utf-8") as dst:
                    for line in src:
                        if line.strip():
                            row = json.loads(line)
                            dst.write(json.dumps({k: row[k] for k in ("id", "inputs", "answer")}, ensure_ascii=False) + "\n")
            PACKAGES[key] = str(target)
    if args.prompt_bytes:
        filler = "本轮为检查点续训轮次，请依据任务合同、数据集说明与检查点信息作出选择。"
        text = PROMPT
        while len((text + filler).encode("utf-8")) <= min(args.prompt_bytes, 7900):
            text += filler
        PROMPT = text
    if args.live:
        sys.path.insert(0, str(ROOT / "src"))
        from agent_harness.agent import OpenAICompatibleProvider, _read_key
        default = OpenAICompatibleProvider(api_key="unused")
        key = os.environ.get("PYTRIO_API_KEY") or _read_key()
        if not key:
            raise SystemExit("no key: set PYTRIO_API_KEY / LLM_API_KEY or create ~/.agent_harness/llm_key")
        LIVE = {"base_url": args.live_base_url or default.base_url, "model": args.live_model or default.model, "key": key}
        print(json.dumps({"live_endpoint": LIVE["base_url"], "model": LIVE["model"]}), file=sys.stderr)
    def run_dataset(key: str) -> None:
        store = Path(tempfile.mkdtemp(prefix=f"smoke-{key}-")) if not args.keep else args.keep / key
        store.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        parent_id, parent_root, state, history = "official-baseline", ROOT / PACKAGES[key], None, []
        try:
            for index in range(1, args.rounds + 1):
                round_started = time.monotonic()
                try:
                    host = run_round(args.sdk, key, index, parent_id, parent_root, state, store, history)
                except Exception as error:
                    print(json.dumps({"dataset": key, "round": index, "failed": str(error)[:300],
                                      "seconds": round(time.monotonic() - round_started, 1)}, ensure_ascii=False), flush=True)
                    return
                record = host.record
                print(json.dumps({"dataset": key, "round": index, "seconds": round(time.monotonic() - round_started, 1),
                                  "selected": record["selected_artifact_id"], "reason": record["reason"][:160],
                                  "calls": record["model_call_ids"],
                                  "config": {k: record["configuration"].get(k) for k in ("rhythm", "selected_arm", "paired_vs_parent", "trained")}},
                                 ensure_ascii=False), flush=True)
                history.append({"id": f"rec-{index}", "selected_artifact_id": record["selected_artifact_id"]})
                if record["selected_artifact_id"] in host.artifacts:
                    parent_id, parent_root = record["selected_artifact_id"], host.artifacts[record["selected_artifact_id"]]
                state = host.state_dir
            print(json.dumps({"dataset": key, "served": parent_id, "prediction_check": serve_predictions(args.sdk, key, parent_root),
                              "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
        finally:
            if not args.keep:
                shutil.rmtree(store, ignore_errors=True)

    if args.parallel:
        threads = [threading.Thread(target=run_dataset, args=(key,)) for key in args.datasets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for key in args.datasets:
            run_dataset(key)
    for number, call in enumerate(CALLS, 1):
        print(json.dumps({"live_call": number, **call}, ensure_ascii=False))


if __name__ == "__main__":
    main()
