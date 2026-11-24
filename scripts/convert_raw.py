"""Convert data/raw into the official JSONL format, into data/sealed/.

The parts that the official packages already contain are regenerated too and compared byte for
byte with the official files; that comparison is the proof that the conversion matches the
platform's own converter. Everything else is very likely to overlap the hidden test stream, so it
is written to data/sealed/ and must not be used for training or for tuning (设计方案 §5.4).

    python scripts/convert_raw.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sealed"
GAS_CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write(path: Path, rows: list[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(rows) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return {"file": path.relative_to(ROOT).as_posix(), "rows": len(rows),
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def same_bytes(rows: list[str], official: Path) -> bool:
    return ("\n".join(rows) + "\n").encode("utf-8") == official.read_bytes()


# ---------------------------------------------------------------------------------------- A

def gas_rows(batch: int) -> list[str]:
    rows = []
    path = RAW / "gas" / "Dataset" / f"batch{batch}.dat"
    for index, text in enumerate((t for t in path.read_text(encoding="utf-8").splitlines() if t.strip()), 1):
        parts = text.split()
        label = int(parts[0])
        values: dict[int, float] = {}
        for token in parts[1:]:
            key, value = token.split(":")
            if int(key) in values:
                raise ValueError(f"{path}:{index}: feature {key} repeated")
            values[int(key)] = float(value)
        if sorted(values) != list(range(1, 129)):
            raise ValueError(f"{path}:{index}: expected features 1..128")
        rows.append(line({"id": f"gas_b{batch:02d}_{index:06d}",
                          "inputs": {"features": [values[i] for i in range(1, 129)]},
                          "answer": GAS_CLASSES[label - 1],
                          "meta": {"source_batch": batch, "source_index": index}}))
    return rows


def convert_gas() -> dict:
    package = ROOT / "cpu-a-guide-20261110-v1" / "public"
    report = {"files": [], "verified_against_official": {}}
    for batch in range(1, 11):
        rows = gas_rows(batch)
        if batch == 1:
            report["verified_against_official"]["batch1 == public/train.jsonl"] = same_bytes(rows, package / "train.jsonl")
            continue
        if batch == 2:
            report["verified_against_official"]["batch2 == public/validation.jsonl"] = same_bytes(rows, package / "validation.jsonl")
            continue
        report["files"].append(write(OUT / "A" / f"batch{batch:02d}.jsonl", rows))
    return report


# ---------------------------------------------------------------------------------------- C

def cmapss_table(path: Path) -> dict[int, list[list[str]]]:
    units: dict[int, list[list[str]]] = {}
    for text in path.read_text(encoding="utf-8").splitlines():
        parts = text.split()
        if parts:
            if len(parts) != 26:
                raise ValueError(f"{path}: expected 26 columns, got {len(parts)}")
            units.setdefault(int(parts[0]), []).append(parts)
    return units


def cmapss_row(subset: str, split: str, unit: int, parts: list[str], answer: int, extra: dict | None = None) -> str:
    cycle = int(parts[1])
    meta = {"source_subset": subset, "source_split": split, "unit": unit, "cycle": cycle, "window_size": 1}
    return line({"id": f"{subset.lower()}_{split}_u{unit:04d}_c{cycle:05d}",
                 "inputs": {"settings": [float(v) for v in parts[2:5]], "sensors": [float(v) for v in parts[5:26]]},
                 "answer": answer, "meta": {**meta, **(extra or {})}})


def official_units(path: Path) -> dict[str, set[int]]:
    units: dict[str, set[int]] = {}
    for text in path.read_text(encoding="utf-8").splitlines():
        meta = json.loads(text)["meta"]
        units.setdefault(meta["source_subset"], set()).add(meta["unit"])
    return units


def convert_cmapss() -> dict:
    package = ROOT / "cpu-c-guide-20261110-v1" / "public"
    in_train, in_validation = official_units(package / "train.jsonl"), official_units(package / "validation.jsonl")
    train_rows, validation_rows, holdout, last, every = [], [], [], [], []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for unit, cycles in sorted(cmapss_table(RAW / "cmapss" / f"train_{subset}.txt").items()):
            final = int(cycles[-1][1])
            rows = [cmapss_row(subset, "train", unit, p, final - int(p[1])) for p in cycles]   # RUL = T − t
            if unit in in_train.get(subset, set()):
                train_rows += rows
            elif unit in in_validation.get(subset, set()):
                validation_rows += rows
            else:
                holdout += rows
        remaining = [int(v) for v in (RAW / "cmapss" / f"RUL_{subset}.txt").read_text().split()]
        test = cmapss_table(RAW / "cmapss" / f"test_{subset}.txt")
        if len(remaining) != len(test):
            raise ValueError(f"{subset}: {len(test)} test engines but {len(remaining)} RUL values")
        for (unit, cycles), r in zip(sorted(test.items()), remaining):
            observed = int(cycles[-1][1])
            # Official benchmark: only the last observed cycle, labelled with r (never 0).
            last.append(cmapss_row(subset, "test", unit, cycles[-1], r))
            # Extension the package explicitly does not ship: earlier cycles get r + T_observed − t.
            every += [cmapss_row(subset, "test", unit, p, r + observed - int(p[1]),
                                 {"target_rule": "r + T_observed - t (extension, not the official benchmark)"})
                      for p in cycles]
    return {"verified_against_official": {
                "train units == public/train.jsonl": same_bytes(train_rows, package / "train.jsonl"),
                "validation units == public/validation.jsonl": same_bytes(validation_rows, package / "validation.jsonl")},
            "files": [write(OUT / "C" / "train_holdout_units.jsonl", holdout),
                      write(OUT / "C" / "test_last_cycle.jsonl", last),
                      write(OUT / "C" / "test_all_cycles_extended.jsonl", every)]}


README = """# data/sealed — 很可能与隐藏测试流重合的数据

由 `scripts/convert_raw.py` 从 `data/raw/` 转换而来，格式与官方 JSONL 完全一致（官方已给出的部分逐字节核对过，见 MANIFEST.json）。

**使用规矩（设计方案 §5.4）**
- 不用于训练要提交的模型，不用于调整要提交系统的任何参数。
- 只用于冻结系统之后的一次性最终测试；测试前记录代码与配置指纹，测试后不再根据结果调参。
- harness 的任务配置把 `data/sealed` 列为泄漏标记，任何写进经验笔记的内容一旦提到它就会被拒绝。
- 打包提交的镜像不包含本目录。

| 文件 | 内容 |
|---|---|
| A/batch03.jsonl … batch10.jsonl | 气体 batch3–10，官方没有给出的部分 |
| C/train_holdout_units.jsonl | 原始训练文件中官方没有放进训练集和验证集的发动机（开发保留集） |
| C/test_last_cycle.jsonl | 官方测试文件，每台发动机只取最后观测周期，答案为 RUL_FD00x 中的 r（官方基准口径） |
| C/test_all_cycles_extended.jsonl | 官方测试文件的全部周期，答案按 r + T_observed − t 推算（官方包明确没有这样扩展，仅供参考） |

B（XJTU-SY 轴承）原始数据尚未下载，暂无转换结果。
"""


def main() -> None:
    manifest = {"A": convert_gas(), "C": convert_cmapss(),
                "B": "data/raw/xjtu-sy is empty; nothing converted"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    checks = [v for part in ("A", "C") for v in manifest[part]["verified_against_official"].values()]
    if not all(checks):
        raise SystemExit("conversion does not match the official files byte for byte")


if __name__ == "__main__":
    main()
