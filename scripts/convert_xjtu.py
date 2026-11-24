"""Convert XJTU-SY bearing vibration data (Mendeley .mat mirror, CC BY 4.0) into the official B JSONL.

Raw input:  data/raw/xjtu-sy/mendeley/bearingN.mat   (N = 1..15; rawnet = 32768 points × 2 channels × snapshots)
Mapping:    N 1–5 → 35Hz12kN/Bearing1_N, 6–10 → 37.5Hz11kN/Bearing2_(N−5), 11–15 → 40Hz10kN/Bearing3_(N−10)

The 17 features per channel reproduce the official converter (verified against the official package to
floating-point precision): time-domain statistics on the raw signal; spectral features on the one-sided
FFT of the mean-removed signal, power = (|FFT| / n)², no window, DC bin kept.

Bearings that are in the official package are only used to VERIFY the conversion. Bearings that are not
(very likely the hidden test stream) go to data/sealed/B and must not be used for training or tuning.

    python scripts/convert_xjtu.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matv5 import loadmat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xjtu-sy" / "mendeley"
PACKAGE = ROOT / "cpu-b-rul-20261110-v1" / "public"
OUT = ROOT / "data" / "sealed" / "B"
FS = 25600.0
CONDITIONS = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
FEATURES = ("rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness", "crest_factor", "shape_factor",
            "impulse_factor", "margin_factor", "spectral_centroid", "spectral_rms", "spectral_kurtosis",
            "dominant_freq", "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")


def bearing_name(index: int) -> tuple[str, str]:
    condition, position = (index - 1) // 5 + 1, (index - 1) % 5 + 1
    return CONDITIONS[condition], f"Bearing{condition}_{position}"


def channel_features(x: np.ndarray) -> dict:
    n = len(x)
    mean, std = x.mean(), x.std()
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    spectrum = np.abs(np.fft.rfft(x - mean)) / n
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    power = spectrum ** 2
    top = min(12800.0, FS / 2)
    return {
        "rms": rms, "peak": peak, "peak_to_peak": float(x.max() - x.min()), "mean_abs": mean_abs,
        "kurtosis": float(np.mean((x - mean) ** 4) / std ** 4), "skewness": float(np.mean((x - mean) ** 3) / std ** 3),
        "crest_factor": peak / rms, "shape_factor": rms / mean_abs, "impulse_factor": peak / mean_abs,
        "margin_factor": peak / float(np.mean(np.sqrt(np.abs(x)))) ** 2,
        "spectral_centroid": float(np.sum(freqs * power) / np.sum(power)),
        "spectral_rms": float(np.sqrt(np.mean(power))),
        "spectral_kurtosis": float(np.mean((power - power.mean()) ** 4) / power.std() ** 4),
        "dominant_freq": float(freqs[np.argmax(spectrum)]),
        "band1_energy_0_2k": float(power[(freqs >= 0) & (freqs < 2000)].sum()),
        "band2_energy_2k_6k": float(power[(freqs >= 2000) & (freqs < 6000)].sum()),
        "band3_energy_6k_12k": float(power[(freqs >= 6000) & (freqs <= top)].sum()),
    }


def convert(index: int) -> tuple[str, list[dict]]:
    condition, name = bearing_name(index)
    rawnet = loadmat(str(RAW / f"bearing{index}.mat"))["rawnet"]
    count = rawnet.shape[2]
    rows = []
    for i in range(count):
        snapshot = i + 1
        rows.append({"id": f"xjtu_{name}_s{snapshot:06d}",
                     "inputs": {"features": {"horizontal": channel_features(rawnet[:, 0, i]),
                                             "vertical": channel_features(rawnet[:, 1, i])}},
                     "answer": count - snapshot,
                     "meta": {"bearing": name, "snapshot": snapshot,
                              "source": f"XJTU-SY_Bearing_Datasets/{condition}/{name}/{snapshot}.csv"}})
    return name, rows


def line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def official_rows() -> dict[str, tuple[dict, str]]:
    rows = {}
    for split in ("train", "validation"):
        for text in (PACKAGE / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(text)
            rows[row["id"]] = (row, text)
    return rows


def compare(ours: list[dict], official: dict) -> dict:
    exact = close = worst = 0.0
    for row in ours:
        ref, text = official[row["id"]]
        exact += line(row) == text
        diffs = [abs(row["inputs"]["features"][ch][k] - ref["inputs"]["features"][ch][k]) /
                 max(abs(ref["inputs"]["features"][ch][k]), 1e-12)
                 for ch in ("horizontal", "vertical") for k in FEATURES]
        worst = max(worst, max(diffs))
        close += max(diffs) < 1e-9 and row["answer"] == ref["answer"] and row["meta"] == ref["meta"]
    return {"rows": len(ours), "byte_identical": int(exact), "within_1e-9": int(close), "max_relative_diff": worst}


README = """# data/sealed/B — 轴承隐藏测试流的候选数据

由 `scripts/convert_xjtu.py` 从 XJTU-SY 原始振动数据（Mendeley 镜像，CC BY 4.0，原始数据见 data/raw/xjtu-sy/mendeley）
转换为官方 B 的 JSONL 格式。转换方法先在官方数据包已有的轴承上逐条核对（结果见 MANIFEST.json），再用于这里的轴承。

这些轴承**不在官方公开数据包里**，很可能就是隐藏测试流。使用规矩与 data/sealed 其他数据相同：
- 不用于训练要提交的模型，不用于调整任何参数；
- 只用于系统冻结后的一次性评估；
- 不写入经验库，不打包进提交镜像。
"""


def main() -> None:
    official = official_rows()
    public_bearings = {row["meta"]["bearing"] for row, _ in official.values()}
    manifest = {"source": "Mendeley mpn45f4gxc (XJTU-SY Bearing Datasets, CC BY 4.0)", "verified": {}, "sealed": {}}
    expected = {f["filename"]: (f.get("content_details") or {}).get("sha256_hash")
                for f in json.loads((RAW / "files.json").read_text(encoding="utf-8"))}
    for index in range(1, 16):
        path = RAW / f"bearing{index}.mat"
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(path.name):
            print(f"skip {path.name}: incomplete or corrupted (sha256 differs from Mendeley's)", flush=True)
            continue
        name, rows = convert(index)
        if name in public_bearings:
            manifest["verified"][name] = compare(rows, official)
            print(f"verify {name}: {manifest['verified'][name]}", flush=True)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            raw = ("\n".join(line(r) for r in rows) + "\n").encode("utf-8")
            target = OUT / f"{name}.jsonl"
            target.write_bytes(raw)
            manifest["sealed"][name] = {"file": target.relative_to(ROOT).as_posix(), "rows": len(rows),
                                        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
            print(f"sealed {name}: {len(rows)} snapshots -> {target.relative_to(ROOT).as_posix()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bad = [k for k, v in manifest["verified"].items() if v["within_1e-9"] != v["rows"]]
    if bad:
        raise SystemExit(f"conversion does not reproduce the official features for {bad}")


if __name__ == "__main__":
    main()
