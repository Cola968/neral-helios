from __future__ import annotations

"""Bounded Pack-6 vs Pack-9 benchmark for Neral 350M.

This script never promotes a checkpoint. It reuses the validated guarded trainer and
changes only `main_pack`, so data, LR, replay share, optimizer, trainable blocks and token
budget stay fixed. The result is evidence for execution geometry only.
"""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from recovery_identity import file_sha

SEQ = 512
BATCH = 4
MICROS = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROS
MAX_ALLOCATED_GIB = 10.25
MAX_RESERVED_GIB = 11.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def discover_anchor(assets: Path) -> tuple[Path, str, int]:
    state = assets / "frontier350_scale_v3_state.json"
    if state.is_file():
        obj = read_json(state)
        p = Path(obj["safe_anchor_path"])
        if p.is_file() and file_sha(p) == obj["safe_anchor_sha256"]:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            tokens = int(ck["processed_tokens"])
            if tokens == int(obj["safe_anchor_tokens"]):
                return p, obj["safe_anchor_sha256"], tokens

    rows = []
    for p in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        if not p.is_file():
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            rows.append((int(ck["processed_tokens"]), p, file_sha(p)))
        except Exception:
            pass
    if not rows:
        raise RuntimeError("no promoted Neral 350M checkpoint found")
    tokens, path, sha = max(rows, key=lambda x: x[0])
    return path, sha, tokens


def run_trial(trainer: Path, anchor: Path, anchor_sha: str, main: Path, replay: Path,
              scratch: Path, name: str, pack: int, updates: int, warmup: int) -> dict:
    out = scratch / name
    budget = TOKENS_PER_UPDATE * updates
    cmd = [
        sys.executable, "-u", str(trainer),
        "--anchor", str(anchor), "--anchor-sha", anchor_sha,
        "--main-data", str(main), "--replay", str(replay),
        "--out", str(out), "--mode", "replay_only",
        "--new-tokens", str(budget), "--max-updates", str(updates),
        "--batch-size", str(BATCH), "--microbatches", str(MICROS),
        "--main-pack", str(pack), "--compile-mode", "default",
        "--fast-replay-only",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    row = {"name": name, "pack": pack, "returncode": proc.returncode, "wall_seconds": wall}
    if proc.returncode != 0:
        row["error_tail"] = proc.stdout[-5000:]
        return row

    metrics = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()]
    train = [x for x in metrics if x.get("event") == "train"]
    if len(train) != updates:
        raise RuntimeError(f"{name}: expected {updates} train rows, got {len(train)}")
    stable = train[warmup:]
    rates = [float(x["tokens_per_second"]) for x in stable]
    summary = read_json(out / "summary.json")
    if int(summary["tokens_per_update"]) != TOKENS_PER_UPDATE or int(summary["new_tokens"]) != budget:
        raise RuntimeError("token-accounting contract changed")
    if not summary.get("frozen_weights_unchanged"):
        raise RuntimeError("frozen-weight invariant failed")

    peak_alloc = max(int(x["peak_allocated_bytes"]) for x in train) / 1024**3
    peak_reserved = max(int(x["peak_reserved_bytes"]) for x in train) / 1024**3
    median = statistics.median(rates)
    row.update(
        median_tps=median,
        mean_tps=statistics.fmean(rates),
        p10_tps=p10(rates),
        peak_tps=max(rates),
        peak_allocated_gib=peak_alloc,
        peak_reserved_gib=peak_reserved,
        stable=(p10(rates) >= 0.90 * median),
        memory_ok=(peak_alloc <= MAX_ALLOCATED_GIB and peak_reserved <= MAX_RESERVED_GIB),
        checkpoint_sha256=summary["checkpoint_sha256"],
        frozen_weights_unchanged=True,
    )
    shutil.rmtree(out, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--updates", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    if not 10 <= a.updates <= 30 or not 2 <= a.warmup < a.updates - 2:
        raise ValueError("bounded benchmark limits violated")

    assets, report, scratch = map(lambda p: Path(p).resolve(), (a.assets, a.report, a.scratch))
    report.mkdir(parents=True, exist_ok=True)
    if scratch.exists():
        raise FileExistsError(scratch)
    scratch.mkdir(parents=True)

    anchor, anchor_sha, anchor_tokens = discover_anchor(assets)
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/train"
    replay = assets / "frontier350_replay_v9_20260912"
    trainer = Path(__file__).parent / "train_guarded_v9.py"

    rows = []
    for name, pack in (("scalev3_pack6", 6), ("candidate_pack9", 9)):
        row = run_trial(trainer, anchor, anchor_sha, main_data, replay, scratch, name, pack, a.updates, a.warmup)
        rows.append(row)
        print(json.dumps({"event": "trial", **row}), flush=True)

    valid = [x for x in rows if x.get("returncode") == 0 and x.get("stable") and x.get("memory_ok")]
    baseline = rows[0]
    candidate = rows[1]
    speedup = None
    if "median_tps" in baseline and "median_tps" in candidate:
        speedup = candidate["median_tps"] / baseline["median_tps"]

    result = {
        "schema": 1,
        "benchmark": "neral350-pack9-v2",
        "anchor_path": str(anchor),
        "anchor_sha256": anchor_sha,
        "anchor_tokens": anchor_tokens,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "updates": a.updates,
        "warmup_excluded": a.warmup,
        "results": rows,
        "speedup_pack9_vs_pack6": speedup,
        "pack9_execution_candidate": bool(candidate in valid and speedup is not None and speedup > 1.0),
        "promotion_allowed": False,
        "next_gate": "Any winner must pass an own-weight canary with unchanged validation/dev/bilingual promotion gates before Scale V3 adoption.",
    }
    (report / "pack9_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
