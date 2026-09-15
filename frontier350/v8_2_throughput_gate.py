from __future__ import annotations

"""Read-only A/B throughput gate for Scale V8.2 on the current continuity checkpoint.

The gate never promotes a benchmark checkpoint and never writes the controller state. Each
trial starts from the same current continuity checkpoint, writes only to disposable scratch,
and the checkpoint/state identities are verified again after all trials.

Gate A is intentionally short: it decides whether Fast2 + coalesced input is worth a longer
5M-token thermal soak. Stable 50k here means median >=50k tok/s, p10 >=45k tok/s and p10
>=90% of the candidate median after warmup. A candidate must also preserve the plain-Fast2
loss trajectory within 2e-4. Gate A alone is NOT a multi-hour thermal-stability claim.
"""

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from apply_v6_fast2_patch import patch_trainer as patch_fast2
from recovery_identity import file_sha
from v7_1_runtime_hotfix import patch_trainer as patch_runtime
from v8_2_profile_contract import profile_command_args

SEQ = 512
BATCH = 4
MICROBATCHES = 12
MAIN_PACK = 9
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES
MAX_RESERVED_BYTES = int(11.0 * 1024**3)
MAX_ALLOCATED_BYTES = int(10.25 * 1024**3)
MAX_LOSS_DIFF = 2e-4


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def gpu_sample() -> dict | None:
    try:
        raw = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
        vals = [x.strip() for x in raw.split(",")]
        return {
            "temp_c": float(vals[0]),
            "vram_used_mib": float(vals[1]),
            "vram_free_mib": float(vals[2]),
            "util_pct": float(vals[3]),
            "power_w": float(vals[4]),
            "t": time.time(),
        }
    except Exception:
        return None


def cool_to(limit_c: float = 65.0, timeout_s: float = 180.0) -> dict | None:
    start = time.monotonic()
    last = gpu_sample()
    while last is not None and last["temp_c"] > limit_c and time.monotonic() - start < timeout_s:
        time.sleep(5)
        last = gpu_sample()
    if last is not None and last["temp_c"] > limit_c + 3:
        raise RuntimeError(f"GPU did not cool sufficiently: {last['temp_c']} C")
    return last


def run_process_with_gpu_sampling(cmd: list[str], log: Path) -> tuple[int, float, list[dict]]:
    samples: list[dict] = []
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            sample = gpu_sample()
            if sample is not None:
                samples.append(sample)
            time.sleep(5)
        code = int(proc.returncode)
    sample = gpu_sample()
    if sample is not None:
        samples.append(sample)
    return code, time.monotonic() - started, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--updates", type=int, default=24)
    ap.add_argument("--warmup-updates", type=int, default=6)
    ap.add_argument("--target-tps", type=float, default=50_000.0)
    args = ap.parse_args()
    if not 16 <= args.updates <= 60:
        raise ValueError("updates must be 16..60")
    if not 3 <= args.warmup_updates < args.updates - 5:
        raise ValueError("invalid warmup window")

    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    scratch = Path(args.scratch).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report must be empty")
    if scratch.exists():
        raise FileExistsError("scratch already exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    state_path = assets / "frontier350_scale_v5_state.json"
    state = load_json(state_path)
    state_sha_before = sha256_file(state_path)
    continuity = state["continuity"]
    anchor = Path(continuity["path"])
    anchor_sha = str(continuity["sha256"])
    anchor_tokens = int(continuity["tokens"])
    if not anchor.is_file() or file_sha(anchor) != anchor_sha:
        raise RuntimeError("current continuity checkpoint identity mismatch")
    anchor_sha_before = file_sha(anchor)

    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/train"
    replay = assets / "frontier350_replay_v9_20260912"
    if not main_data.is_dir() or not replay.is_dir():
        raise FileNotFoundError("training corpora missing")

    trainer_source = Path(__file__).resolve().parent / "train_guarded_v9.py"
    fast2_trainer = Path(__file__).resolve().parent / "_v82_gate_fast2_train_guarded_v9.py"
    source_text = trainer_source.read_text(encoding="utf-8")
    fast2_trainer.write_text(patch_runtime(patch_fast2(source_text)), encoding="utf-8")

    budget = TOKENS_PER_UPDATE * args.updates
    base = [
        "--anchor", str(anchor), "--anchor-sha", anchor_sha,
        "--main-data", str(main_data), "--replay", str(replay),
        "--mode", "replay_only", "--new-tokens", str(budget),
        "--max-updates", str(args.updates), "--batch-size", str(BATCH),
        "--microbatches", str(MICROBATCHES), "--main-pack", str(MAIN_PACK),
        "--compile-mode", "default", "--lr", "2.6e-5", "--lr-end", "1.0e-5",
        "--replay-boost", "1.50", "--fast-replay-only",
    ]

    variants = [
        ("fast2_plain", "fast2", "off"),
        ("fast2c_pageable", "fast2c", "coalesced-pageable"),
        ("fast2c_pin", "fast2c", "coalesced-pin"),
    ]
    rows: list[dict] = []

    try:
        for name, profile, input_profile in variants:
            out = scratch / name
            trial_base = [*base, "--out", str(out)]
            cli, selected_input = profile_command_args(trial_base, profile, None, "off", input_profile)
            before_gpu = cool_to()
            log = report / f"{name}.log"
            code, wall, gpu = run_process_with_gpu_sampling(
                [sys.executable, "-u", str(fast2_trainer), *cli], log
            )
            row: dict = {
                "name": name, "profile": profile, "input_profile": selected_input,
                "returncode": code, "wall_seconds": wall,
                "gpu_before": before_gpu, "gpu_samples": gpu,
            }
            if code != 0:
                row["error_tail"] = log.read_text(encoding="utf-8", errors="replace")[-6000:]
                rows.append(row)
                print(json.dumps({"event": "v8_2_gate_trial", **row}), flush=True)
                shutil.rmtree(out, ignore_errors=True)
                continue

            metrics_path = out / "metrics.jsonl"
            metrics = [json.loads(x) for x in metrics_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            train = [x for x in metrics if x.get("event") == "train"]
            if len(train) != args.updates:
                raise RuntimeError(f"{name}: expected {args.updates} train rows, got {len(train)}")
            stable_rows = train[args.warmup_updates:]
            rates = [float(x["tokens_per_second"]) for x in stable_rows]
            losses = [float(x["loss"]) for x in train]
            med = statistics.median(rates)
            low = p10(rates)
            mean = statistics.fmean(rates)
            peak = max(rates)
            peak_alloc = max(int(x.get("peak_allocated_bytes", 0)) for x in train)
            peak_reserved = max(int(x.get("peak_reserved_bytes", 0)) for x in train)
            max_temp = max((float(x["temp_c"]) for x in gpu), default=float("nan"))
            stable = low >= 0.90 * med
            memory_ok = peak_alloc <= MAX_ALLOCATED_BYTES and peak_reserved <= MAX_RESERVED_BYTES
            target_reached = (
                memory_ok and stable and med >= args.target_tps and low >= 0.90 * args.target_tps
                and (not math.isfinite(max_temp) or max_temp < 83.0)
            )
            summary = load_json(out / "summary.json")
            row.update({
                "median_tps": med, "mean_tps": mean, "p10_tps": low, "peak_tps": peak,
                "stable_samples": len(rates), "stable": stable, "target_reached": target_reached,
                "peak_allocated_gib": peak_alloc / 1024**3,
                "peak_reserved_gib": peak_reserved / 1024**3,
                "memory_ok": memory_ok, "max_temp_c": max_temp,
                "summary_tokens": summary.get("new_tokens"),
                "tokens_per_update": summary.get("tokens_per_update"),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "losses": losses,
            })
            shutil.copy2(metrics_path, report / f"{name}_metrics.jsonl")
            shutil.copy2(out / "summary.json", report / f"{name}_summary.json")
            rows.append(row)
            print(json.dumps({"event": "v8_2_gate_trial", **row}), flush=True)
            shutil.rmtree(out, ignore_errors=True)
    finally:
        fast2_trainer.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)

    anchor_sha_after = file_sha(anchor)
    state_sha_after = sha256_file(state_path)
    read_only_ok = anchor_sha_after == anchor_sha_before and state_sha_after == state_sha_before
    if not read_only_ok:
        raise RuntimeError("benchmark mutated continuity checkpoint or controller state")

    valid = [r for r in rows if r.get("returncode") == 0 and "median_tps" in r]
    if not valid:
        raise RuntimeError("no V8.2 throughput trial completed")
    plain = next((r for r in valid if r["name"] == "fast2_plain"), None)
    if plain is None:
        raise RuntimeError("plain Fast2 reference failed; semantic A/B is invalid")
    plain_losses = list(plain["losses"])
    for row in valid:
        losses = list(row["losses"])
        if len(losses) != len(plain_losses):
            row["semantic_max_loss_diff"] = float("inf")
            row["semantic_ok"] = False
        else:
            diff = max(abs(a - b) for a, b in zip(losses, plain_losses)) if losses else 0.0
            row["semantic_max_loss_diff"] = diff
            row["semantic_ok"] = diff <= MAX_LOSS_DIFF
    eligible = [r for r in valid if r.get("semantic_ok")]
    if not eligible:
        raise RuntimeError("no numerically equivalent V8.2 candidate survived")
    best = max(eligible, key=lambda r: float(r["median_tps"]))
    speedup = float(best["median_tps"]) / float(plain["median_tps"])
    gate_pass = bool(best.get("target_reached") and best.get("semantic_ok"))

    result = {
        "schema": 2,
        "benchmark": "neral350-v8.2-throughput-gate-a",
        "anchor_path": str(anchor), "anchor_sha256": anchor_sha, "anchor_tokens": anchor_tokens,
        "target_tps": args.target_tps,
        "stable_definition": "post-warmup median >= target; p10 >=90% target and >=90% median; memory within limits; no >=83C sample; max loss-trajectory delta vs plain Fast2 <=2e-4",
        "max_loss_diff": MAX_LOSS_DIFF,
        "updates_per_trial": args.updates,
        "warmup_updates_excluded": args.warmup_updates,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "read_only_identity_verified": read_only_ok,
        "plain_fast2": plain,
        "best": best,
        "best_speedup_vs_plain_fast2": speedup,
        "gate_a_pass": gate_pass,
        "promotion": False,
        "next_gate": "Gate-A winner must pass a disposable 5M-token zero-throttle thermal soak before production adoption.",
        "results": rows,
    }
    (report / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "v8_2_gate_selection", **result}), flush=True)
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
