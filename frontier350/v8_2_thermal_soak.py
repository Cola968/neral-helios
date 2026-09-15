from __future__ import annotations

"""Gate B: disposable 5M-token thermal soak for the Gate-A V8.2 winner.

This never promotes its output.  It starts from the exact Gate-A anchor, verifies controller
state/checkpoint identity before and after, and requires zero thermal-throttle events.  The
5M budget matches a normal Scale V5 micro-stage so a short benchmark spike cannot be called
"stable 50k".
"""

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

from apply_v6_fast2_patch import patch_trainer as patch_fast2
from recovery_identity import file_sha
from v7_1_runtime_hotfix import patch_trainer as patch_runtime
from v8_2_profile_contract import profile_command_args
from v8_2_throughput_gate import (
    BATCH, MAIN_PACK, MAX_ALLOCATED_BYTES, MAX_RESERVED_BYTES, MICROBATCHES,
    TOKENS_PER_UPDATE, cool_to, load_json, p10, run_process_with_gpu_sampling, sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--target-tps", type=float, default=50_000.0)
    ap.add_argument("--tokens", type=int, default=5_000_000)
    args = ap.parse_args()
    if args.tokens < 2_000_000 or args.tokens > 10_000_000:
        raise ValueError("thermal soak must be 2M..10M tokens")

    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    scratch = Path(args.scratch).resolve()
    selection = load_json(Path(args.selection).resolve())
    if not selection.get("gate_a_pass"):
        raise RuntimeError("Gate A did not pass; thermal soak is forbidden")
    best = selection["best"]
    profile = str(best["profile"])
    input_profile = str(best.get("input_profile", "off"))
    if profile not in ("fast2", "fast2c"):
        raise RuntimeError(f"unsupported Gate-A winner: {profile}")
    if scratch.exists():
        raise FileExistsError("thermal-soak scratch already exists")
    scratch.mkdir(parents=True)

    state_path = assets / "frontier350_scale_v5_state.json"
    state_sha_before = sha256_file(state_path)
    anchor = Path(selection["anchor_path"])
    anchor_sha = str(selection["anchor_sha256"])
    anchor_tokens = int(selection["anchor_tokens"])
    if not anchor.is_file() or file_sha(anchor) != anchor_sha:
        raise RuntimeError("Gate-A anchor changed before thermal soak")
    anchor_sha_before = file_sha(anchor)

    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/train"
    replay = assets / "frontier350_replay_v9_20260912"
    trainer_source = Path(__file__).resolve().parent / "train_guarded_v9.py"
    fast2_trainer = Path(__file__).resolve().parent / "_v82_soak_fast2_train_guarded_v9.py"
    fast2_trainer.write_text(
        patch_runtime(patch_fast2(trainer_source.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )

    out = scratch / "winner_5m"
    base = [
        "--anchor", str(anchor), "--anchor-sha", anchor_sha,
        "--main-data", str(main_data), "--replay", str(replay),
        "--out", str(out), "--mode", "replay_only",
        "--new-tokens", str(args.tokens), "--batch-size", str(BATCH),
        "--microbatches", str(MICROBATCHES), "--main-pack", str(MAIN_PACK),
        "--compile-mode", "default", "--lr", "2.6e-5", "--lr-end", "1.0e-5",
        "--replay-boost", "1.50", "--fast-replay-only",
    ]
    cli, selected_input = profile_command_args(base, profile, None, "off", input_profile)
    log = report / "thermal_soak.log"
    cool_to()
    try:
        code, wall, gpu = run_process_with_gpu_sampling(
            [sys.executable, "-u", str(fast2_trainer), *cli], log
        )
        log_text = log.read_text(encoding="utf-8", errors="replace")
        throttle_events = log_text.count('"event": "v8_thermal_throttle"') + log_text.count('"event":"v8_thermal_throttle"')
        if code != 0:
            result = {
                "schema": 1, "benchmark": "neral350-v8.2-thermal-soak-b",
                "gate_b_pass": False, "returncode": code,
                "error_tail": log_text[-8000:], "promotion": False,
            }
            (report / "thermal_soak.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise SystemExit(code)

        metrics = [json.loads(x) for x in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        train = [x for x in metrics if x.get("event") == "train"]
        full = [x for x in train if int(x.get("tokens_per_update", 0)) == TOKENS_PER_UPDATE]
        stable_rows = full[10:]
        if len(stable_rows) < 50:
            raise RuntimeError(f"too few post-warmup full updates: {len(stable_rows)}")
        rates = [float(x["tokens_per_second"]) for x in stable_rows]
        med = statistics.median(rates)
        low = p10(rates)
        mean = statistics.fmean(rates)
        peak = max(rates)
        peak_alloc = max(int(x.get("peak_allocated_bytes", 0)) for x in train)
        peak_reserved = max(int(x.get("peak_reserved_bytes", 0)) for x in train)
        max_temp = max((float(x["temp_c"]) for x in gpu), default=float("nan"))
        memory_ok = peak_alloc <= MAX_ALLOCATED_BYTES and peak_reserved <= MAX_RESERVED_BYTES
        throughput_ok = med >= args.target_tps and low >= 0.90 * args.target_tps and low >= 0.90 * med
        thermal_ok = throttle_events == 0 and (not math.isfinite(max_temp) or max_temp < 78.0)
        summary = load_json(out / "summary.json")

        state_sha_after = sha256_file(state_path)
        anchor_sha_after = file_sha(anchor)
        read_only_ok = state_sha_after == state_sha_before and anchor_sha_after == anchor_sha_before
        gate_pass = bool(read_only_ok and memory_ok and throughput_ok and thermal_ok)
        result = {
            "schema": 1,
            "benchmark": "neral350-v8.2-thermal-soak-b",
            "anchor_sha256": anchor_sha,
            "anchor_tokens": anchor_tokens,
            "profile": profile,
            "input_profile": selected_input,
            "requested_tokens": args.tokens,
            "completed_new_tokens": summary.get("new_tokens"),
            "post_warmup_full_samples": len(rates),
            "median_tps": med,
            "mean_tps": mean,
            "p10_tps": low,
            "peak_tps": peak,
            "target_tps": args.target_tps,
            "peak_allocated_gib": peak_alloc / 1024**3,
            "peak_reserved_gib": peak_reserved / 1024**3,
            "max_sampled_temp_c": max_temp,
            "thermal_throttle_events": throttle_events,
            "wall_seconds": wall,
            "memory_ok": memory_ok,
            "throughput_ok": throughput_ok,
            "thermal_ok": thermal_ok,
            "read_only_identity_verified": read_only_ok,
            "gate_b_pass": gate_pass,
            "promotion": False,
            "stable_50k_proven_for_5m_stage": gate_pass,
        }
        (report / "thermal_soak.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        shutil.copy2(out / "metrics.jsonl", report / "thermal_soak_metrics.jsonl")
        shutil.copy2(out / "summary.json", report / "thermal_soak_summary.json")
        print(json.dumps({"event": "v8_2_thermal_soak", **result}), flush=True)
        if not gate_pass:
            raise SystemExit(3)
    finally:
        fast2_trainer.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
