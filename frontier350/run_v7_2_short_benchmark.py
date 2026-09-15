from __future__ import annotations

"""Read-only short benchmark suite for the final Neral 350M V7.2 continuity checkpoint.

The suite never trains, saves a checkpoint, changes optimizer state, or edits continuity
metadata. It verifies the exact 2B checkpoint identity before and after evaluation.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from recovery_identity import file_sha

TARGET_TOKENS = 2_000_000_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_eval(script: str, args: list[str], report: Path, timings: dict[str, float]) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *args]
    print(json.dumps({"event": "short_bench_start", "script": script, "argv": cmd}), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    timings[script] = time.perf_counter() - t0
    print(json.dumps({"event": "short_bench_done", "script": script, "seconds": timings[script]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--rotation", type=int, choices=range(4), default=0)
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report directory must be empty")
    report.mkdir(parents=True, exist_ok=True)

    state_path = assets / "frontier350_scale_v5_state.json"
    state_sha_before = file_sha(state_path)
    state = read_json(state_path)
    cont = state.get("continuity") or {}
    checkpoint = Path(str(cont.get("path", "")))
    expected_sha = str(cont.get("sha256", ""))
    tokens = int(cont.get("tokens", -1))

    if tokens != TARGET_TOKENS:
        raise RuntimeError(f"expected exact 2B continuity, got {tokens:,}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha_before = file_sha(checkpoint)
    if checkpoint_sha_before != expected_sha:
        raise RuntimeError("continuity checkpoint SHA mismatch")

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/bilingual_holdout.json"
    for required in (tokenizer, valid, cases):
        if not required.exists():
            raise FileNotFoundError(required)

    timings: dict[str, float] = {}
    suite_t0 = time.perf_counter()

    validation_out = report / "validation_short.json"
    run_eval(
        "eval_validation_loss350.py",
        ["--checkpoint", str(checkpoint), "--valid-data", str(valid), "--out", str(validation_out),
         "--seq", "512", "--batch", "4", "--batches", "8", "--seed", "351337"],
        report, timings,
    )

    smoke_out = report / "recovery_smoke.json"
    run_eval(
        "eval_recovery_smoke_v3.py",
        ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer), "--out", str(smoke_out),
         "--expected-tokens", str(TARGET_TOKENS), "--rotation", str(args.rotation)],
        report, timings,
    )

    nis_out = report / "nis2_core.json"
    run_eval(
        "nis2_core350.py",
        ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer), "--out", str(nis_out),
         "--expected-tokens", str(TARGET_TOKENS), "--max-new", "24"],
        report, timings,
    )

    checkpoint_sha_after = file_sha(checkpoint)
    state_sha_after = file_sha(state_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during read-only benchmark")
    if state_sha_after != state_sha_before:
        raise RuntimeError("continuity state changed during read-only benchmark")

    validation = read_json(validation_out)
    smoke = read_json(smoke_out)
    nis = read_json(nis_out)
    val = float(validation["validation_loss"])
    if not math.isfinite(val):
        raise RuntimeError("non-finite validation loss")

    summary = {
        "schema": 1,
        "suite": "neral350-v7.2-2b-short-benchmark",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "checkpoint_processed_tokens": TARGET_TOKENS,
        "read_only_verified": True,
        "validation": {
            "loss": val,
            "perplexity": float(validation["validation_perplexity"]),
            "batches": int(validation["batches"]),
            "seq": int(validation["seq"]),
            "batch": int(validation["batch"]),
        },
        "smoke": {
            "rotation": int(smoke["rotation"]),
            "cases": int(smoke["cases"]),
            "overall_accuracy_pct": float(smoke["overall_accuracy_pct"]),
            "categories": smoke["categories"],
        },
        "nis2_core": nis["scores"],
        "full_nis2_status": nis["full_nis2"]["status"],
        "timings_seconds": timings,
        "wall_seconds": time.perf_counter() - suite_t0,
        "note": "Short read-only diagnostic. Smoke and NIS2-core are not a final public benchmark or SAFE promotion gate.",
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
