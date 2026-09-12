"""RTX 4070 throughput sweep for the guarded Neral 350M continuation path.

No benchmark checkpoint is promoted. Every trial starts from the same accepted 1.010B
checkpoint. The sweep compares the exact legacy path against a replay-only fast path that
removes redundant flattened gradient copies and per-microbatch host synchronizations, then
packs adjacent MAIN microbatches into larger physical forwards. The optimizer-update token
budget, 25% replay share, three replay domains, data, LR and loss weighting stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from recovery_identity import file_sha

ANCHOR_SHA = "ce518b87e19e1064be40b7a33f16cf4b06ab2284af30526c76729ef074c2d3f2"
ANCHOR_REL = "runs/neral350_autolearn_neral-autolearn-v1-20260912-r2_r02_selected/resume.pt"
SEQ = 512
BATCH = 4
MICROBATCHES = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES
MAX_ALLOCATED = int(10.25 * 1024**3)
MAX_RESERVED = int(11.0 * 1024**3)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def gpu_state() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def percentile10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--target-tps", type=float, default=50_000.0)
    ap.add_argument("--updates", type=int, default=14)
    ap.add_argument("--warmup-updates", type=int, default=4)
    a = ap.parse_args()
    if not 8 <= a.updates <= 30 or not 1 <= a.warmup_updates < a.updates - 2:
        raise ValueError("benchmark update/warmup bounds rejected")
    if a.target_tps < 10_000:
        raise ValueError("implausible target")
    if not a.request_id or len(a.request_id) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in a.request_id):
        raise ValueError("unsafe request id")

    assets = Path(a.assets).resolve()
    report = Path(a.report).resolve()
    scratch = Path(a.scratch).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report directory is not empty")
    if scratch.exists():
        raise FileExistsError("scratch directory already exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    anchor = assets / ANCHOR_REL
    if not anchor.is_file() or file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("accepted 1.010B Auto-Learn anchor missing or checksum changed")
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/train"
    replay = assets / "frontier350_replay_v9_20260912"
    if not main_data.is_dir() or not replay.is_dir():
        raise FileNotFoundError("training corpora missing")

    trainer = Path(__file__).parent / "train_guarded_v9.py"
    budget = TOKENS_PER_UPDATE * a.updates
    results: list[dict] = []

    def run_trial(name: str, pack: int, compile_mode: str, fast: bool) -> dict:
        out = scratch / name
        log = report / f"{name}.log"
        metrics_copy = report / f"{name}_metrics.jsonl"
        summary_copy = report / f"{name}_summary.json"
        before = gpu_state()
        cmd = [
            sys.executable, "-u", str(trainer),
            "--anchor", str(anchor), "--anchor-sha", ANCHOR_SHA,
            "--main-data", str(main_data), "--replay", str(replay),
            "--out", str(out), "--mode", "replay_only",
            "--new-tokens", str(budget), "--max-updates", str(a.updates),
            "--batch-size", str(BATCH), "--microbatches", str(MICROBATCHES),
            "--main-pack", str(pack), "--compile-mode", compile_mode,
        ]
        if fast:
            cmd.append("--fast-replay-only")
        started = time.perf_counter()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - started
        after = gpu_state()
        row: dict = {
            "name": name,
            "main_pack": pack,
            "main_physical_batch": BATCH * min(pack, MICROBATCHES - 3),
            "compile_mode": compile_mode,
            "fast_replay_only": fast,
            "returncode": proc.returncode,
            "wall_seconds": wall,
            "gpu_before": before,
            "gpu_after": after,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "replay_fraction": 3 / MICROBATCHES,
        }
        try:
            if proc.returncode != 0:
                tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
                row.update(pass_memory=False, stable=False, error=tail)
                return row
            metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
            train = [r for r in metrics if r.get("event") == "train"]
            if len(train) != a.updates:
                raise ValueError(f"expected {a.updates} train rows, got {len(train)}")
            stable_rows = train[a.warmup_updates:]
            rates = [float(r["tokens_per_second"]) for r in stable_rows]
            if not rates or not all(math.isfinite(x) and x > 0 for x in rates):
                raise ValueError("invalid throughput samples")
            peak_alloc = max(int(r["peak_allocated_bytes"]) for r in train)
            peak_res = max(int(r.get("peak_reserved_bytes", 0)) for r in train)
            summary = load_json(out / "summary.json")
            if summary.get("new_tokens") != budget or summary.get("tokens_per_update") != TOKENS_PER_UPDATE:
                raise ValueError("token-accounting contract changed")
            if bool(summary.get("fast_replay_only")) != fast or int(summary.get("main_pack", -1)) != pack:
                raise ValueError("execution contract evidence mismatch")
            shutil.copy2(out / "metrics.jsonl", metrics_copy)
            shutil.copy2(out / "summary.json", summary_copy)
            med = statistics.median(rates)
            p10 = percentile10(rates)
            mean = statistics.fmean(rates)
            peak = max(rates)
            pass_memory = peak_alloc <= MAX_ALLOCATED and peak_res <= MAX_RESERVED
            stable_ok = p10 >= 0.90 * med
            row.update(
                median_tps=med, mean_tps=mean, p10_tps=p10, peak_tps=peak,
                stable_samples=len(rates), peak_allocated_bytes=peak_alloc,
                peak_reserved_bytes=peak_res, peak_allocated_gib=peak_alloc / 1024**3,
                peak_reserved_gib=peak_res / 1024**3, pass_memory=pass_memory,
                stable=stable_ok,
                target_reached=bool(pass_memory and stable_ok and med >= a.target_tps and p10 >= 0.90 * a.target_tps),
            )
            return row
        except Exception as exc:
            row.update(pass_memory=False, stable=False, error=f"evidence parse failed: {exc}")
            return row
        finally:
            shutil.rmtree(out, ignore_errors=True)

    # Exact legacy baseline first.
    baseline = run_trial("legacy_pack1_default", 1, "default", False)
    results.append(baseline);print(json.dumps({"event":"trial",**baseline}),flush=True)

    # Fast path: same update semantics, then progressively larger main physical batches.
    for pack in (1, 2, 3, 4):
        row = run_trial(f"fast_pack{pack}_default", pack, "default", True)
        results.append(row);print(json.dumps({"event":"trial",**row}),flush=True)

    valid_fast = [r for r in results if r.get("fast_replay_only") and r.get("returncode") == 0 and r.get("pass_memory") and r.get("stable") and "median_tps" in r]
    if not valid_fast:
        raise RuntimeError("no stable fast-path candidate survived")
    best_default = max(valid_fast, key=lambda r: r["median_tps"])

    # Compiler-mode sweep only on the best safe fast-path geometry.
    for mode in ("reduce-overhead", "max-autotune-no-cudagraphs"):
        row = run_trial(f"fast_pack{best_default['main_pack']}_{mode}", int(best_default["main_pack"]), mode, True)
        results.append(row);print(json.dumps({"event":"trial",**row}),flush=True)

    eligible = [r for r in results if r.get("fast_replay_only") and r.get("returncode") == 0 and r.get("pass_memory") and r.get("stable") and "median_tps" in r]
    best = max(eligible, key=lambda r: r["median_tps"])
    if "median_tps" not in baseline:
        raise RuntimeError("legacy baseline did not produce a valid measurement")
    speedup = best["median_tps"] / baseline["median_tps"]
    selection = {
        "schema": 2,
        "benchmark": "neral350-throughput-50k-v1",
        "request_id": a.request_id,
        "anchor_sha256": ANCHOR_SHA,
        "anchor_tokens": 1_010_003_584,
        "target_tps": a.target_tps,
        "updates_per_trial": a.updates,
        "warmup_updates_excluded": a.warmup_updates,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "replay_fraction": 0.25,
        "semantics": "same 24,576 tokens/update, same 25% replay share and weighted objective; fast path removes redundant replay_only vector copies/host syncs and may pack adjacent main microbatches",
        "memory_limits": {"allocated_bytes": MAX_ALLOCATED, "reserved_bytes": MAX_RESERVED},
        "baseline": baseline,
        "best": best,
        "speedup_vs_legacy": speedup,
        "target_reached": bool(best.get("target_reached")),
        "results": results,
        "promotion": False,
        "next_gate": "Winning fast path must pass a bounded 1M-token canary with validation/dev/bilingual retention gates before adoption.",
    }
    (report / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"event":"selection",**selection}),flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
