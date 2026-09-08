"""Lightweight live telemetry sidecar for Neral-50M I1.

This module never imports torch, opens checkpoints, mutates trainer state, or touches the
training corpus. It only reads JSON/JSONL metrics already emitted by I1 stages, optionally
queries nvidia-smi for monitoring data, prints compact live lines, and atomically writes a
small report file for the runner.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = (
    {
        "name": "B3a",
        "run_dir": "runs/neral50m_i1_b3a_300m",
        "kind": "tokens",
        "target": 300_000_000,
        "progress_key": "cumulative_tokens",
    },
    {
        "name": "B3b",
        "run_dir": "runs/neral50m_i1_b3b_500m",
        "kind": "tokens",
        "target": 500_000_000,
        "progress_key": "cumulative_tokens",
    },
    {
        "name": "RawMath C3",
        "run_dir": "runs/neral50m_i1_rawmath_c3",
        "kind": "tokens",
        "target": 30_000_000,
        "progress_key": "stage_c_tokens",
    },
    {
        "name": "Instruct D3",
        "run_dir": "runs/neral50m_i1_instruct_d3",
        "kind": "steps",
        "target": 4_000,
        "progress_key": "step",
    },
)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def read_last_jsonl(path: Path, chunk_size: int = 65_536) -> dict | None:
    """Read the last complete JSON object without scanning a growing metrics file."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return None
            data = b""
            pos = end
            while pos > 0 and data.count(b"\n") < 2:
                size = min(chunk_size, pos)
                pos -= size
                f.seek(pos)
                data = f.read(size) + data
            for raw in reversed(data.splitlines()):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    return value
    except OSError:
        return None
    return None


def nested_number(row: dict | None, *path: str) -> float | None:
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def gpu_snapshot() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        first = result.stdout.strip().splitlines()[0]
        values = [x.strip() for x in first.split(",")]
        if len(values) != 5:
            return {}
        temp, power, used, total, util = (float(x) for x in values)
        return {
            "temperature_c": temp,
            "power_w": power,
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_pct": util,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def stage_complete(summary: dict | None) -> bool:
    return bool(summary and summary.get("complete") is True)


def choose_stage(assets: Path) -> tuple[dict, Path, dict | None, dict | None]:
    last = STAGES[-1]
    for stage in STAGES:
        run_dir = assets / stage["run_dir"]
        summary = read_json(run_dir / "summary.json")
        metrics = read_last_jsonl(run_dir / "metrics.jsonl")
        if not stage_complete(summary):
            return stage, run_dir, summary, metrics
        last = stage
    run_dir = assets / last["run_dir"]
    return last, run_dir, read_json(run_dir / "summary.json"), read_last_jsonl(run_dir / "metrics.jsonl")


def source_progress(run_dir: Path, stage: dict) -> float:
    run = read_json(run_dir / "run.json") or {}
    if stage["name"] in {"B3a", "B3b"}:
        value = run.get("source_cumulative_tokens", 0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    return 0.0


def collect_status(assets: Path, *, include_gpu: bool = True) -> dict:
    stage, run_dir, summary, row = choose_stage(assets)
    row = row or {}
    progress_key = stage["progress_key"]
    current = nested_number(row, progress_key)
    if current is None and summary:
        summary_key = "completed_step" if stage["kind"] == "steps" else progress_key
        current = nested_number(summary, summary_key)
    current = current or 0.0
    start = source_progress(run_dir, stage)
    target = float(stage["target"])
    denom = max(1.0, target - start)
    progress_pct = max(0.0, min(100.0, (current - start) / denom * 100.0)) if stage["name"] in {"B3a", "B3b"} else max(0.0, min(100.0, current / target * 100.0))

    train_loss = nested_number(row, "train_loss")
    if train_loss is None:
        train_loss = nested_number(row, "loss")
    tps = nested_number(row, "tokens_per_second")
    if tps is None:
        tps = nested_number(row, "optimizer_tokens_per_second")

    status = {
        "campaign": "Neral-50M I1",
        "stage": stage["name"],
        "kind": stage["kind"],
        "current": int(current),
        "target": int(target),
        "progress_pct": round(progress_pct, 3),
        "step": int(nested_number(row, "step") or nested_number(summary, "completed_step") or 0),
        "train_loss": train_loss,
        "tokens_per_second": tps,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "complete": all(stage_complete(read_json(assets / item["run_dir"] / "summary.json")) for item in STAGES),
    }

    if stage["name"] in {"B3a", "B3b"}:
        status["validation_loss"] = nested_number(row, "validation", "macro")
        status["old_validation_loss"] = nested_number(row, "old_validation", "macro")
    elif stage["name"] == "RawMath C3":
        status["math_dev_nll"] = nested_number(row, "math_dev", "nll")
        status["language_validation_loss"] = nested_number(row, "language_validation", "macro")
    else:
        status["sft_validation_nll"] = nested_number(row, "sft_validation", "macro_answer_nll")
        status["base_validation_loss"] = nested_number(row, "base_validation", "macro")
        status["math_dev_nll"] = nested_number(row, "math_dev", "nll")

    if stage["kind"] == "tokens" and tps and tps > 0:
        remaining = max(0.0, target - current)
        status["eta_seconds_optimizer_only"] = round(remaining / tps, 1)

    if include_gpu:
        status["gpu"] = gpu_snapshot()
    return status


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    os.replace(tmp, path)


def compact_line(status: dict) -> str:
    parts = [
        f"stage={status['stage']}",
        f"progress={status['progress_pct']:.1f}%",
        f"step={status['step']}",
    ]
    if status.get("train_loss") is not None:
        parts.append(f"loss={status['train_loss']:.4f}")
    if status.get("validation_loss") is not None:
        parts.append(f"val={status['validation_loss']:.4f}")
    if status.get("tokens_per_second") is not None:
        parts.append(f"tok/s={status['tokens_per_second']:.0f}")
    gpu = status.get("gpu") or {}
    if gpu.get("temperature_c") is not None:
        parts.append(f"gpu={gpu['temperature_c']:.0f}C")
    if gpu.get("power_w") is not None:
        parts.append(f"power={gpu['power_w']:.0f}W")
    return "[I1 LIVE] " + " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--reports", default="i1_reports")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--stop-file")
    ap.add_argument("--no-gpu-monitor", action="store_true")
    args = ap.parse_args()
    if args.interval < 5:
        raise ValueError("Live telemetry interval must be at least 5 seconds")

    assets = Path(args.assets).resolve()
    reports = Path(args.reports).resolve()
    stop_file = Path(args.stop_file).resolve() if args.stop_file else None
    output = reports / "LIVE_STATUS.json"
    if not assets.is_dir():
        raise FileNotFoundError(assets)

    while True:
        status = collect_status(assets, include_gpu=not args.no_gpu_monitor)
        atomic_write_json(output, status)
        print(compact_line(status), flush=True)
        if status["complete"] or (stop_file and stop_file.exists()):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
