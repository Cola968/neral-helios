from __future__ import annotations

"""Run Frontier 350M training while mirroring numeric metrics to local/optional Trackio telemetry.

Only numeric training metrics are forwarded. Corpus text, prompts, tokenizer vocabulary,
weights, checkpoints and local paths are never sent to Trackio.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_observability import TrainingObserver


def read_new(path: Path, offset: int) -> tuple[int, list[dict]]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf8", errors="replace") as f:
            f.seek(offset)
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
            return f.tell(), rows
    except (FileNotFoundError, OSError):
        return offset, []


def numeric_metrics(row: dict) -> dict[str, float | int]:
    mapping = {
        "step": "source_step",
        "processed_tokens": "tokens_total",
        "progress_total": "progress_total",
        "loss": "train_loss",
        "validation_loss": "val_loss",
        "best_validation_loss": "best_val_loss",
        "lr": "learning_rate",
        "grad_norm": "grad_norm",
        "tokens_per_second": "tokens_per_second",
        "gpu_mem_allocated_gb": "gpu_mem_allocated_gb",
        "gpu_mem_reserved_gb": "gpu_mem_reserved_gb",
        "gpu_peak_allocated_gb": "gpu_peak_allocated_gb",
    }
    out: dict[str, float | int] = {}
    for src, dst in mapping.items():
        value = row.get(src)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[dst] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--telemetry-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing wrapped training command")

    metrics_path = Path(args.metrics)
    observer = TrainingObserver(
        Path(args.telemetry_dir),
        run_name=args.run_name,
        group="frontier350-training",
        config={"bridge": "frontier350-jsonl-v1", "poll_seconds": args.poll_seconds},
    )
    offset = 0
    bridge_step = 0
    status = "failed"
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            offset, rows = read_new(metrics_path, offset)
            for row in rows:
                values = numeric_metrics(row)
                if values:
                    bridge_step += 1
                    observer.log(bridge_step, values)
                    if values.get("tokens_per_second", 999999) < 1000 and values.get("tokens_total", 0) > 1_000_000:
                        observer.alert("Frontier throughput low", f"tokens_per_second={values['tokens_per_second']:.1f}", "WARN", bridge_step)
            time.sleep(args.poll_seconds)
        offset, rows = read_new(metrics_path, offset)
        for row in rows:
            values = numeric_metrics(row)
            if values:
                bridge_step += 1
                observer.log(bridge_step, values)
        if process.returncode:
            observer.alert("Frontier training failed", f"exit_code={process.returncode}", "ERROR", bridge_step)
            raise SystemExit(process.returncode)
        status = "completed"
    finally:
        if process.poll() is None:
            process.terminate()
        observer.finish(status)


if __name__ == "__main__":
    main()
