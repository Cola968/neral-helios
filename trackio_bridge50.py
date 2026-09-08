"""Run a Neral-50M campaign while mirroring numeric metrics.jsonl rows to TrainingObserver/Trackio.

The bridge never reads corpus files or checkpoints. It only tails JSONL telemetry under
<assets>/runs and forwards numeric metrics. This lets legacy 50M trainers gain live metrics
without changing their weight-update code.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from training_observability import TrainingObserver

SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def phase_name(path: Path) -> str:
    return SAFE.sub("_", path.parent.name)[:80] or "phase"


def flatten_numeric(row: dict) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    aliases = {
        "loss": "train_loss",
        "train_loss": "train_loss",
        "lr": "learning_rate",
        "learning_rate": "learning_rate",
        "tokens": "tokens_total",
        "tokens_total": "tokens_total",
        "tokens_per_second": "tokens_per_second",
        "tok_s": "tokens_per_second",
        "grad_norm": "grad_norm",
        "step": "source_step",
    }
    for src, dst in aliases.items():
        value = row.get(src)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[dst] = value
    validation = row.get("validation")
    if isinstance(validation, dict):
        macro = validation.get("macro")
        if isinstance(macro, (int, float)):
            out["val_loss"] = macro
        domains = validation.get("domains")
        if isinstance(domains, dict):
            for key, value in domains.items():
                if isinstance(value, (int, float)):
                    out[f"val_{SAFE.sub('_', str(key))[:40]}"] = value
    for key in ("math_dev_nll", "sft_macro_nll", "free_repeated_trigram_pct", "free_eos_pct"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    return out


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


def scan(root: Path, offsets: dict[Path, int], observer: TrainingObserver, bridge_step: int, started: float) -> int:
    runs = root / "runs"
    if not runs.is_dir():
        return bridge_step
    for path in sorted(runs.rglob("metrics.jsonl")):
        try:
            if path.stat().st_mtime < started - 60:
                continue
        except OSError:
            continue
        offset, rows = read_new(path, offsets.get(path, 0))
        offsets[path] = offset
        phase = phase_name(path)
        for row in rows:
            metrics = flatten_numeric(row)
            if not metrics:
                continue
            bridge_step += 1
            observer.log(bridge_step, {f"{phase}/{k}": v for k, v in metrics.items()})
    return bridge_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--telemetry-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing wrapped training command")
    if not 1.0 <= args.poll_seconds <= 120.0:
        raise ValueError("poll-seconds must be 1..120")

    assets = Path(args.assets).resolve()
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    observer = TrainingObserver(
        Path(args.telemetry_dir),
        run_name=args.run_name,
        group="50m-training",
        config={"bridge": "metrics-jsonl-v1", "poll_seconds": args.poll_seconds},
    )
    started = time.time()
    offsets: dict[Path, int] = {}
    bridge_step = 0
    status = "failed"
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            bridge_step = scan(assets, offsets, observer, bridge_step, started)
            time.sleep(args.poll_seconds)
        bridge_step = scan(assets, offsets, observer, bridge_step, started)
        if process.returncode:
            observer.alert("Wrapped training failed", f"exit_code={process.returncode}", "ERROR")
            raise SystemExit(process.returncode)
        status = "completed"
    finally:
        if process.poll() is None:
            process.terminate()
        observer.finish(status)


if __name__ == "__main__":
    main()
