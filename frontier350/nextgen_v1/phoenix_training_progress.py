"""Read-only, local Phoenix pilot progress from actual on-disk training receipts.

Does not train, load checkpoints, import Torch, certify GPU performance or grade answers.
Never uploads, persists or prints private paths, prompts or manifest contents.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_NEW_TOKENS = 262_144
SHA = re.compile(r"[0-9a-f]{64}\Z")


def integer(value, name: str, *, minimum=0, maximum=MAX_NEW_TOKENS) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name} in training receipt")
    return value


def finite(value, name: str, *, positive=False) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"invalid {name} in training receipt")
    if positive and value <= 0:
        raise ValueError(f"invalid {name} in training receipt")
    return float(value)


def regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("missing, symlinked or oversized local training receipt")
    return path


def read_log(path: Path) -> tuple[list[dict], bool]:
    text = regular_file(path).read_text(encoding="utf-8")
    rows: list[dict] = []
    lines = text.splitlines(keepends=True)
    partial = False
    for n, line in enumerate(lines):
        if not line.endswith("\n"):
            if n == len(lines) - 1:
                partial = True  # The trainer could be writing the final line.
                continue
            raise ValueError("incomplete middle training receipt")
        try:
            row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid complete training receipt") from exc
        if not isinstance(row, dict):
            raise ValueError("non-object training receipt")
        rows.append(row)
    if not rows:
        raise ValueError("no complete training receipt found")
    return rows, partial


def summarize(run_dir: Path) -> dict:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("local run directory must exist and not be a symlink")
    rows, partial = read_log(run_dir / "metrics.jsonl")
    target = None
    seen_start = False
    last_step = last_tokens = 0
    invocation_start = 0
    latest_val = None
    elapsed = None
    steps = 0
    speed = None
    for row in rows:
        event = row.get("event")
        if event == "separate_weight_only_pilot":
            lineage = row.get("lineage")
            if not isinstance(lineage, dict):
                raise ValueError("missing native pilot lineage")
            candidate = integer(lineage.get("new_target_tokens"), "target", minimum=1)
            if target is not None and candidate != target:
                raise ValueError("training target changed between invocations")
            target = candidate
            invocation_start = integer(row.get("new_processed_tokens"), "invocation start", maximum=target)
            start_step = integer(row.get("new_step"), "start step", maximum=MAX_NEW_TOKENS)
            if seen_start and (invocation_start != last_tokens or start_step != last_step):
                raise ValueError("resume receipt does not continue preceding step")
            if not seen_start:
                last_tokens, last_step = invocation_start, start_step
            seen_start = True
            elapsed = speed = None
        elif event == "step":
            if not seen_start:
                raise ValueError("training step without initialization receipt")
            step = integer(row.get("new_step"), "step", minimum=1)
            tokens = integer(row.get("new_tokens"), "new tokens", maximum=target)
            loss = finite(row.get("loss"), "training loss")
            finite(row.get("grad_norm"), "gradient norm")
            finite(row.get("lr"), "learning rate", positive=True)
            duration = finite(row.get("elapsed_s"), "elapsed time")
            if step != last_step + 1 or tokens <= last_tokens or duration < 0:
                raise ValueError("nonmonotonic or missing optimizer step")
            last_step, last_tokens = step, tokens
            steps += 1
            elapsed = duration
            speed = round((tokens - invocation_start) / duration, 2) if duration > 0 else None
        elif event == "validation":
            if not seen_start:
                raise ValueError("validation without initialization receipt")
            token = integer(row.get("new_tokens"), "validation tokens", maximum=target)
            step = integer(row.get("new_step"), "validation step", minimum=1)
            if step > last_step + 1 or token < last_tokens or token > target:
                raise ValueError("validation does not match optimizer sequence")
            latest_val = finite(row.get("loss"), "validation loss")
        else:
            raise ValueError("unknown training receipt event")
    if not seen_start:
        raise ValueError("no native pilot initialization receipt")
    completed = False
    stale_summary_ignored = False
    summary_path = run_dir / "summary.json"
    # An exact resume can leave the previous invocation's summary on disk.
    # It must not be misread as proof that the new invocation has finished.
    if summary_path.exists() or summary_path.is_symlink():
        if summary_path.is_symlink() or not summary_path.is_file():
            raise ValueError("invalid local completion receipt")
        stale_summary_ignored = (
            rows[-1].get("event") == "separate_weight_only_pilot" or
            summary_path.stat().st_mtime_ns < (run_dir / "metrics.jsonl").stat().st_mtime_ns
        )
    if (summary_path.exists() or summary_path.is_symlink()) and not stale_summary_ignored:
        summary = json.loads(regular_file(summary_path).read_text(encoding="utf-8"))
        if (not isinstance(summary, dict) or summary.get("event") != "weight_only_pilot_finished"
                or summary.get("source_unchanged") is not True
                or integer(summary.get("new_processed_tokens"), "summary tokens", maximum=target) != last_tokens
                or integer(summary.get("new_optimizer_steps"), "summary step", maximum=MAX_NEW_TOKENS) != last_step
                or not isinstance(summary.get("resume_sha256"), str)
                or not SHA.fullmatch(summary["resume_sha256"])):
            raise ValueError("completion receipt does not match training log")
        completed = True
    return {"status": "summary_reported_checkpoint_not_verified" if completed else "in_progress_or_interrupted",
            "new_tokens": last_tokens, "target_new_tokens": target,
            "progress_percent": round(100 * last_tokens / target, 2),
            "optimizer_steps": last_step, "step_receipts": steps,
            "last_training_loss": loss if steps else None,
            "last_validation_loss": latest_val,
            "observed_wall_tokens_per_second_including_overheads": speed,
            "invocation_elapsed_seconds": elapsed,
            "incomplete_last_log_line_ignored": partial,
            "stale_summary_ignored": stale_summary_ignored,
            "checkpoint_bytes_or_gpu_verified": False,
            "quality_improvement_verified": False,
            "training_started_by_this_tool": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--follow", action="store_true", help="Print changed receipts every 3 seconds")
    args = parser.parse_args(argv)
    last = None
    while True:
        try:
            result = summarize(args.run_dir)
        except (ValueError, OSError, UnicodeError, KeyError, TypeError) as exc:
            result = {"status": "receipt_unavailable_or_invalid", "error_type": type(exc).__name__,
                      "training_started_by_this_tool": False}
        output = json.dumps(result, ensure_ascii=False, sort_keys=True)
        if output != last:
            print(output, flush=True)
            last = output
        if not args.follow or result["status"] == "summary_reported_checkpoint_not_verified":
            return 0 if result["status"] != "receipt_unavailable_or_invalid" else 1
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())