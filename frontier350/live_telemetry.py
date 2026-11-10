from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _tail_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def latest_event(path: Path) -> dict[str, Any] | None:
    for line in reversed(_tail_text(path).splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") in {"train", "validation", "startup"}:
            return row
    return None


def latest_train_event(path: Path) -> dict[str, Any] | None:
    for line in reversed(_tail_text(path).splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "train":
            return row
    return None


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not (seconds < float("inf")):
        return "n/a"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def make_body(
    *,
    row: dict[str, Any] | None,
    train_row: dict[str, Any] | None,
    run_id: str,
    workflow: str,
    metrics_path: Path,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if row is None:
        return (
            "# Arctus Live Training Telemetry\n\n"
            "**Status:** waiting for the first metrics row\n\n"
            f"- Workflow: `{workflow}`\n"
            f"- Run: `{run_id}`\n"
            f"- Metrics: `{metrics_path}`\n"
            f"- Last update: `{now}`\n\n"
            "_This issue is a live telemetry sink and may be overwritten by the next long run._"
        )

    base = train_row or row
    processed = int(base.get("processed_tokens", 0) or 0)
    run_until = int(base.get("run_until_tokens", base.get("target_tokens", 0)) or 0)
    target = int(base.get("target_tokens", run_until) or run_until)
    tps = float(base.get("tokens_per_second", 0.0) or 0.0)
    pct = (100.0 * processed / run_until) if run_until else 0.0
    remaining = max(0, run_until - processed)
    eta_s = remaining / tps if tps > 0 else None
    event = row.get("event", "unknown")
    validation = row.get("validation_loss") if event == "validation" else None

    lines = [
        "# Arctus Live Training Telemetry",
        "",
        f"**Status:** {'completed target' if run_until and processed >= run_until else 'training'}",
        "",
        f"- Workflow: `{workflow}`",
        f"- Run: `{run_id}`",
        f"- Event: `{event}`",
        f"- Processed: **{processed:,} / {run_until:,} tokens ({pct:.2f}%)**",
        f"- Global target: {target:,} tokens",
        f"- Step: {base.get('step', 'n/a')}",
        f"- Loss: **{fmt_num(base.get('loss'), 4)}**",
        f"- Throughput: **{tps:,.0f} tok/s**" if tps > 0 else "- Throughput: n/a",
        f"- ETA at latest TPS: **{human_duration(eta_s)}**",
        f"- Grad norm: {fmt_num(base.get('grad_norm'), 4)}",
        f"- GPU allocated: {fmt_num(base.get('gpu_mem_allocated_gb'), 2)} GiB",
        f"- GPU reserved: {fmt_num(base.get('gpu_mem_reserved_gb'), 2)} GiB",
        f"- GPU peak allocated: {fmt_num(base.get('gpu_peak_allocated_gb'), 2)} GiB",
    ]
    if validation is not None:
        lines.append(f"- Validation loss: **{fmt_num(validation, 4)}**")
    lines += [
        f"- Last update: `{now}`",
        "",
        "_Updated automatically from `metrics.jsonl`; this avoids GitHub running-job log `BlobNotFound` responses._",
    ]
    return "\n".join(lines)


def patch_issue(repo: str, issue: int, token: str, body: str) -> None:
    payload = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{issue}",
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "neral-live-telemetry",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"GitHub telemetry update failed with HTTP {response.status}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workflow", default="Arctus long run")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    args = ap.parse_args()

    if args.interval < 30:
        raise ValueError("telemetry interval must be at least 30 seconds")

    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"missing {args.token_env}")

    metrics = Path(args.metrics)
    last_body = ""
    failures = 0

    while True:
        row = latest_event(metrics)
        train_row = latest_train_event(metrics)
        body = make_body(
            row=row,
            train_row=train_row,
            run_id=args.run_id,
            workflow=args.workflow,
            metrics_path=metrics,
        )
        if body != last_body:
            try:
                patch_issue(args.repo, args.issue, token, body)
                last_body = body
                failures = 0
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                failures += 1
                print(f"telemetry publish failed ({failures}): {exc}", flush=True)

        base = train_row or row or {}
        processed = int(base.get("processed_tokens", 0) or 0)
        run_until = int(base.get("run_until_tokens", base.get("target_tokens", 0)) or 0)
        if run_until and processed >= run_until:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
