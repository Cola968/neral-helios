from __future__ import annotations

"""Read-only V8 NIS2-v2 readiness benchmark for the current continuity checkpoint."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from recovery_identity import file_sha


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--expected-tokens", type=int, required=True)
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report directory must be empty")
    report.mkdir(parents=True, exist_ok=True)

    state_path = assets / "frontier350_scale_v5_state.json"
    state_sha_before = file_sha(state_path)
    state = read_json(state_path)
    continuity = state.get("continuity") or {}
    checkpoint = Path(str(continuity.get("path", "")))
    expected_sha = str(continuity.get("sha256", ""))
    tokens = int(continuity.get("tokens", -1))
    if tokens != int(args.expected_tokens):
        raise RuntimeError(f"expected {args.expected_tokens:,} continuity tokens, got {tokens:,}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha_before = file_sha(checkpoint)
    if checkpoint_sha_before != expected_sha:
        raise RuntimeError("continuity checkpoint SHA mismatch")

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)

    nis_out = report / "nis2_chat_core_v2_140.json"
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).parent / "nis2_core350.py"),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        "--out", str(nis_out),
        "--expected-tokens", str(tokens),
        "--token-tolerance", "0",
        "--max-new", "32",
        "--health-max-new", "48",
    ]
    print(json.dumps({"event": "v8_nis2_start", "argv": cmd}), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    seconds = time.perf_counter() - t0

    checkpoint_sha_after = file_sha(checkpoint)
    state_sha_after = file_sha(state_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during read-only NIS2 benchmark")
    if state_sha_after != state_sha_before:
        raise RuntimeError("continuity state changed during read-only NIS2 benchmark")

    nis = read_json(nis_out)
    counts = nis.get("case_counts") or {}
    if int(counts.get("total", -1)) != 140:
        raise RuntimeError(f"expected 140 NIS2-v2 cases, got {counts}")
    summary = {
        "schema": 1,
        "suite": "neral350-v8-nis2-readiness",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "checkpoint_processed_tokens": tokens,
        "read_only_verified": True,
        "benchmark": nis["benchmark"],
        "suite_sha256": nis["suite_sha256"],
        "case_counts": counts,
        "scores": nis["scores"],
        "full_nis2_status": nis["full_nis2"]["status"],
        "wall_seconds": seconds,
        "note": "NIS2 chat-core v2 is a 140-case readiness diagnostic; it is not the canonical full NIS2 score or a SAFE promotion gate.",
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
