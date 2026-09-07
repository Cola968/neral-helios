"""Run the complete frozen Neral model-only benchmark bundle for one checkpoint.

The bundle executes free100, v3, v4, RawMath500, v5 long-context and Targeted v1. It is
orchestration only and does not modify individual scoring rules. No SOLVE/Helios/external
model is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPORTS = ("free.json", "v3.json", "v4.json", "rawmath.json", "v5.json", "targeted.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    commands = [
        [py, "free_generation50.py", "--checkpoint", str(checkpoint), "--out", str(out / "free.json"), "--max-tokens", "64"],
        [py, "benchmark_v3.py", str(checkpoint), "--out", str(out / "v3.json"), "--max-new-tokens", "96"],
        [py, "benchmark_v4.py", str(checkpoint), "--out", str(out / "v4.json"), "--max-new-tokens", "96"],
        [py, "benchmark_raw_math50.py", str(checkpoint), "--out", str(out / "rawmath.json"), "--max-new-tokens", "32"],
        [py, "benchmark_v5_long_context.py", str(checkpoint), "--out", str(out / "v5.json")],
        [py, "benchmark_targeted50.py", str(checkpoint), "--out", str(out / "targeted.json")],
    ]
    for cmd in commands:
        run(cmd)
    files = {name: sha256(out / name) for name in REPORTS}
    targeted = json.loads((out / "targeted.json").read_text(encoding="utf8"))
    manifest = {
        "definition": "Complete frozen Neral model-only benchmark bundle; no SOLVE/Helios/external model.",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "reports_sha256": files,
        "targeted_suite_sha256": targeted["suite_sha256"],
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
