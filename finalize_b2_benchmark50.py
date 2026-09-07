"""Benchmark a completed local Neral-50M B2 checkpoint without changing weights.

This is a handoff path for the local RTX runner: it refuses incomplete B2 runs, verifies
checkpoint identity, executes the hash-locked frozen benchmark bundle for 20M/B1/B2,
builds the existing B2 comparison, and applies the separate v5 long-context protection
gate. It never starts B3, RawMath, or instruction tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_B2_TOKENS = 100_000_000
FROZEN_VALIDATION_FILES = ("val_de.bin", "val_en.bin")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def validate_b2_summary(summary: dict) -> None:
    if not summary.get("complete"):
        raise ValueError("B2 is incomplete; benchmark handoff requires a completed stage")
    if summary.get("stop") != "target_reached":
        raise ValueError(f"B2 stop reason is not target_reached: {summary.get('stop')}")
    target = int(summary.get("target_cumulative_tokens", 0))
    cumulative = int(summary.get("cumulative_tokens", 0))
    if target < MIN_B2_TOKENS:
        raise ValueError(f"B2 target is below the frozen 100M milestone: {target:,}")
    if cumulative < target or cumulative < MIN_B2_TOKENS:
        raise ValueError(f"B2 cumulative tokens are incomplete: {cumulative:,}/{target:,}")
    if summary.get("train_block_replay") is not False:
        raise ValueError("B2 summary does not prove train_block_replay=False")
    if summary.get("instruction_tuning_performed") is not False:
        raise ValueError("B2 benchmark must precede instruction tuning")


def first_file(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing {label}; checked: " + ", ".join(map(str, paths)))


def resolve_state(assets: Path) -> dict[str, Path]:
    b2_dir = assets / "runs" / "neral50m_b2_100m"
    summary_path = b2_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing B2 summary: {summary_path}")
    summary = load_json(summary_path)
    validate_b2_summary(summary)

    baseline20 = first_file([
        assets / "export" / "pilot_02" / "neral20m.pt",
        assets / "checkpoints" / "neral20m.pt",
    ], "exact 20M Preview02 checkpoint")
    b1 = first_file([
        assets / "runs" / "neral50m_stage_b_full" / "best_validation.pt",
        assets / "runs" / "neral50m_stage_b_full" / "latest.pt",
    ], "B1 checkpoint")
    b2 = first_file([
        b2_dir / "best_validation.pt",
        b2_dir / "latest.pt",
    ], "B2 checkpoint")

    expected_best = summary.get("best_sha256")
    if b2.name == "best_validation.pt" and expected_best and sha256_file(b2) != expected_best:
        raise ValueError("B2 best checkpoint SHA-256 does not match summary.json")
    return {"baseline20": baseline20, "b1": b1, "b2": b2, "summary": summary_path.resolve()}


def stage_frozen_validation_data(assets: Path) -> tuple[dict[str, str], list[Path]]:
    """Copy private frozen validation binaries into the ephemeral trusted checkout only.

    The source files stay under the local assets root. They are never uploaded by this
    script. Newly staged files are removed in ``finally`` by the caller.
    """
    source_dir = assets / "pretrain50_data"
    dest_dir = ROOT / "pretrain50_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, str] = {}
    created: list[Path] = []
    for name in FROZEN_VALIDATION_FILES:
        source = source_dir / name
        dest = dest_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen validation source: {source}")
        source_hash = sha256_file(source)
        provenance[name] = source_hash
        if dest.exists():
            if sha256_file(dest) != source_hash:
                raise ValueError(f"Existing benchmark validation file differs from private frozen source: {dest}")
            continue
        shutil.copy2(source, dest)
        if sha256_file(dest) != source_hash:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Staged validation hash mismatch: {name}")
        created.append(dest)
    return provenance, created


def run_checked(args: list[str]) -> None:
    print(json.dumps({"exec": args}, ensure_ascii=False), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def run_benchmark(checkpoint: Path, stage: str, reports: Path) -> None:
    run_checked([
        sys.executable, "-X", "utf8", "run_frozen_benchmarks50.py",
        "--checkpoint", str(checkpoint), "--stage", stage, "--root", str(reports),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--reports", default="b2_final_benchmark_reports")
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    reports = Path(args.reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    state = resolve_state(assets)
    validation_hashes, staged_files = stage_frozen_validation_data(assets)

    try:
        # Same frozen protocol and budgets for every checkpoint. Existing valid reports can be
        # reused only through run_frozen_benchmarks50.py's provenance/hash checks.
        for stage in ("baseline20", "b1", "b2"):
            run_benchmark(state[stage], stage, reports)

        comparison_json = reports / "B2_COMPARISON.json"
        comparison_md = reports / "B2_COMPARISON.md"
        run_checked([
            sys.executable, "-X", "utf8", "summarize_b2.py",
            "--baseline20-free", str(reports / "baseline20" / "free.json"),
            "--baseline20-v3", str(reports / "baseline20" / "v3.json"),
            "--baseline20-v4", str(reports / "baseline20" / "v4.json"),
            "--baseline20-rawmath", str(reports / "baseline20" / "rawmath.json"),
            "--b1-free", str(reports / "b1" / "free.json"),
            "--b1-v3", str(reports / "b1" / "v3.json"),
            "--b1-v4", str(reports / "b1" / "v4.json"),
            "--b1-rawmath", str(reports / "b1" / "rawmath.json"),
            "--b2-free", str(reports / "b2" / "free.json"),
            "--b2-v3", str(reports / "b2" / "v3.json"),
            "--b2-v4", str(reports / "b2" / "v4.json"),
            "--b2-rawmath", str(reports / "b2" / "rawmath.json"),
            "--training-summary", str(state["summary"]),
            "--out-json", str(comparison_json), "--out-md", str(comparison_md),
        ])

        long_gate = reports / "B2_LONG_CONTEXT_GATE.json"
        gate = subprocess.run([
            sys.executable, "-X", "utf8", "long_context_gate50.py",
            str(reports / "b1" / "v5.json"), str(reports / "b2" / "v5.json"),
            "--mode", "protect", "--out", str(long_gate),
        ], cwd=ROOT)

        summary = load_json(state["summary"])
        handoff = {
            "definition": "Completed B2 100M benchmark handoff; no later weight-changing stage started.",
            "b2_complete": True,
            "cumulative_tokens": int(summary["cumulative_tokens"]),
            "target_cumulative_tokens": int(summary["target_cumulative_tokens"]),
            "checkpoints": {k: {"path": str(state[k]), "sha256": sha256_file(state[k])} for k in ("baseline20", "b1", "b2")},
            "frozen_validation_sha256": validation_hashes,
            "comparison_json": str(comparison_json),
            "long_context_gate": load_json(long_gate) if long_gate.is_file() else None,
            "later_training_started": False,
            "next_decision": "Review frozen B2 reports before B3/RawMath/Instruct.",
        }
        (reports / "B2_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(handoff, ensure_ascii=False), flush=True)
        if gate.returncode != 0:
            raise SystemExit(gate.returncode)
    finally:
        for path in staged_files:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
