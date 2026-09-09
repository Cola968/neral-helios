from __future__ import annotations

"""Orchestrate a gated multi-window Neral-50M continuation campaign.

Each stage calls the already-audited isolated mobile trainer with a fresh optimizer and a
different immutable data window.  The campaign is intentionally an alternative B2 descendant:
it never touches I1, never promotes itself, and stops immediately when the new-window gain or
old-B2 retention gate fails.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_stage(*, trainer: Path, init_from: Path, data: Path, old_data: Path,
              out: Path, tokens: int, seq: int, batch: int) -> dict:
    cmd = [
        sys.executable, "-u", str(trainer),
        "--init-from", str(init_from),
        "--data", str(data),
        "--old-data", str(old_data),
        "--out", str(out),
        "--tokens", str(tokens),
        "--seq", str(seq),
        "--batch", str(batch),
    ]
    print(json.dumps({"launch": cmd}), flush=True)
    subprocess.run(cmd, check=True)
    report = json.loads((out / "report.json").read_text(encoding="utf8"))
    if not (out / "candidate_local_only.pt").is_file():
        raise RuntimeError(f"stage checkpoint missing: {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--old-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--tokens", action="append", type=int, required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-cumulative-old-regression", type=float, default=0.03)
    args = ap.parse_args()
    if len(args.data) != len(args.tokens):
        raise ValueError("--data and --tokens counts must match")
    if not 1 <= len(args.data) <= 8:
        raise ValueError("campaign supports 1..8 gated stages")

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    trainer = Path(__file__).with_name("train_mobile_data_pilot50.py")
    initial = Path(args.init_from)
    old_data = Path(args.old_data)
    current = initial
    first_old_macro = None
    stages: list[dict] = []
    stop_reason = None

    for index, (data_s, budget) in enumerate(zip(args.data, args.tokens), 1):
        stage_out = root / f"stage_{index:02d}"
        report = run_stage(
            trainer=trainer,
            init_from=current,
            data=Path(data_s),
            old_data=old_data,
            out=stage_out,
            tokens=int(budget),
            seq=args.seq,
            batch=args.batch,
        )
        if first_old_macro is None:
            first_old_macro = float(report["baseline_old_b2"]["macro"])
        final_old = float(report["final_old_b2"]["macro"])
        cumulative_old_regression = (final_old - first_old_macro) / max(1e-9, first_old_macro)
        stage = {
            "index": index,
            "data": str(Path(data_s).resolve()),
            "budget_tokens": int(budget),
            "processed_tokens": int(report["processed_tokens"]),
            "tokens_per_second": float(report["tokens_per_second"]),
            "mobile_ce_improvement": float(report["mobile_ce_improvement"]),
            "stage_old_relative_regression": float(report["old_b2_relative_regression"]),
            "cumulative_old_b2_regression": cumulative_old_regression,
            "stage_eligible": bool(report["eligible_for_frozen_eval"]),
            "checkpoint": str((stage_out / "candidate_local_only.pt").resolve()),
        }
        stages.append(stage)
        print(json.dumps(stage, indent=2), flush=True)

        if not stage["stage_eligible"]:
            stop_reason = f"stage_{index}_local_gate_failed"
            break
        if cumulative_old_regression > args.max_cumulative_old_regression:
            stop_reason = f"stage_{index}_cumulative_old_b2_regression"
            break
        current = stage_out / "candidate_local_only.pt"

    completed_all = len(stages) == len(args.data) and stop_reason is None
    total_processed = sum(s["processed_tokens"] for s in stages)
    campaign = {
        "stage": "Neral-50M Mobile Progressive Scale Campaign",
        "candidate_only": True,
        "promotion": False,
        "quality_claim": False,
        "initial_checkpoint": str(initial.resolve()),
        "old_validation_root": str(old_data.resolve()),
        "requested_stages": len(args.data),
        "completed_stages": len(stages),
        "completed_all_stages": completed_all,
        "stop_reason": stop_reason,
        "total_processed_tokens": total_processed,
        "max_cumulative_old_regression": args.max_cumulative_old_regression,
        "final_candidate": str(current.resolve()) if completed_all else None,
        "requires_frozen_nis2_generation_math_code_eval": True,
        "stages": stages,
    }
    (root / "campaign_report.json").write_text(
        json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )
    print(json.dumps(campaign, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
