"""Prepare one V10 interpolation snapshot without touching CUDA or benchmark data."""
import argparse
import json
from pathlib import Path

from interpolate_recovery_checkpoint import interpolate
from recovery_identity import file_sha

ANCHOR_SHA = "16656c86ff7846a39c3f75d7ddc8a629efe31245f4ed3d6fa58f1867f63b059e"
CANDIDATE_SHA = "72e423eae5026c26eac82a2aed8e9c0b38ad8bcb1c83e1dadd52a34fd6960a2c"
ALPHAS = (0.075, 0.05, 0.025, 0.0125)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--v9-request-id", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--alpha", required=True, type=float)
    args = ap.parse_args()
    if args.alpha not in ALPHAS:
        raise ValueError("alpha outside locked V10 grid")

    assets = Path(args.assets).resolve()
    scratch = Path(args.scratch).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise FileExistsError("alpha scratch is non-empty; refusing overwrite")
    scratch.mkdir(parents=True, exist_ok=True)

    v9_root = assets / "runs" / ("neral350_v9_" + args.v9_request_id)
    anchor = v9_root / "replay_only_a010.pt"
    candidate = v9_root / "confirmation" / "resume.pt"
    if file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("V9 1.006B anchor identity changed")
    if file_sha(candidate) != CANDIDATE_SHA:
        raise ValueError("V9 1.008B confirmation identity changed")

    checkpoint = scratch / "candidate.pt"
    result = interpolate(anchor, candidate, checkpoint, args.alpha)
    result.update(
        schema=1,
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha(checkpoint),
        anchor_sha256=ANCHOR_SHA,
        source_candidate_sha256=CANDIDATE_SHA,
    )
    (scratch / "prepared.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
