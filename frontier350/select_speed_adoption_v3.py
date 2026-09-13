from __future__ import annotations

"""Fail-closed selector for Neral 350M execution-only speed candidates.

This script never changes a trainer or checkpoint. It summarizes which optimizations are
eligible for a later own-weight canary. Final Scale V3 adoption still requires unchanged
validation/dev/bilingual gates.
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.report)

    pack = load(root / "pack9_selection.json")
    replay = load(root / "replay_fusion.json")
    fused = load(root / "fused_execution.json")
    checkpoint = load(root / "checkpoint_resume.json")

    shas = {
        pack.get("anchor_sha256"),
        replay.get("anchor_sha256"),
        fused.get("anchor_sha256"),
        checkpoint.get("anchor_sha256"),
    }
    tokens = {
        int(pack.get("anchor_tokens", -1)),
        int(replay.get("anchor_tokens", -2)),
        int(fused.get("anchor_tokens", -3)),
        int(checkpoint.get("anchor_tokens", -4)),
    }
    identity_ok = len(shas) == 1 and None not in shas and len(tokens) == 1 and min(tokens) >= 0

    candidates = {
        "pack9": {
            "eligible": bool(pack.get("pack9_execution_candidate")),
            "measured_speedup": pack.get("speedup_pack9_vs_pack6"),
            "reason": "same guarded trainer; geometry only",
        },
        "replay_fusion": {
            "eligible": bool(replay.get("numerical_candidate") and replay.get("speed_candidate")),
            "measured_speedup": replay.get("speedup"),
            "gradient_relative_l2_delta": replay.get("gradient_relative_l2_delta"),
            "loss_abs_delta": replay.get("loss_abs_delta"),
        },
        "fused_execution": {
            "eligible": bool(fused.get("eligible_for_canary")),
            "measured_speedup": fused.get("speedup"),
            "numerical": fused.get("numerical"),
        },
        "compact_resume": {
            "eligible": bool(checkpoint.get("eligible_for_canary")),
            "save_speedup": checkpoint.get("save_speedup"),
            "size_reduction_fraction": checkpoint.get("size_reduction_fraction"),
            "roundtrip": checkpoint.get("roundtrip"),
        },
    }

    if not identity_ok:
        for row in candidates.values():
            row["eligible"] = False
            row["blocked_by"] = "anchor_identity_mismatch"

    eligible = [name for name, row in candidates.items() if row["eligible"]]
    report = {
        "schema": 1,
        "selector": "neral350-speed-adoption-v3",
        "anchor_identity_ok": identity_ok,
        "anchor_sha256": next(iter(shas)) if identity_ok else None,
        "anchor_tokens": next(iter(tokens)) if identity_ok else None,
        "candidates": candidates,
        "eligible_for_guarded_canary": eligible,
        "canary_order": [x for x in ("pack9", "replay_fusion", "compact_resume", "fused_execution") if x in eligible],
        "automatic_production_adoption": False,
        "promotion_allowed": False,
        "quality_gate_unchanged": True,
        "required_next_gate": "same-data own-weight canary followed by full validation/dev/bilingual retention gates before any Scale V3 integration",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
