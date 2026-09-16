from __future__ import annotations

"""Conservative, read-only gate for stable accepted tokens per wall-clock second.

Measures training + checkpoint + screening + audit time, NOT CUDA step peaks.
Passing this gate proves only stage continuity, not full NIS2/chat/SAFE quality.
"""

import argparse
import json
import math
from pathlib import Path

STAGE_TOKENS = 5_000_000


def evaluate(history: list[dict], *, target_tps: float = 40_000.0, stages: int = 3) -> dict:
    if target_tps <= 0 or not math.isfinite(target_tps) or stages < 3:
        raise ValueError("invalid target or required stage count")
    if not isinstance(history, list):
        raise ValueError("progress must be a JSON list")
    rows: list[dict] = []
    reasons: list[str] = []
    previous_end = -1.0
    previous_tokens: int | None = None
    for index, event in enumerate(history):
        if not isinstance(event, dict):
            raise ValueError("progress event must be an object")
        status = event.get("status")
        if status not in ("CONTINUITY_COMMIT", "STAGE_REJECTED_NO_REWIND"):
            raise ValueError(f"unknown progress event at index {index}: {status!r}")
        start = event.get("stage_start_elapsed_seconds")
        end = event.get("stage_end_elapsed_seconds")
        if start is None or end is None:
            return {"passed": False, "status": "INSUFFICIENT_EVIDENCE",
                    "reason": "missing stage wall-clock timestamps (pre-V8.2 progress cannot certify 40k)",
                    "target_accepted_tps": target_tps, "required_stages": stages,
                    "quality_scope": "continuity health only; full model quality unverified"}
        start, end = float(start), float(end)
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start or start < previous_end:
            raise ValueError(f"nonmonotonic stage times at index {index}")
        before = int(event.get("tokens_before", event.get("continuity_tokens", -1)))
        after = int(event.get("tokens_after", event.get("continuity_tokens", -1)))
        if before < 0 or after < before or (previous_tokens is not None and before != previous_tokens):
            raise ValueError(f"nonmonotonic or disconnected accepted-token accounting at index {index}")
        accepted = after - before
        if status == "CONTINUITY_COMMIT":
            if accepted <= 0 or int(event.get("stage_tokens", -1)) != accepted:
                raise ValueError(f"invalid accepted stage tokens at index {index}")
        elif accepted != 0:
            raise ValueError(f"rejected stage increased accepted tokens at index {index}")
        audit = event.get("audit")
        audit_status = audit.get("status") if isinstance(audit, dict) else None
        healthy = (event.get("health") or {}).get("healthy") is True if status == "CONTINUITY_COMMIT" else False
        rows.append({"round": int(event.get("round", index + 1)), "status": status,
                     "start": start, "end": end, "before": before, "after": after,
                     "accepted": accepted, "seconds": end - start,
                     "accepted_tps": accepted / (end - start),
                     "health_passed": healthy, "audit_status": audit_status})
        previous_end, previous_tokens = end, after

    window = rows[-stages:]
    if len(window) < stages:
        reasons.append(f"need {stages} timed stages, got {len(window)}")
    if any(r["status"] != "CONTINUITY_COMMIT" for r in window):
        reasons.append("recent window contains a rejected stage")
    if any(r["accepted"] != STAGE_TOKENS for r in window):
        reasons.append("recent window contains partial salvage or an incomplete stage")
    if any(not r["health_passed"] for r in window):
        reasons.append("missing or failing continuity health evidence")
    if any(r["audit_status"] not in (None, "SAFE_PROMOTION") for r in window):
        reasons.append("recent full SAFE audit not promoted")
    rates = [r["accepted_tps"] for r in window]
    if len(window) == stages and any(rate < target_tps for rate in rates):
        reasons.append("at least one stage below target accepted tokens/s")
    overall = None
    if len(window) == stages:
        elapsed = window[-1]["end"] - window[0]["start"]
        overall = (window[-1]["after"] - window[0]["before"]) / elapsed
        if overall < target_tps:
            reasons.append("end-to-end window throughput below target")
    passed = not reasons
    return {"passed": passed, "status": "PASSED_CONTINUITY_ONLY" if passed else "NOT_MET",
            "reasons": reasons, "target_accepted_tps": target_tps,
            "required_stages": stages, "measured_stage_tps": rates,
            "measured_end_to_end_accepted_tps": overall,
            "window_rounds": [r["round"] for r in window],
            "quality_scope": "continuity health only; full NIS2/chat/math/code quality and SAFE release remain separately unverified"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-tps", type=float, default=40_000.0)
    parser.add_argument("--stages", type=int, default=3)
    a = parser.parse_args()
    report = evaluate(json.loads(a.progress.read_text(encoding="utf-8-sig")),
                      target_tps=a.target_tps, stages=a.stages)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
