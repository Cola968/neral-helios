"""Choose the M1 source checkpoint from completed local evidence only.

Policy: M1 may start from verified I2 E2 only when the hard E2 gate passed and the local
checkpoint hash matches both the E2 result and training summary. Otherwise it falls back to
the completed I1 instruction checkpoint. The completed C3-vs-D3 ablation is required as an
evidence barrier, but C3 is diagnostic only and is never auto-promoted as the chat-model source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8-sig"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_e2_report(reports: Path) -> tuple[Path | None, dict | None]:
    candidates: list[tuple[int, Path, dict]] = []
    for root in reports.glob("runner_50m_i2_*"):
        try:
            run_id = int(root.name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        result_path = root / "I2_RESULT.json"
        if not result_path.is_file():
            continue
        try:
            result = load(result_path)
        except Exception:
            continue
        if "E2" not in str(result.get("stage", "")):
            continue
        candidates.append((run_id, root, result))
    if not candidates:
        return None, None
    _, root, result = max(candidates, key=lambda item: item[0])
    return root, result


def choose(assets: Path, ablation_run_id: int) -> dict:
    assets = assets.resolve()
    reports = assets / "reports"
    ablation_path = reports / f"stage_ablation_{ablation_run_id}" / "STAGE_ABLATION.json"
    if not ablation_path.is_file():
        raise FileNotFoundError(f"completed ablation evidence missing: {ablation_path}")
    ablation = load(ablation_path)
    diagnosis = dict(ablation.get("diagnosis", {}))

    i1_rel = Path("runs/neral50m_i1_instruct_d3/best_instruct.pt")
    i1 = assets / i1_rel
    i1_summary_path = assets / "runs/neral50m_i1_instruct_d3/summary.json"
    if not i1.is_file() or not i1_summary_path.is_file():
        raise FileNotFoundError("completed I1 instruction source is missing")
    i1_summary = load(i1_summary_path)
    if not bool(i1_summary.get("complete")):
        raise ValueError("I1 instruction source is not complete")
    i1_sha = sha256(i1)
    expected_i1_sha = str(i1_summary.get("best_sha256", "")).lower()
    if expected_i1_sha and i1_sha != expected_i1_sha:
        raise ValueError("I1 best checkpoint SHA mismatch")

    selected = {
        "source_label": "i1_instruct",
        "checkpoint_rel": i1_rel.as_posix(),
        "sha256": i1_sha,
        "reason": "fallback_to_verified_i1",
    }
    e2_state = {"found": False, "eligible": False, "report_run_id": None, "reason": "no_completed_e2_report"}

    e2_root, e2_result = latest_e2_report(reports)
    if e2_root is not None and e2_result is not None:
        e2_state["found"] = True
        try:
            e2_state["report_run_id"] = int(e2_root.name.rsplit("_", 1)[1])
        except Exception:
            pass
        gate_pass = e2_result.get("weakness_gate_pass") is True
        effective = e2_result.get("effective_weight_candidate") is True
        e2_rel = Path("runs/neral50m_i2_weakness_e2/best_repair.pt")
        e2 = assets / e2_rel
        e2_summary_path = assets / "runs/neral50m_i2_weakness_e2/summary.json"
        if gate_pass and effective and e2.is_file() and e2_summary_path.is_file():
            e2_summary = load(e2_summary_path)
            e2_sha = sha256(e2)
            result_sha = str(e2_result.get("candidate_sha", "")).lower()
            summary_sha = str(e2_summary.get("best_sha256", "")).lower()
            summary_ok = bool(e2_summary.get("complete")) and bool(e2_summary.get("effective_weight_candidate"))
            hashes_ok = bool(result_sha and summary_sha and e2_sha == result_sha == summary_sha)
            if summary_ok and hashes_ok:
                selected = {
                    "source_label": "i2_e2",
                    "checkpoint_rel": e2_rel.as_posix(),
                    "sha256": e2_sha,
                    "reason": "verified_e2_hard_gate_pass",
                }
                e2_state.update(eligible=True, reason="gate_effective_summary_and_hashes_verified")
            else:
                e2_state["reason"] = "e2_summary_or_hash_verification_failed"
        elif gate_pass and effective:
            e2_state["reason"] = "e2_checkpoint_or_summary_missing"
        else:
            e2_state["reason"] = "e2_gate_failed_or_no_effective_candidate"

    return {
        "policy_version": 1,
        "ablation_run_id": ablation_run_id,
        "ablation_required": True,
        "ablation_diagnosis": diagnosis,
        "selected": selected,
        "e2": e2_state,
        "c3_auto_promotion": False,
        "note": "C3 is diagnostic only. Automatic M1 source is verified E2 or verified I1.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--ablation-run-id", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = choose(Path(args.assets), args.ablation_run_id)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
