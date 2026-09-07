"""Resolve the exact hardened B2 frozen benchmark used as the G1 before-state.

GPU-control reports are persisted below a run directory while the control bridge itself adds
request/run nesting. This helper searches recursively, verifies checkpoint identity and the
complete frozen report set, and returns the newest exact match. It never touches weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED=("free.json","v3.json","v4.json","v5.json","rawmath.json")


def resolve(assets: Path, checkpoint_sha: str) -> dict:
    checkpoint_sha=checkpoint_sha.strip().lower()
    if len(checkpoint_sha)!=64:raise ValueError("checkpoint SHA-256 must be 64 hex characters")
    root=assets.resolve()/"reports"/"gpu_control"
    if not root.is_dir():raise FileNotFoundError(root)
    handoffs=sorted(root.rglob("B2_HANDOFF.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    inspected=[]
    for handoff in handoffs:
        try:obj=json.loads(handoff.read_text(encoding="utf8"))
        except Exception as exc:
            inspected.append({"handoff":str(handoff),"reason":f"invalid_json:{type(exc).__name__}"});continue
        actual=str(obj.get("checkpoints",{}).get("b2",{}).get("sha256","")).lower()
        if actual!=checkpoint_sha:
            inspected.append({"handoff":str(handoff),"reason":"checkpoint_sha_mismatch","checkpoint_sha256":actual});continue
        if not obj.get("b2_complete") or int(obj.get("cumulative_tokens",0))<100_000_000:
            inspected.append({"handoff":str(handoff),"reason":"incomplete_b2_handoff"});continue
        report_dir=handoff.parent/"b2"
        missing=[name for name in REQUIRED if not (report_dir/name).is_file()]
        if missing:
            inspected.append({"handoff":str(handoff),"reason":"missing_reports","missing":missing});continue
        return {
            "resolved":True,"checkpoint_sha256":checkpoint_sha,"handoff":str(handoff.resolve()),
            "benchmark_root":str(handoff.parent.resolve()),"b2_report_dir":str(report_dir.resolve()),
            "required_reports":list(REQUIRED),"inspected_candidates":len(inspected)+1,
            "generation_health_proceed":bool((obj.get("generation_health_gate") or {}).get("proceed")),
            "long_context_proceed":bool((obj.get("long_context_gate") or {}).get("proceed")),
            "weights_modified":False,
        }
    raise FileNotFoundError(f"No hardened B2 benchmark matches checkpoint {checkpoint_sha}; inspected {len(handoffs)} handoffs")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--assets",required=True);ap.add_argument("--checkpoint-sha",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    result=resolve(Path(a.assets),a.checkpoint_sha)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
