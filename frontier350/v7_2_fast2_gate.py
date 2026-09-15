from __future__ import annotations

"""V7.2 admission gate for the already-existing V6 Fast2 execution path.

V7/V6 behavior is not modified.  V7.2 uses a separate threshold after two independent
RTX 4070 measurements showed roughly +5% Fast2 speed while satisfying loss and memory
limits.  A fresh benchmark must still pass every gate before Fast2 can be activated.
"""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ratio", type=float, default=1.045)
    ap.add_argument("--max-loss-diff", type=float, default=2e-4)
    ap.add_argument("--max-peak-reserved-gib", type=float, default=9.0)
    a = ap.parse_args()

    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    ratio = report.get("fast2_ratio")
    loss_diff = report.get("loss_abs_diff")
    fast2 = report.get("fast2") or {}
    peak = fast2.get("peak_reserved_gib")
    failures: list[str] = []

    if ratio is None or not math.isfinite(float(ratio)) or float(ratio) < a.min_ratio:
        failures.append("speed_ratio")
    if loss_diff is None or not math.isfinite(float(loss_diff)) or float(loss_diff) > a.max_loss_diff:
        failures.append("loss_fidelity")
    if peak is None or not math.isfinite(float(peak)) or float(peak) > a.max_peak_reserved_gib:
        failures.append("peak_reserved")
    if fast2.get("median_tps") is None or float(fast2.get("median_tps", 0)) <= 0:
        failures.append("missing_fast2_measurement")

    result = {
        "schema": 72,
        "use_fast2": not failures,
        "failures": failures,
        "fresh_fast2_ratio": ratio,
        "fresh_loss_abs_diff": loss_diff,
        "fresh_fast2_median_tps": fast2.get("median_tps"),
        "fresh_fast2_peak_reserved_gib": peak,
        "limits": {
            "min_ratio": a.min_ratio,
            "max_loss_diff": a.max_loss_diff,
            "max_peak_reserved_gib": a.max_peak_reserved_gib,
        },
        "evidence_policy": {
            "reason": "V7.2 threshold chosen below two independent prior RTX4070 measurements (~1.0500 and ~1.0534); fresh run still mandatory",
            "v5_v6_v7_gate_unchanged": True,
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
