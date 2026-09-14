from __future__ import annotations

import argparse
import json
from pathlib import Path

from v7_policy import fast2_gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ratio", type=float, default=1.06)
    ap.add_argument("--max-loss-diff", type=float, default=2e-4)
    ap.add_argument("--max-peak-reserved-gib", type=float, default=9.0)
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    allowed, failures = fast2_gate(
        report,
        min_ratio=a.min_ratio,
        max_loss_diff=a.max_loss_diff,
        max_peak_reserved_gib=a.max_peak_reserved_gib,
    )
    result = {
        "schema": 7,
        "use_fast2": allowed,
        "failures": failures,
        "fast2_ratio": report.get("fast2_ratio"),
        "loss_abs_diff": report.get("loss_abs_diff"),
        "fast2_peak_reserved_gib": (report.get("fast2") or {}).get("peak_reserved_gib"),
        "limits": {
            "min_ratio": a.min_ratio,
            "max_loss_diff": a.max_loss_diff,
            "max_peak_reserved_gib": a.max_peak_reserved_gib,
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
