"""Build the frozen-harness report consumed by Neral promotion gates.

The historical NIS inputs remain unchanged. When Targeted-v1 reports are supplied, their
math/code/German/reasoning accuracies are attached as direct diagnostics but are not folded
into NIS 1.0, preserving score comparability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def targeted_metrics(report):
    if not report:
        return {}
    out = {"targeted_overall_pct": 100 * report.get("overall", {}).get("accuracy", 0.0)}
    for domain, row in report.get("domains", {}).items():
        out[f"targeted_{domain}_pct"] = 100 * row.get("accuracy", 0.0)
    return out


def bundle(args, prefix):
    base = metrics(load(getattr(args, f"{prefix}_free")), load(getattr(args, f"{prefix}_v3")), load(getattr(args, f"{prefix}_v4")), load(getattr(args, f"{prefix}_rawmath")))
    targeted_path = getattr(args, f"{prefix}_targeted")
    return base | targeted_metrics(load(targeted_path) if targeted_path else None)


def main():
    ap = argparse.ArgumentParser()
    for prefix in ("baseline20", "candidate"):
        for kind in ("free", "v3", "v4", "rawmath"):
            ap.add_argument(f"--{prefix}-{kind}", required=True)
        ap.add_argument(f"--{prefix}-targeted")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    baseline = bundle(args, "baseline20")
    candidate = bundle(args, "candidate")
    free_b = load(args.baseline20_free)
    free_c = load(args.candidate_free)
    if not free_b.get("suite_sha256") or free_b["suite_sha256"] != free_c.get("suite_sha256"):
        raise ValueError("20M/50M frozen free-suite mismatch")
    targeted_sha = None
    if bool(args.baseline20_targeted) != bool(args.candidate_targeted):
        raise ValueError("Supply both targeted reports or neither")
    if args.baseline20_targeted:
        tb, tc = load(args.baseline20_targeted), load(args.candidate_targeted)
        if not tb.get("suite_sha256") or tb["suite_sha256"] != tc.get("suite_sha256"):
            raise ValueError("20M/50M frozen targeted-suite mismatch")
        targeted_sha = tb["suite_sha256"]
    report = {
        "definition": "Direct identical frozen-harness 20M Preview02 vs final 50M candidate. Tools excluded.",
        "stages": {"baseline20": baseline, "instruct": candidate},
        "nis": {"baseline20": nis_v1(baseline), "instruct": nis_v1(candidate)},
        "free_suite_sha256": free_b["suite_sha256"],
        "targeted_suite_sha256": targeted_sha,
        "targeted_metrics_in_nis": False,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"nis20": report["nis"]["baseline20"]["score"], "nis50": report["nis"]["instruct"]["score"], "targeted_suite_sha256": targeted_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
