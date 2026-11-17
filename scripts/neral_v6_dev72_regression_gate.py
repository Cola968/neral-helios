"""CPU-only NERAL v6 72-task development regression blocker.

Consumes scalar GitHub evaluation only. Never issues release approval.
The 72 related templates are exploratory, not independent OOD evidence.
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path

SUITE = "4b9a302a4cdb07b1e9ed3601cc1382a200ef4e4048a3afac29e72a4324f902dd"
V5 = "2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"
V6 = "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd"
VARIANTS = ("qwen_fewshot", "neral_v5", "neral_v6")
DOMAINS = ("code_trace", "math", "reasoning")
PAIRS = {
    "v5_vs_qwen": ("neral_v5", "qwen_fewshot"),
    "v6_vs_qwen": ("neral_v6", "qwen_fewshot"),
    "v6_vs_v5": ("neral_v6", "neral_v5"),
}


def _int(value, lower: int, upper: int) -> bool:
    return type(value) is int and lower <= value <= upper


def analyze_dev72(report: dict) -> dict:
    """Validate counts, SHA identities and pair deltas before blocking promotion."""
    from scripts.neral_v6_fresh_dev_eval import report_exposes_locked_holdout
    if report_exposes_locked_holdout(report):
        raise ValueError("Locked holdout material is not an input to this decision")
    if (not isinstance(report, dict)
            or report.get("schema") != 1
            or report.get("status") != "NERAL_V6_NEW_STRUCTURAL_DEV_72_THREE_WAY"
            or report.get("total") != 72 or report.get("task_families") != 12
            or report.get("suite_sha256") != SUITE
            or report.get("v5_adapter_sha256") != V5
            or report.get("v6_adapter_sha256") != V6
            or any(report.get(field) is not False for field in
                   ("release_validated", "weights_promoted",
                    "independent_benchmark_proven"))):
        raise ValueError("Dataset/adapters/result provenance does not match pinned dev72")
    scores = report.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(VARIANTS):
        raise ValueError("Three aligned variants required")
    for name in VARIANTS:
        record = scores[name]
        if not isinstance(record, dict) or record.get("total") != 72:
            raise ValueError("Invalid per-variant total")
        for field in ("correct", "ungraded_completed", "incomplete"):
            if not _int(record.get(field), 0, 72):
                raise ValueError("Invalid per-variant counts")
        if sum(record[field] for field in ("correct","ungraded_completed","incomplete")) > 72:
            raise ValueError("More outcomes than original cases")
        domains = record.get("per_domain")
        if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
            raise ValueError("Incomplete family/domain diagnostics")
        if any(not _int(domains[d], 0, 24) for d in DOMAINS):
            raise ValueError("Unbounded domain correctness")
        if sum(domains.values()) != record["correct"]:
            raise ValueError("Domain and overall correctness disagree")
    paired = report.get("paired")
    if not isinstance(paired, dict) or set(paired) != set(PAIRS):
        raise ValueError("Pairwise comparison missing")
    for label, (left, right) in PAIRS.items():
        pair = paired[label]
        if (not isinstance(pair, dict)
                or not _int(pair.get("improvements"), 0, 72)
                or not _int(pair.get("regressions"), 0, 72)
                or pair["improvements"] + pair["regressions"] > 72
                or pair["improvements"] - pair["regressions"]
                   != scores[left]["correct"] - scores[right]["correct"]):
            raise ValueError("Invalid paired comparison")
    v6, v5, q = (scores[k] for k in ("neral_v6","neral_v5","qwen_fewshot"))
    math_delta = v6["per_domain"]["math"] - v5["per_domain"]["math"]
    ungraded_delta = v6["ungraded_completed"] - v5["ungraded_completed"]
    no_regression = (
        v6["correct"] >= v5["correct"] and
        v6["correct"] >= q["correct"] and
        v6["ungraded_completed"] <= v5["ungraded_completed"] and
        all(v6["per_domain"][d] >= v5["per_domain"][d] for d in DOMAINS)
    )
    return {
        "schema": 1,
        "status": ("NERAL_V6_DEV_NO_REGRESSION_SIGNAL" if no_regression
                   else "NERAL_V6_DEV_REGRESSION_WARNING"),
        "suite_sha256": SUITE,
        "v5_adapter_sha256": V5,
        "v6_adapter_sha256": V6,
        "source_total": 72,
        "qwen_correct": q["correct"],
        "v5_correct": v5["correct"],
        "v6_correct": v6["correct"],
        "v6_minus_v5_pp": round(100*(v6["correct"]-v5["correct"])/72,3),
        "v6_minus_qwen_pp": round(100*(v6["correct"]-q["correct"])/72,3),
        "v6_math_minus_v5": math_delta,
        "v6_ungraded_minus_v5": ungraded_delta,
        "developer_no_regression_gate_passed": no_regression,
        "can_promote_adapter": False,
        "general_improvement_proven": False,
        "release_validated": False,
        "independently_blinded_evaluation": False,
        "reason": (
            "This development suite has 72 correlated tasks and measures "
            "strict verified outputs; ungraded responses are not necessarily "
            "wrong. No change to model weights or Helios deployment is permitted."
        ),
    }


def _reject_duplicate_keys(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("Duplicate JSON field")
        data[key] = value
    return data


def analyze_file(original: Path, target: Path) -> dict:
    """Create-only aggregate decision; never copy private prompts or raw answers."""
    original, target = Path(original), Path(target)
    if (original.is_symlink() or not original.is_file()
            or not 0 < original.stat().st_size <= 100_000
            or target.is_symlink() or target.exists()
            or not target.parent.is_dir() or target.parent.is_symlink()):
        raise ValueError("Missing, linked, oversized or existing report path")
    raw = original.read_bytes()
    try:
        incoming = json.loads(raw.decode("utf8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed scalar report") from exc
    result = analyze_dev72(incoming)
    result["source_report_sha256"] = sha256(raw).hexdigest()
    with target.open("x", encoding="utf8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_file(args.summary, args.out)
    print(json.dumps({
        "status": result["status"],
        "developer_no_regression_gate_passed": result["developer_no_regression_gate_passed"],
        "can_promote_adapter": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
