"""Frozen Arctus 350M capability evaluator.

Never silently accept duplicate/missing predictions or claim a checkpoint
has not regressed when even one evaluated skill category fell back.
This is a small project diagnostic, not a validated NIS2 benchmark.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(str(text).casefold().strip().split())


def _phrases(spec: dict, key: str, *, required: bool) -> list[str]:
    value = spec.get(key)
    if value is None and not required:
        return []
    if (not isinstance(value, list) or not value
        or not all(isinstance(x, str) and x.strip() for x in value)):
        raise ValueError("invalid nonempty scoring phrase list: " + key)
    return value


def score_one(text: str, spec: dict) -> bool:
    if not isinstance(text, str) or not isinstance(spec, dict):
        raise ValueError("scoring requires a text response and scoring specification")
    t = norm(text)
    typ = spec.get("type")
    if typ == "exact":
        if not isinstance(spec.get("value"), str):
            raise ValueError("exact score needs a string target")
        return text.strip() == spec["value"].strip()
    if typ == "integer":
        value = spec.get("value")
        if type(value) is not int:
            raise ValueError("integer score needs an integer target")
        # Match signed integers even when followed by punctuation. Do not
        # falsely match decimal/thousands substrings in long model answers.
        found = re.findall(r"(?<![\w.,])[+-]?\d+(?![\w]|[.,]\d)", text)
        return bool(found) and int(found[-1]) == value
    if typ == "contains_all":
        return all(norm(x) in t for x in _phrases(spec, "values", required=True))
    if typ == "contains_any":
        values = _phrases(spec, "values", required=True)
        and_values = _phrases(spec, "and_values", required=False)
        return any(norm(x) in t for x in values) and all(norm(x) in t for x in and_values)
    raise ValueError(f"unknown scoring type: {typ}")


def score(report: dict, cases: list[dict]) -> dict:
    if not isinstance(report, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("nonempty case list and report object required")
    cmap = {}
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("benchmark case must be an object")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in cmap:
            raise ValueError("missing or duplicate reference case ID")
        if not isinstance(item.get("category"), str) or not item["category"].strip():
            raise ValueError("case category missing")
        cmap[case_id] = item
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("case/report row count mismatch")
    got = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("invalid report row")
        case_id = row["id"]
        if case_id in got:
            raise ValueError("duplicate scored prediction ID: " + case_id)
        if not isinstance(row.get("text"), str):
            raise ValueError("missing or nontext prediction")
        got[case_id] = row
    if set(got) != set(cmap):
        raise ValueError("case/report ID mismatch")

    cats = collections.defaultdict(list)
    detail = []
    for case_id, case in cmap.items():
        value = got[case_id]["text"]
        ok = score_one(value, case["scoring"])
        cats[case["category"]].append(ok)
        detail.append({"id": case_id, "category": case["category"], "pass": ok, "text": value})
    by = {
        key: {"passed": sum(group), "cases": len(group),
              "pct": 100 * sum(group) / len(group)}
        for key, group in sorted(cats.items())
    }
    total = sum(item["pass"] for item in detail)
    frozen_cases = json.dumps(
        sorted(cases, key=lambda case: case["id"]), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return {
        "suite_sha256": hashlib.sha256(frozen_cases).hexdigest(),
        "cases": len(detail),
        "passed": total,
        "overall_pct": 100 * total / len(detail),
        "by_category": by,
        "details": detail,
    }


def _validated_metrics(report: dict) -> tuple[float, dict[str, float], dict[str, int]]:
    if not isinstance(report, dict) or not isinstance(report.get("by_category"), dict):
        raise ValueError("invalid capability score report")
    cats = report["by_category"]
    if not cats:
        raise ValueError("empty capability categories")
    totals = []
    percents = {}
    sizes = {}
    for name, group in cats.items():
        if not isinstance(name, str) or not name or not isinstance(group, dict):
            raise ValueError("invalid capability category")
        n, passed, pct = group.get("cases"), group.get("passed"), group.get("pct")
        if (type(n) is not int or not 1 <= n <= 10000
            or type(passed) is not int or not 0 <= passed <= n
            or type(pct) not in (int, float) or not math.isfinite(pct)
            or not 0 <= pct <= 100
            or abs(pct - 100 * passed / n) > 1e-6):
            raise ValueError("invalid category counts or non-finite percentage")
        totals.append((n, passed))
        percents[name] = float(pct)
        sizes[name] = n
    overall = report.get("overall_pct")
    if (type(overall) not in (int, float) or not math.isfinite(overall)
        or not 0 <= overall <= 100
        or abs(overall - 100 * sum(p for _, p in totals) / sum(n for n, _ in totals)) > 1e-6):
        raise ValueError("invalid or contradictory overall capability score")
    if ("cases" in report and (type(report["cases"]) is not int
                               or report["cases"] != sum(n for n, _ in totals))):
        raise ValueError("reported case count disagrees with categories")
    if ("passed" in report and (type(report["passed"]) is not int
                                or report["passed"] != sum(p for _, p in totals))):
        raise ValueError("reported passed count disagrees with categories")
    return float(overall), percents, sizes


def compare(base: dict, cand: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(cand, dict):
        raise ValueError("invalid capability comparison report")
    expected = base.get("suite_sha256")
    actual = cand.get("suite_sha256")
    if (not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or actual != expected):
        raise ValueError("cannot compare different or unverified frozen suites")
    base_overall, base_cats, base_sizes = _validated_metrics(base)
    cand_overall, cand_cats, cand_sizes = _validated_metrics(cand)
    if base_cats.keys() != cand_cats.keys() or base_sizes != cand_sizes:
        raise ValueError("incomparable capability category lists or sizes")
    deltas = {name: cand_cats[name] - base_cats[name] for name in sorted(base_cats)}
    # Keep the legacy hard-regression threshold for diagnosing historical
    # reports, but NEVER call a candidate "no regression" if any category
    # went down by even one correct answer.
    hard_reg = [name for name, delta in deltas.items() if delta < -20.0]
    all_categories_ok = all(delta >= -1e-9 for delta in deltas.values())
    overall_ok = cand_overall >= base_overall - 1e-9
    return {
        "overall_delta_pp": cand_overall - base_overall,
        "category_delta_pp": deltas,
        "hard_regressions": hard_reg,
        "no_hard_category_regression": not hard_reg,
        "category_no_regression": all_categories_ok,
        "nonnegative_overall": overall_ok,
        "capability_no_regression": all_categories_ok and overall_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-score")
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text(encoding="utf8"))
    cases = json.loads(Path(a.cases).read_text(encoding="utf8"))
    res = score(report, cases)
    if a.baseline_score:
        base = json.loads(Path(a.baseline_score).read_text(encoding="utf8"))
        res["comparison"] = compare(base, res)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in res.items() if k != "details"}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
