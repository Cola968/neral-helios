"""Diagnose saved NERAL v5 A/B responses locally without GPU or new inference.

Reads the 320-case *saved* local Qwen and NERAL v5 responses, verifies original
case IDs, model provenance, scores and raw hashes, then exports only aggregate
counts. No prompts, responses, or per-example scores leave the Windows runner.
This is in-distribution development diagnostics, not independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_v5_style_eval import (
    CASES, CATEGORIES, BASE_REVISION, case_hash, grade,
)

EXPECTED = {case["id"]: case for case in CASES}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FINAL_LINE = re.compile(
    r"\s*(?:FINAL_ANSWER|Antwort|Ergebnis|Endergebnis|Result)\s*[:=]\s*(.+?)\s*\Z",
    re.IGNORECASE,
)
FIELDS = frozenset({
    "id", "correct", "format_valid", "answer_sha256", "local_raw"
})
TOP_LEVEL = frozenset({
    "variant", "adapter_sha256", "base_revision", "case_hash", "scores"
})


def _safe_read(path: Path) -> dict:
    if (path.is_symlink() or not path.is_file() or
            not 0 < path.stat().st_size <= 200_000):
        raise ValueError("Missing, linked or oversized local benchmark file")
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid saved local benchmark JSON") from exc
    if not isinstance(data, dict) or set(data) != TOP_LEVEL:
        raise ValueError("Unexpected saved benchmark fields")
    return data


def scan_saved_rows(root: Path, variant: str, adapter_sha: str) -> dict:
    """Return validated rows indexed by case ID; all response strings stay local."""
    root = Path(root)
    if variant not in ("base", "neral_v5"):
        raise ValueError("Unknown saved model variant")
    folder = root / variant
    if (root.is_symlink() or folder.is_symlink() or
            not folder.is_dir()):
        raise ValueError("Non-regular saved run folder")
    data = _safe_read(folder / "local.json")
    if (data["variant"] != variant or data["adapter_sha256"] != adapter_sha or
            data["base_revision"] != BASE_REVISION or
            data["case_hash"] != case_hash()):
        raise ValueError("Saved benchmark model provenance mismatch")
    scores = data["scores"]
    if not isinstance(scores, list) or len(scores) != len(CASES):
        raise ValueError("Expected complete 320-case saved benchmark")
    verified = {}
    for row in scores:
        if (not isinstance(row, dict) or set(row) != FIELDS or
                not isinstance(row["id"], str) or
                row["id"] not in EXPECTED or row["id"] in verified or
                type(row["correct"]) is not bool or
                type(row["format_valid"]) is not bool or
                not isinstance(row["answer_sha256"], str) or
                SHA256.fullmatch(row["answer_sha256"]) is None or
                not isinstance(row["local_raw"], str) or
                len(row["local_raw"]) > 250):
            raise ValueError("Malformed or duplicate saved example score")
        raw = row["local_raw"]
        digest = hashlib.sha256(raw.encode("utf8")).hexdigest()
        truncated = digest != row["answer_sha256"]
        if truncated and len(raw) != 250:
            raise ValueError("Saved raw text or SHA256 tampered")
        if not truncated and (
            grade(EXPECTED[row["id"]], raw) !=
            (row["correct"], row["format_valid"])
        ):
            raise ValueError("Saved correctness flags differ from scored answer")
        verified[row["id"]] = {**row, "_truncated": truncated}
    if set(verified) != set(EXPECTED):
        raise ValueError("Saved case IDs differ from exact v5 320-case test")
    return verified


def _model_stats(rows: dict) -> tuple[dict, dict]:
    totals = {
        "correct": 0, "format_valid": 0, "format_invalid": 0,
        "format_valid_wrong": 0, "recoverable_last_line": 0,
        "blank": 0, "truncated_raw": 0,
    }
    by_cat = {cat: {**{k: 0 for k in totals}, "total": 0}
              for cat in CATEGORIES}
    for case_id, row in rows.items():
        case = EXPECTED[case_id]
        category = case["category"]
        c = by_cat[category]
        c["total"] += 1
        if row["correct"]:
            totals["correct"] += 1
            c["correct"] += 1
        if row["format_valid"]:
            totals["format_valid"] += 1
            c["format_valid"] += 1
            if not row["correct"]:
                totals["format_valid_wrong"] += 1
                c["format_valid_wrong"] += 1
        else:
            totals["format_invalid"] += 1
            c["format_invalid"] += 1
        if row["_truncated"]:
            totals["truncated_raw"] += 1
            c["truncated_raw"] += 1
            continue
        raw = row["local_raw"]
        if not raw.strip():
            totals["blank"] += 1
            c["blank"] += 1
        # Conservatively recognize only an explicit FINAL_ANSWER/Antwort on
        # the last line; never search for expected digits inside reasoning.
        if not row["format_valid"] and raw.strip():
            last = raw.strip().splitlines()[-1]
            match = FINAL_LINE.fullmatch(last)
            if match and grade(case, match.group(1).strip())[0]:
                totals["recoverable_last_line"] += 1
                c["recoverable_last_line"] += 1
    if totals["format_valid"] + totals["format_invalid"] != len(CASES):
        raise ValueError("Saved scores do not partition into format classes")
    return totals, by_cat


def audit(root: Path, *, adapter_sha: str) -> dict:
    if not isinstance(adapter_sha, str) or SHA256.fullmatch(adapter_sha) is None:
        raise ValueError("Sealed trained adapter SHA-256 required")
    b = scan_saved_rows(root, "base", adapter_sha)
    n = scan_saved_rows(root, "neral_v5", adapter_sha)
    base, cats_b = _model_stats(b)
    neral, cats_n = _model_stats(n)
    gained = sum(not b[key]["correct"] and n[key]["correct"] for key in b)
    lost = sum(b[key]["correct"] and not n[key]["correct"] for key in b)
    category_summary = {
        cat: {"base": cats_b[cat], "neral_v5": cats_n[cat]}
        for cat in CATEGORIES
    }
    return {
        "schema": 1, "status": "NERAL_V5_LOCAL_ERROR_AUDIT",
        "total": len(CASES), "base": base, "neral_v5": neral,
        "per_category": category_summary,
        "paired": {"improvements": gained, "regressions": lost,
                   "net_gain": gained - lost},
        "case_hash": case_hash(), "base_revision": BASE_REVISION,
        "adapter_sha256": adapter_sha,
        "independent_external_benchmark": False,
        "general_improvement_proven": False,
        "release_validated": False,
        "limitations": "Local inspection of existing style-disjoint in-distribution saved A/B responses. Truncated raw cannot be regraded; last-line recovery is conservative. No new model inference.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--saved", required=True, type=Path)
    p.add_argument("--adapter-sha", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    if (args.out.is_symlink() or args.out.exists() or
            not args.out.parent.is_dir() or args.out.parent.is_symlink()):
        p.error("Summary output must be a new regular file inside existing directory")
    result = audit(args.saved, adapter_sha=args.adapter_sha)
    with args.out.open("x", encoding="utf8") as dest:
        json.dump(result, dest, sort_keys=True, ensure_ascii=True,
                  separators=(",", ":"), allow_nan=False)
        dest.write("\n")
    print("NERAL_V5_LOCAL_ERROR_AUDIT: scalar-only report created; zero GPU calls")


if __name__ == "__main__":
    main()
