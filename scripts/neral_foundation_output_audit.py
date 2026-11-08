"""Read-only scalar audit of the original Qwen3 vs Qwen3.5 512-token run.

Only strict completed final answers are correct. A response cut at the token
cap may be on the right path or wrong: this cannot be inferred automatically.
Raw answers and manual review remain exclusively on the local Windows PC.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

STATUS = "NERAL_FOUNDATION_512_TRUNCATION_DIAGNOSTIC_ONLY"
SOURCE_RUN = "42499079467"
MODELS = ("qwen3:8b", "qwen3.5:9b")
EXPECTED_COUNTS = {
    "qwen3:8b": {"correct": 29, "wrong_final": 11, "ungraded": 1, "incomplete": 31},
    "qwen3.5:9b": {"correct": 8, "wrong_final": 2, "ungraded": 0, "incomplete": 62},
}
FINAL_LINE = re.compile(r"FINAL_ANSWER: (-?(?:0|[1-9][0-9]{0,12}))\Z")


class AuditError(ValueError):
    """Missing, unsafe or inconsistent private benchmark evidence."""


def _grade(text: str, answer: int, *, complete: bool) -> dict:
    if not complete:
        return {"status": "incomplete", "value": None, "format_valid": False}
    if not isinstance(text, str) or not text.strip() or len(text) > 16384:
        return {"status": "ungraded", "value": None, "format_valid": False}
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or len(lines) > 256 or sum("FINAL_ANSWER:" in line for line in lines) != 1:
        return {"status": "ungraded", "value": None, "format_valid": False}
    match = FINAL_LINE.fullmatch(lines[-1])
    if not match:
        return {"status": "ungraded", "value": None, "format_valid": False}
    value = int(match.group(1))
    return {"status": "correct" if value == answer else "wrong_final",
            "value": value, "format_valid": True}


def audit_rows(rows: list[dict], *, expected_counts: dict | None = EXPECTED_COUNTS):
    """Return public scalar-only diagnostics and a *private* review list."""
    if type(rows) is not list or len(rows) != 144:
        raise AuditError("Expected 72 matched cases for each of two models")
    expected_fields = {
        "id", "domain", "family", "prompt_private", "answer_private",
        "model", "raw_private", "grade", "done_reason"
    }
    seen, source = set(), {}
    counts = {name: Counter() for name in MODELS}
    capped = {name: set() for name in MODELS}
    local_review = []
    for row in rows:
        if type(row) is not dict or set(row) != expected_fields:
            raise AuditError("Unexpected private response schema")
        model, cid = row["model"], row["id"]
        domain, family = row["domain"], row["family"]
        prompt, answer = row["prompt_private"], row["answer_private"]
        raw, reason = row["raw_private"], row["done_reason"]
        if (model not in MODELS or type(cid) is not str or not 1 <= len(cid) <= 120
                or (model, cid) in seen or domain not in ("math", "reasoning", "code_trace")
                or type(family) is not str or not 1 <= len(family) <= 80
                or type(prompt) is not str or not 1 <= len(prompt) <= 1800
                or type(answer) is not int or not -2_000_000 < answer < 2_000_000
                or type(raw) is not str or len(raw) > 16384
                or reason not in ("stop", "length")):
            raise AuditError("Unsafe, malformed, or duplicate private case")
        key = (domain, family, prompt, answer)
        if cid in source and source[cid] != key:
            raise AuditError("Models were given nonidentical task data")
        source[cid] = key
        seen.add((model, cid))
        grade = _grade(raw, answer, complete=(reason == "stop"))
        if row["grade"] != grade:
            raise AuditError("Stored grade disagrees with independently parsed final answer")
        counts[model][grade["status"]] += 1
        if reason == "length":
            capped[model].add(cid)
            counts[model]["token_cap_incomplete"] += 1
            # This is NOT a completed correct response, even when a valid
            # final marker is visible before the length stop.
            provisional = _grade(raw, answer, complete=True)
            if provisional["status"] == "correct":
                counts[model]["visible_matching_final_but_incomplete"] += 1
            elif provisional["status"] == "wrong_final":
                counts[model]["visible_wrong_final_but_incomplete"] += 1
            else:
                counts[model]["no_valid_terminal_answer_before_cap"] += 1
            local_review.append({
                "case_id": cid, "model": model,
                "prompt_private": prompt, "gold_private": answer,
                "last_800_chars_private": raw[-800:],
                "requires_manual_reasoning_review": True
            })

    if len(source) != 72 or any(sum(m == model for m, _ in seen) != 72
                                for model in MODELS):
        raise AuditError("Missing or duplicated paired tasks")
    if (any(sum(1 for domain, _, _, _ in source.values() if domain == d) != 24
            for d in ("math", "reasoning", "code_trace"))
            or len({(domain, family) for domain, family, _, _ in source.values()}) != 12):
        raise AuditError("Wrong benchmark domain and family balance")

    if expected_counts is not None:
        actual = {
            model: {key: counts[model][key] for key in
                    ("correct", "wrong_final", "ungraded", "incomplete")}
            for model in MODELS
        }
        if actual != expected_counts:
            raise AuditError("Private run does not match published authenticated results")

    a, b = (capped[m] for m in MODELS)
    pairs = {
        "both_capped": len(a & b),
        "only_qwen3_capped": len(a - b),
        "only_qwen35_capped": len(b - a),
        "neither_capped": 72 - len(a | b)
    }
    models = {}
    for model in MODELS:
        c = counts[model]
        models[model] = {
            "strict_correct": c["correct"],
            "strict_wrong_final": c["wrong_final"],
            "strict_ungraded": c["ungraded"],
            "token_cap_incomplete": c["token_cap_incomplete"],
            "visible_matching_final_but_incomplete": c["visible_matching_final_but_incomplete"],
            "visible_wrong_final_but_incomplete": c["visible_wrong_final_but_incomplete"],
            "no_valid_terminal_answer_before_cap": c["no_valid_terminal_answer_before_cap"]
        }
    report = {
        "schema": 1, "status": STATUS, "source_run_id": SOURCE_RUN,
        "source_generation_budget": 512, "cases_per_model": 72,
        "models": models, "pairs": pairs,
        "cannot_infer_correct_reasoning_from_truncation": True,
        "raw_responses_uploaded": False, "private_review_uploaded": False,
        "no_model_calls_or_training": True, "model_superiority_proven": False,
        "release_validated": False, "adapter_promoted": False,
        "limitations": (
            "A length stop proves no correctness of preceding reasoning. "
            "Correctness after a longer generation requires a separate fair rerun. "
            "Complete-only success rates are selection-biased."
        )
    }
    return report, local_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.run_dir.name != f"v7-foundation-local-{SOURCE_RUN}-1":
        raise AuditError("Only the pinned original run is permitted")
    src = args.run_dir / "private-responses.jsonl"
    if (src.is_symlink() or not src.is_file()
            or not 1 <= src.stat().st_size <= 4_000_000):
        raise AuditError("Missing or unsafe private source")
    for out in (args.aggregate_out, args.private_out):
        if out.exists() or out.is_symlink() or not out.parent.is_dir():
            raise AuditError("Both audit reports must use fresh, separate files")
    if args.aggregate_out.resolve() == args.private_out.resolve():
        raise AuditError("Private and public destinations must differ")
    try:
        rows = [json.loads(line) for line in
                src.read_text(encoding="utf8").splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise AuditError("Corrupt original local log") from exc
    report, local_review = audit_rows(rows)

    # The review file lives OUTSIDE Git checkout; never upload it.
    with args.private_out.open("x", encoding="utf8") as out:
        for row in local_review:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.aggregate_out.open("x", encoding="utf8") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({"status": report["status"], "models": report["models"],
                      "pairs": report["pairs"], "raw_responses_uploaded": False},
                     sort_keys=True))


if __name__ == "__main__":
    main()
