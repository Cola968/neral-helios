"""Report-only development diagnostics. No torch, tokenizer, model or network IO.

The current eval_chat350 report omits prompts, EOS identity and generation
settings. Never reconstruct them from decoded text, summary averages, the local
tokenizer lock or the current evaluator's source. Partial observations remain
useful, but are NOT a valid comparison of free generation. See README.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

try:
    from .chat350_eval_contract import parse_chat_prefix, trim_hallucinated_user_turn
    from .chat350_generation_metrics import distinct_ngram_fraction, repeated_ngram_fraction
except ImportError:  # Also support python frontier350/chat350_dev_generation_diagnostics.py.
    from chat350_eval_contract import parse_chat_prefix, trim_hallucinated_user_turn
    from chat350_generation_metrics import distinct_ngram_fraction, repeated_ngram_fraction


CONTRACT = "segment_bpe_mask_specials_v1"
IDENTITY_KEYS = ("tokenizer_sha256", "case_sha256", "evaluation_contract")
MISSING_METADATA = (
    "last_user_text_or_versioned_echo_measurements",
    "eos_token_id",
    "per_case_max_new_tokens",
    "generation_settings",
    "development_suite_provenance_and_expected_case_ids",
)
RAW_BASIS = "raw_generated_token_ids_including_stop"
TEXT_BASIS = "stored_decoded_text"
WORDS = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def _normalized(text: str) -> str:
    # NFC preserves case, umlauts, ß, punctuation and compatibility distinctions.
    return " ".join(unicodedata.normalize("NFC", text).split())


def measure_prompt_echo(prefix: str, text: str) -> dict:
    """Pure helper for synthetic dev cases / a future evaluator, not legacy rows.

    Apply the existing scorer's hallucinated-User boundary. Never strip an
    Assistant header or use fuzzy/case-insensitive matching. Prefix includes
    exact/normalized echoes, but cannot turn 'Ja' -> 'Jahr' into an echo.
    """
    if not isinstance(text, str):
        raise ValueError("generation text must be a string")
    latest_user = parse_chat_prefix(prefix)[-1][1]
    answer, _ = trim_hallucinated_user_turn(text)
    user_norm, answer_norm = _normalized(latest_user), _normalized(answer)
    starts = answer_norm.startswith(user_norm)
    suffix = answer_norm[len(user_norm):]
    if starts and suffix and user_norm[-1].isalnum() and suffix[0].isalnum():
        starts = False
    return {
        "exact_echo": answer == latest_user,
        "normalized_echo": answer_norm == user_norm,
        "prefix_echo": starts,
    }


def _finite_json(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _rate(summary: dict, key: str, scale: float = 1.0) -> float:
    value = summary.get(key)
    if (type(value) not in (int, float) or not 0 <= value <= scale
            or (isinstance(value, float) and not math.isfinite(value))):
        raise ValueError("invalid/missing summary metric: " + key)
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _validate_report(report: dict) -> list[dict]:
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    _finite_json(report)
    if "schema_version" in report or report.get("evaluation_contract") != CONTRACT:
        raise ValueError("unsupported report version/evaluation_contract")
    for key in ("checkpoint_sha256", "tokenizer_sha256", "case_sha256"):
        digest = report.get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest == "0" * 64:
            raise ValueError("invalid/missing report identity: " + key)
    rows, summary = report.get("rows"), report.get("summary")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict):
        raise ValueError("nonempty rows and summary are required")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("generation row must be an object")
        for key in ("id", "category"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("missing/invalid row " + key)
        if row["id"] in ids:
            raise ValueError("duplicate case id")
        ids.add(row["id"])
        text, tokens = row.get("text"), row.get("generated_ids")
        if not isinstance(text, str):
            raise ValueError("generation text must be a string")
        try:
            text.encode("utf8", "strict")
        except UnicodeError:
            raise ValueError("invalid UTF-8 generation text") from None
        if trim_hallucinated_user_turn(text)[1]:
            raise ValueError("stored text violates hallucinated-User trimming contract")
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("generated_ids must be nonnegative integer IDs")
        if not tokens and text:
            raise ValueError("nonempty decoded text without any generated tokens")
        if "expected_exact" in row and not isinstance(row["expected_exact"], str):
            raise ValueError("expected_exact must be text")
    n = len(rows)
    if type(summary.get("cases")) is not int or summary["cases"] != n:
        raise ValueError("summary case count does not match rows")
    eos = _rate(summary, "eos_fraction")
    _rate(summary, "mean_repeated_trigram_fraction")
    _rate(summary, "mean_distinct2_fraction")
    if _rate(summary, "utf8_fraction") != 1:
        raise ValueError("summary UTF-8 fraction does not match rows")
    mean_length = _rate(summary, "mean_generated_tokens_before_eos", scale=sum(len(r["generated_ids"]) for r in rows))
    # eval_chat350 stops immediately on EOS; each reported EOS removes exactly
    # one final token from length. This checks arithmetic, NOT EOS identity.
    if not _close(eos * n, round(eos * n)) or not _close(mean_length * n, sum(len(r["generated_ids"]) for r in rows) - eos * n):
        raise ValueError("inconsistent source EOS count/token length summary")
    exact = [r["text"].strip() == r["expected_exact"].strip() for r in rows if "expected_exact" in r]
    if "exact_instruction_pct" not in summary:
        raise ValueError("missing exact instruction summary metric")
    if exact:
        if not _close(_rate(summary, "exact_instruction_pct", 100), 100 * sum(exact) / len(exact)):
            raise ValueError("exact instruction summary does not match rows")
    elif summary["exact_instruction_pct"] is not None:
        raise ValueError("exact instruction summary without targets")
    return rows


def _not_measured(basis: str, reason: str) -> dict:
    return {"status": "not_measured", "basis": basis, "numerator": None,
            "denominator": 0, "rate": None, "reason": reason}


def _fraction(count: int, cases: int) -> dict:
    return {"status": "measured", "basis": TEXT_BASIS,
            "numerator": count, "denominator": cases, "rate": count / cases}


def _moments(values: list, basis: str) -> dict:
    total = math.fsum(values) if any(isinstance(v, float) for v in values) else sum(values)
    return {"sum": total, "mean": total / len(values), "basis": basis}


def _aggregate(rows: list[dict], min_words: int) -> dict:
    # Sorted callers make floating point accumulation independent of input order.
    words = [len(WORDS.findall(unicodedata.normalize("NFC", r["text"]))) for r in rows]
    tokens = [r["generated_ids"] for r in rows]
    n = len(rows)
    value = {
        "cases": n,
        "word_count": _moments(words, TEXT_BASIS),
        "raw_token_count": _moments([len(ids) for ids in tokens], RAW_BASIS),
        "empty": _fraction(sum(not _normalized(r["text"]) for r in rows), n),
        # Empty text is also below the threshold. These are overlapping rates.
        "very_short": _fraction(sum(w < min_words for w in words), n),
        "repeated_trigram": _moments([repeated_ngram_fraction(ids, 3) for ids in tokens], RAW_BASIS),
        "distinct_bigram": _moments([distinct_ngram_fraction(ids, 2) for ids in tokens], RAW_BASIS),
    }
    for metric in ("exact_echo", "normalized_echo", "prefix_echo"):
        value[metric] = _not_measured(TEXT_BASIS, "last_user_text_missing")
    value["eos"] = _not_measured("raw_token_ids_with_verified_eos_identity", "eos_token_id_missing")
    value["max_new_without_eos"] = _not_measured("raw_token_ids_with_verified_eos_identity_and_limit", "eos_token_id_and_max_new_tokens_missing")
    return value


def diagnose_report(report: dict, min_words: int = 3) -> dict:
    """Return validated partial observations, never a passing generation gate."""
    if type(min_words) is not int or min_words <= 0:
        raise ValueError("min_words must be a positive integer")
    rows = sorted(_validate_report(report), key=lambda r: r["id"])
    categories = sorted({r["category"] for r in rows})
    return {
        "diagnostics_schema_version": 1,
        "status": "incomplete_metadata",
        "valid_comparison": False,
        "automatic_promotion": False,
        "min_words": min_words,
        "provenance": {k: report[k] for k in ("checkpoint_sha256", *IDENTITY_KEYS)},
        "provenance_verification": "report_claims_only_no_source_artifacts_opened",
        "text_scope": "stored_scorer_text_may_already_be_trimmed",
        "missing_metadata": list(MISSING_METADATA),
        "overall": _aggregate(rows, min_words),
        "by_category": {cat: _aggregate([r for r in rows if r["category"] == cat], min_words) for cat in categories},
        "reported_eos": {"status": "reported_unverified", "basis": "source_summary",
                         "rate": report["summary"]["eos_fraction"], "numerator": None,
                         "reason": "no_eos_identity_to_recompute_from_raw_tokens"},
    }


def compare_reports(reports: list[dict], min_words: int = 3) -> dict:
    """Check identities/subgroups, then refuse absent generation provenance.

    No checkpoint ranking or aggregate 'improvement' is inferred. Matching
    observed sets alone cannot establish completeness of the original suite.
    """
    if not isinstance(reports, list) or len(reports) < 2:
        raise ValueError("comparison requires at least two reports")
    diagnosed = [diagnose_report(r, min_words) for r in reports]
    first = reports[0]
    assignments = {r["id"]: r["category"] for r in first["rows"]}
    for report in reports[1:]:
        for key in IDENTITY_KEYS:
            if first[key] != report[key]:
                raise ValueError("report identity mismatch: " + key)
        current = {r["id"]: r["category"] for r in report["rows"]}
        if assignments.keys() != current.keys():
            raise ValueError("report case ID sets differ")
        if assignments != current:
            raise ValueError("report per-case categories differ")
    return {
        "diagnostics_schema_version": 1, "status": "not_comparable",
        "valid_comparison": False, "automatic_promotion": False,
        "checked_identity": {
            **{k: first[k] for k in IDENTITY_KEYS},
            "observed_case_ids": sorted(assignments),
            "category_counts": dict(sorted(Counter(assignments.values()).items())),
            "suite_completeness": "unknown_without_expected_case_ids",
            "generation_settings": "unknown",
        },
        "missing_metadata": list(MISSING_METADATA),
        "reports": diagnosed,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite JSON constant")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, action="append", type=Path,
                        help="existing eval_chat350 JSON; repeat for comparison")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(args.out.resolve() == path.resolve() or
           (args.out.exists() and path.exists() and args.out.samefile(path)) for path in args.report):
        parser.error("output must not overwrite an input report")
    try:
        reports = [json.loads(path.read_text(encoding="utf8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant) for path in args.report]
        result = (diagnose_report(reports[0], args.min_words) if len(reports) == 1
                  else compare_reports(reports, args.min_words))
    except (ValueError, OSError, RecursionError):
        # Never reflect the parser's payload/path (possibly a private prompt).
        result = {"diagnostics_schema_version": 1, "status": "invalid_report",
                  "valid_comparison": False, "automatic_promotion": False,
                  "error": "invalid_json_report_or_inconsistent_evaluation_evidence"}
    try:
        args.out.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2,
                                       allow_nan=False) + "\n", encoding="utf8")
    except OSError:
        print('{"status":"output_write_failed","valid_comparison":false}', file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "valid_comparison": False}))
    # Legacy reports cannot satisfy the full diagnostic contract. Exit nonzero
    # even when useful partial observations were successfully written.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
