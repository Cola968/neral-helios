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
import os
from pathlib import Path
import re
import sys
import tempfile
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


class ReportValidationError(ValueError):
    """A safe, fixed diagnostic message that never interpolates report values."""


def _normalized(text: str) -> str:
    # NFC preserves case, umlauts, ß, punctuation and compatibility distinctions.
    return " ".join(unicodedata.normalize("NFC", text).split())


def _word_continuation(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LNM" or category == "Pc"


def measure_prompt_echo(prefix: str, text: str) -> dict:
    """Pure helper for synthetic dev cases / a future evaluator, not legacy rows.

    Apply the existing scorer's hallucinated-User boundary. Never strip an
    Assistant header or use fuzzy/case-insensitive matching. Prefix includes
    exact/normalized echoes, but cannot turn 'Ja' -> 'Jahr' into an echo.
    """
    if not isinstance(text, str):
        raise ValueError("generation text must be a string")
    latest_user = parse_chat_prefix(prefix)[-1][1]
    try:
        prefix.encode("utf8", "strict")
        text.encode("utf8", "strict")
    except UnicodeError:
        raise ValueError("echo input must be valid UTF-8 text") from None
    answer, _ = trim_hallucinated_user_turn(text)
    user_norm, answer_norm = _normalized(latest_user), _normalized(answer)
    starts = answer_norm.startswith(user_norm)
    suffix = answer_norm[len(user_norm):]
    if starts and suffix and _word_continuation(user_norm[-1]):
        if (_word_continuation(suffix[0]) or
                (suffix[0] in "'’-" and len(suffix) > 1 and _word_continuation(suffix[1]))):
            starts = False
    return {
        "exact_echo": answer == latest_user,
        "normalized_echo": answer_norm == user_norm,
        "prefix_echo": starts,
    }


def _finite_json(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportValidationError("report contains a nonfinite number")
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
        raise ReportValidationError("invalid/missing summary metric: " + key)
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _validate_stop_summary(rows: list[dict], summary: dict) -> None:
    """Reject summaries that no fixed EOS ID could have produced.

    This is an existential consistency check, NOT verification of EOS identity.
    Candidates are never exposed or used for diagnostic metrics. The evaluator
    stops on the first EOS, so a possible EOS may occur only at row ends.
    Grouping ngram deltas by terminal ID avoids rescoring all rows per candidate.
    """
    n = len(rows)
    eos = _rate(summary, "eos_fraction")
    count = round(eos * n)
    lengths = [len(row["generated_ids"]) for row in rows]
    mean_length = _rate(summary, "mean_generated_tokens_before_eos", sum(lengths))
    if (not math.isclose(eos, count / n, rel_tol=0, abs_tol=1e-12)
            or count > sum(length > 0 for length in lengths)
            or not _close(mean_length * n, sum(lengths) - count)):
        raise ReportValidationError("inconsistent source EOS count/token length summary")
    wanted_rep = _rate(summary, "mean_repeated_trigram_fraction")
    wanted_d2 = _rate(summary, "mean_distinct2_fraction")
    occurrences, terminals = Counter(), Counter()
    raw_rep, raw_d2, deltas = [], [], {}
    for row in rows:
        ids = row["generated_ids"]
        rep, d2 = repeated_ngram_fraction(ids, 3), distinct_ngram_fraction(ids, 2)
        raw_rep.append(rep)
        raw_d2.append(d2)
        occurrences.update(ids)
        if ids:
            terminal = ids[-1]
            terminals[terminal] += 1
            rep_delta, d2_delta = deltas.setdefault(terminal, ([], []))
            rep_delta.append(repeated_ngram_fraction(ids[:-1], 3) - rep)
            d2_delta.append(distinct_ngram_fraction(ids[:-1], 2) - d2)
    rep_sum, d2_sum = math.fsum(raw_rep), math.fsum(raw_d2)
    if count == 0:
        consistent = _close(rep_sum / n, wanted_rep) and _close(d2_sum / n, wanted_d2)
    else:
        consistent = any(
            terminals[token] == occurrences[token] == count
            and _close((rep_sum + math.fsum(change[0])) / n, wanted_rep)
            and _close((d2_sum + math.fsum(change[1])) / n, wanted_d2)
            for token, change in deltas.items()
        )
    if not consistent:
        raise ReportValidationError("source EOS/ngram summary cannot match a consistent stop token")


def _validate_report(report: dict) -> list[dict]:
    if not isinstance(report, dict):
        raise ReportValidationError("report must be an object")
    _finite_json(report)
    if "schema_version" in report or report.get("evaluation_contract") != CONTRACT:
        raise ReportValidationError("unsupported report version/evaluation_contract")
    for key in ("checkpoint_sha256", "tokenizer_sha256", "case_sha256"):
        digest = report.get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest == "0" * 64:
            raise ReportValidationError("invalid/missing report identity: " + key)
    rows, summary = report.get("rows"), report.get("summary")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict):
        raise ReportValidationError("nonempty rows and summary are required")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ReportValidationError("generation row must be an object")
        for key in ("id", "category"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ReportValidationError("missing/invalid row " + key)
        if row["id"] in ids:
            raise ReportValidationError("duplicate case id")
        ids.add(row["id"])
        text, tokens = row.get("text"), row.get("generated_ids")
        if not isinstance(text, str):
            raise ReportValidationError("generation text must be a string")
        try:
            text.encode("utf8", "strict")
        except UnicodeError:
            raise ReportValidationError("invalid UTF-8 generation text") from None
        if trim_hallucinated_user_turn(text)[1]:
            raise ReportValidationError("stored text violates hallucinated-User trimming contract")
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ReportValidationError("generated_ids must be nonnegative integer IDs")
        if not tokens and text:
            raise ReportValidationError("nonempty decoded text without any generated tokens")
        if "expected_exact" in row and not isinstance(row["expected_exact"], str):
            raise ReportValidationError("expected_exact must be text")
    n = len(rows)
    if type(summary.get("cases")) is not int or summary["cases"] != n:
        raise ReportValidationError("summary case count does not match rows")
    _validate_stop_summary(rows, summary)
    if _rate(summary, "utf8_fraction") != 1:
        raise ReportValidationError("summary UTF-8 fraction does not match rows")
    exact = [r["text"].strip() == r["expected_exact"].strip() for r in rows if "expected_exact" in r]
    if "exact_instruction_pct" not in summary:
        raise ReportValidationError("missing exact instruction summary metric")
    if exact:
        if not _close(_rate(summary, "exact_instruction_pct", 100), 100 * sum(exact) / len(exact)):
            raise ReportValidationError("exact instruction summary does not match rows")
    elif summary["exact_instruction_pct"] is not None:
        raise ReportValidationError("exact instruction summary without targets")
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
        raise ReportValidationError("min_words must be a positive integer")
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
        raise ReportValidationError("comparison requires at least two reports")
    diagnosed = [diagnose_report(r, min_words) for r in reports]
    first = reports[0]
    assignments = {r["id"]: r["category"] for r in first["rows"]}
    targets = {r["id"]: r["expected_exact"] for r in first["rows"] if "expected_exact" in r}
    for report in reports[1:]:
        for key in IDENTITY_KEYS:
            if first[key] != report[key]:
                raise ReportValidationError("report identity mismatch: " + key)
        current = {r["id"]: r["category"] for r in report["rows"]}
        if assignments.keys() != current.keys():
            raise ReportValidationError("report case ID sets differ")
        if assignments != current:
            raise ReportValidationError("report per-case categories differ")
        if targets != {r["id"]: r["expected_exact"] for r in report["rows"] if "expected_exact" in r}:
            raise ReportValidationError("report per-case expected targets differ")
    return {
        "diagnostics_schema_version": 1, "status": "not_comparable",
        "valid_comparison": False, "automatic_promotion": False,
        "checked_identity": {
            **{k: first[k] for k in IDENTITY_KEYS},
            "observed_case_ids": sorted(assignments),
            "category_counts": dict(sorted(Counter(assignments.values()).items())),
            "expected_targets_consistent": True,
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
            raise ReportValidationError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ReportValidationError("nonfinite JSON constant")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, action="append", type=Path,
                        help="existing eval_chat350 JSON; repeat for comparison")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        aliases_input = any(args.out.resolve() == path.resolve() or
                            (args.out.exists() and path.exists() and args.out.samefile(path))
                            for path in args.report)
    except (OSError, RuntimeError):  # resolve can raise RuntimeError for symlink loops on Python 3.11.
        print('{"status":"input_output_path_error","valid_comparison":false}', file=sys.stderr)
        return 2
    if aliases_input:
        parser.error("output must not overwrite an input report")
    try:
        reports = [json.loads(path.read_text(encoding="utf8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant) for path in args.report]
        result = (diagnose_report(reports[0], args.min_words) if len(reports) == 1
                  else compare_reports(reports, args.min_words))
    except ReportValidationError as exc:
        result = {"diagnostics_schema_version": 1, "status": "invalid_report",
                  "valid_comparison": False, "automatic_promotion": False,
                  "error": "invalid_json_report_or_inconsistent_evaluation_evidence",
                  "reason": str(exc)}
    except (ValueError, OSError, RecursionError):
        # Never reflect the parser's payload/path (possibly a private prompt).
        result = {"diagnostics_schema_version": 1, "status": "invalid_report",
                  "valid_comparison": False, "automatic_promotion": False,
                  "error": "invalid_json_report_or_inconsistent_evaluation_evidence",
                  "reason": "report is unreadable or is not valid finite JSON"}
    temporary = None
    try:
        payload = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
        # Same-directory replacement is atomic; a locked destination/disk-full
        # error must not truncate an existing result. Always honor exit status.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=args.out.parent,
                                         prefix=".chat350-diagnostics-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.out)
    except OSError:
        print('{"status":"output_write_failed","valid_comparison":false}', file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Never mask the original write failure with cleanup errors.
    print(json.dumps({"status": result["status"], "valid_comparison": False}))
    # Legacy reports cannot satisfy the full diagnostic contract. Exit nonzero
    # even when useful partial observations were successfully written.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
