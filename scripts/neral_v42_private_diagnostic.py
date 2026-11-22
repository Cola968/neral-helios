"""Privacy-preserving metadata-only audit of the one existing 5+5 run.

No new inference, no private test prompts, expected answers, result text,
hashes, file paths, user messages or source URLs are printed or uploaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from collections import Counter
import ast
from scripts.neral_v41_blind200_live import _exact_text, _code_only

ORIGIN = 45008615332
_ALLOWED_GRADES = {"pass", "fail", "ungraded"}
_ALLOWED_REASONS = {
    "http_0", "http_200", "http_409", "http_503", "http_504",
    "chain_setup_failed", "runtime_error", "task_timeout",
    "worker_failed", "benchmark_exception", "empty_response",
    "needs_independent_human_review", "no_required_function",
    "private_docker_tests_fail", "private_docker_tests_pass",
    "no_exact_fresh_independent_source", "headline_mismatch",
    "web_number_mismatch", "topic_contamination", "prior_topic_leak",
    "string_mismatch", "exact_mismatch", "independent_exact",
    "exact_string", "unsupported_web_reference", "unsupported_grader",
    "docker_not_available", "docker_unavailable_or_timeout",
    "independent_rss_headline", "independent_api_snapshot",
}
_CATEGORIES = ("everyday", "multiturn", "live_web", "math_logic", "programming")


def safe_enum(value, allow):
    return value if type(value) is str and value in allow else "other"


def classify_exact_format(row, case):
    """Public enum only: parsing vs wrong answer, never expose either value."""
    ref = case["sealed_reference"]
    kind = ref["kind"]
    value = row.get("model_text") or ""
    if kind in ("integer", "web_integer", "yesno"):
        extracted = _exact_text(value, kind)
        if extracted is None:
            return "unparseable"
        expected = ref["value"].casefold() if kind == "yesno" else ref["value"]
        return "parsed_wrong" if extracted != expected else "parsed_correct"
    if kind == "string":
        actual = value.strip().rstrip(".! ")
        return "parsed_correct" if actual.casefold() == ref["value"].casefold() else "string_wrong"
    return "other_kind"


def code_syntax_class(row):
    candidate = _code_only(row.get("model_text") or "")
    try:
        ast.parse(candidate)
        return "syntax_valid"
    except SyntaxError:
        return "syntax_invalid"


def safe_checker_trace_class(row):
    """Classify old failure message locally; never return its raw content."""
    if row.get("error_code") != "checker_veto":
        return "not_applicable"
    text = row.get("model_text") or ""
    if not isinstance(text, str):
        return "unavailable"
    marker = "Ich konnte die Aufgabe nicht vollständig absichern. "
    if text.startswith(marker):
        from neral_v42_quality import checker_veto_class
        return checker_veto_class(
            text[len(marker):],
            executed=row.get("tool_used"))
    return "unavailable_from_legacy_checkpoint"


def audit(root: Path):
    sealed_path = (Path(os.environ["LOCALAPPDATA"]) / "NERAL-private" /
                   "artifacts" / "blind200-partial-run44935727057" / "sealed200.jsonl")
    if not sealed_path.is_file():
        raise RuntimeError("Original private sealed suite missing")
    cases = {case["id"]: case for case in (
        json.loads(line) for line in sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip())}

    for tier in ("Quartz", "Obsidian"):
        path = root / tier / "private-details.jsonl"
        if not path.is_file():
            raise RuntimeError("Expected sealed private metadata missing; refuse regenerated cases")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if type(row) is not dict or row.get("tier") != tier:
                raise RuntimeError("Sealed metadata tier mismatch")
            rows.append(row)
        if len(rows) != 5:
            raise RuntimeError("This metadata audit is authorized for exactly 5 results per tier")
        seen = set()
        for row in rows:
            category = row.get("category")
            if category not in _CATEGORIES or category in seen:
                raise RuntimeError("Unexpected original run category")
            seen.add(category)
            # Note: model_text is private. Only length / format flags survive.
            value = row.get("model_text")
            length = len(value) if type(value) is str else 0
            code_fence = bool(re.search(r"^\s*[\x60]{3}", value)) if type(value) is str else False
            json_start = value.lstrip().startswith("{") if type(value) is str else False
            # Error codes are an internal fixed vocabulary; never echo raw arbitrary text.
            raw_error = row.get("error")
            error = safe_enum(raw_error, _ALLOWED_REASONS) if raw_error else "none"
            reason = safe_enum(row.get("reason"), _ALLOWED_REASONS)
            case = cases.get(row.get("id"))
            if case is None or case.get("category") != category:
                raise RuntimeError("Local sealed task identity mismatch")
            exact_class = classify_exact_format(row, case) if reason == "exact_mismatch" else "not_applicable"
            code_class = code_syntax_class(row) if reason == "private_docker_tests_fail" else "not_applicable"
            checker_trace = safe_checker_trace_class(row)
            error_code = safe_enum(row.get("error_code"), {
                "runtime_error", "request_deadline", "hf_time_budget", "hf_token_budget",
                "hf_invalid_final", "hf_reasoning_unclosed", "checker_veto",
                "no_tool_called", "format_mismatch", "z3_schema_error",
                "sandbox_error", "runtime_unavailable", "context_exceeded",
                "connection_or_timeout"
            }) if row.get("error_code") else "none"
            grade = safe_enum(row.get("grade"), _ALLOWED_GRADES)
            print(f"PRIVACY_SAFE_RESULT tier={tier} category={category} "
                  f"grade={grade} reason={reason} error={error} "
                  f"error_code={error_code} chars={length} code_fence={int(code_fence)} "
                  f"json_start={int(json_start)} exact_parse={exact_class} code_syntax={code_class} "
                  f"checker_issue_class={checker_trace} "
                  f"generated={int(row.get('model_generated') is True)} "
                  f"adapter={int(row.get('adapter_loaded') is True)} "
                  f"repair_attempts={min(max(int(row.get('repair_attempts') or 0), 0), 9)} "
                  f"latency_ms={int(row.get('latency_ms') or 0)}")
        if seen != set(_CATEGORIES):
            raise RuntimeError("Incomplete sealed five-category metadata")
        print(f"PRIVACY_SAFE_AUDIT_COMPLETE tier={tier} count={len(rows)}")


def main():
    root = Path(os.environ["LOCALAPPDATA"]) / "NERAL-private" / "v42-smoke5" / str(ORIGIN)
    audit(root)


if __name__ == "__main__":
    main()
