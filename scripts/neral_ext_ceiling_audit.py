"""Read-only NERAL-EXT numeric measurement audit on LOCAL saved GPU outputs.

No inference, no raw upload, no code execution. Reports separately:
(1) truncated outputs, (2) never functionally scored Python,
(3) completed numeric outputs unparseable by original scorer,
(4) completed numeric parseable results, and (5) diagnostic lenient recovery.

This is an exploratory *regrade* of a previously consumed suite, not a
fresh independent benchmark or an automatic release promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_ext import _value, read_cases, read_responses, semantic_grade, write_report

VARIANTS = ("base", "base_fewshot", "neral_v5")
PARTS = "ABCD"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MARKER = re.compile(
    r"(?:FINAL_ANSWER|FINAL|Final answer|Answer|Antwort|Ergebnis)\s*:\s*(.+)",
    re.IGNORECASE,
)
BOX = re.compile(r"\\boxed\{([^{}\n]{1,80})\}(?:[.!])?\s*\Z")
DIRECT = re.compile(
    r"(?:The answer is|Das Ergebnis ist|The final bill is)\s+(.+)",
    re.IGNORECASE,
)
MAX_SAMPLES = 20

# Deliberately unreliable: diagnostic only, NEVER counted as a true answer.
TAIL_NUMBER = re.compile(r"(?<![\w/])([+-]?\d{1,20}(?:[.,]\d{1,20})?(?:/\d{1,20})?)(?![\w/])")


def unverified_last_number(case: dict, raw: str) -> Fraction | None:
    """Return an untrusted numeric hint; NOT a semantic correctness grade."""
    if (case.get("kind") not in ("integer", "rational", "eur")
            or not isinstance(raw, str) or len(raw) > 4096):
        return None
    matches = TAIL_NUMBER.findall(raw)
    if not matches:
        return None
    # This intentionally ignores currency/units; it is unsafe for grading.
    return _value("rational", matches[-1])



def _candidate_value(case: dict, candidate: str) -> Fraction | None:
    """Exact value with very limited unit/typography normalization."""
    if not isinstance(candidate, str) or len(candidate) > 80:
        return None
    s = candidate.strip().removesuffix(".").strip()
    if s.startswith("**") and s.endswith("**") and len(s) >= 5:
        s = s[2:-2].strip()
    if case["kind"] == "eur":
        s = re.sub(r"\s*€\Z", " EUR", s)
    return _value(case["kind"], s)


def extract_terminal_numeric(case: dict, raw: str) -> Fraction | None:
    """Conservative diagnostic, no unsafe arbitrary 'last number' extraction.

    Only accept unambiguous explicit final line, a single ending boxed value,
    a simple answer sentence, or one direct numeric response. No truncations.
    """
    if (case.get("kind") not in ("integer", "rational", "eur")
            or not isinstance(raw, str) or not raw.strip() or len(raw) > 4096
            or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in raw)):
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or len(lines) > 120:
        return None
    matches = [MARKER.fullmatch(line) for line in lines]
    if any(x is not None for x in matches):
        if sum(x is not None for x in matches) != 1 or matches[-1] is None:
            return None
        return _candidate_value(case, matches[-1].group(1))
    if len(re.findall(r"\\boxed\{", raw)) > 1:
        return None
    boxed = BOX.search(lines[-1])
    if boxed is not None:
        return _candidate_value(case, boxed.group(1))
    if len(lines) == 1:
        direct = DIRECT.fullmatch(lines[0])
        return _candidate_value(case, direct.group(1) if direct else lines[0])
    # No last-number fallback: it can grade a quoted intermediate as the final.
    return None


def _response_map(rows: list[dict], cases: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text", "complete"}:
            raise ValueError("Response schema mismatch")
        if (not isinstance(row["id"], str) or row["id"] in result
                or type(row["complete"]) is not bool
                or not isinstance(row["text"], str) or len(row["text"]) > 4096):
            raise ValueError("Unsafe response entry")
        result[row["id"]] = {"text": row["text"], "complete": row["complete"]}
    if set(result) != {case["id"] for case in cases}:
        raise ValueError("Responses do not match exact dataset IDs")
    return result


def _stats(cases: list[dict], response_map: dict) -> dict:
    out = Counter()
    for case in cases:
        out["total"] += 1
        entry = response_map[case["id"]]
        is_python = case["kind"] == "python"
        if is_python:
            out["python_no_functional_grader"] += 1
            if not entry["complete"]:
                out["incomplete_censored"] += 1
            continue
        out["numeric_total"] += 1
        if not entry["complete"]:
            out["incomplete_censored"] += 1
            out["numeric_incomplete_censored"] += 1
            continue
        out["numeric_complete"] += 1
        old = semantic_grade(case, entry["text"], complete=True)
        if old == "correct":
            out["old_semantic_correct"] += 1
        elif old == "incorrect":
            out["old_semantic_wrong"] += 1
        else:
            out["old_ungraded_completed"] += 1
        parsed = extract_terminal_numeric(case, entry["text"])
        expected = _value(case["kind"], case["answer"])
        if expected is None:
            raise ValueError("Invalid pinned reference answer")
        if parsed is None:
            out["numeric_ungraded_completed"] += 1
            # A last number that coincides with the reference may still be an
            # intermediate, an example, or an unrelated quantity.
            hint = unverified_last_number(case, entry["text"])
            if hint is None:
                out["unverified_no_numeric_tail"] += 1
            elif hint == expected:
                out["unverified_last_number_matches_reference"] += 1
            else:
                out["unverified_last_number_differs"] += 1
        elif parsed == expected:
            out["new_lenient_correct"] += 1
            if old != "correct":
                out["recovered_correct"] += 1
        else:
            out["new_lenient_wrong"] += 1
            if old == "correct":
                raise ValueError("Diagnostic extractor regressed old correct answer")
    for key in ("total", "python_no_functional_grader", "numeric_total",
                "incomplete_censored", "numeric_incomplete_censored",
                "numeric_complete", "old_semantic_correct", "old_semantic_wrong",
                "old_ungraded_completed", "numeric_ungraded_completed",
                "new_lenient_correct", "new_lenient_wrong", "recovered_correct",
                "unverified_last_number_matches_reference",
                "unverified_last_number_differs", "unverified_no_numeric_tail"):
        out.setdefault(key, 0)
    assert out["numeric_complete"] + out["numeric_incomplete_censored"] == out["numeric_total"]
    assert (out["new_lenient_correct"] + out["new_lenient_wrong"] +
            out["numeric_ungraded_completed"] == out["numeric_complete"])
    assert (out["unverified_last_number_matches_reference"] +
            out["unverified_last_number_differs"] +
            out["unverified_no_numeric_tail"] == out["numeric_ungraded_completed"])
    return dict(out)


def _private_samples(cases: list[dict], model: str, data: dict,
                     stratum: str, limit: int) -> list[dict]:
    """Private reviewer file; no prompt, answer or response appears in scalar."""
    indexed = []
    for case in cases:
        response = data[case["id"]]
        parsed = extract_terminal_numeric(case, response["text"]) if response["complete"] else None
        status = ("incomplete_censored" if not response["complete"]
                  else "python_ungraded" if case["kind"] == "python"
                  else "numeric_parseable" if parsed is not None
                  else "numeric_ungraded")
        indexed.append((status, hashlib.sha256(
            (model + "|" + case["id"]).encode("utf-8")).digest(), case, response))
    indexed.sort(key=lambda x: (x[0], x[1]))
    if not indexed:
        return []
    # deterministic round-robin across failure modes for first 20
    groups: dict[str, list] = {}
    for entry in indexed:
        groups.setdefault(entry[0], []).append(entry)
    selected = []
    while len(selected) < min(limit, len(indexed)):
        for category in sorted(groups):
            if groups[category] and len(selected) < min(limit, len(indexed)):
                selected.append(groups[category].pop(0))
    return [{
        "variant":model, "stratum":stratum, "id":case["id"],
        "kind":case["kind"], "status":status,
        "prompt_private":case["prompt"], "reference_answer_private":case["answer"],
        "model_output_private":response["text"],
        "complete":response["complete"],
        "note": ("Incomplete text was erased during original inference; "
                 "cannot recover partial reasoning." if not response["complete"] else ""),
    } for status, _, case, response in selected]


def diagnose(cases: list[dict], meta: list[dict], raw_variants: dict,
             *, review_samples: int = MAX_SAMPLES) -> tuple[dict, list[dict]]:
    if (not isinstance(cases, list) or not cases or len(cases) > 500
            or not isinstance(meta, list) or len(meta) != len(cases)
            or set(raw_variants) != set(VARIANTS)
            or type(review_samples) is not int or not 0 <= review_samples <= MAX_SAMPLES):
        raise ValueError("Untrusted audit input/variant settings")
    ident = {c["id"] for c in cases}
    mapping = {}
    for row in meta:
        if (not isinstance(row, dict) or row.get("id") not in ident
                or row.get("stratum") not in PARTS
                or row["id"] in mapping):
            raise ValueError("Unmatched stratum metadata")
        mapping[row["id"]] = row["stratum"]
    if set(mapping) != ident:
        raise ValueError("Incomplete case metadata")
    variants = {name:_response_map(raw_variants[name],cases) for name in VARIANTS}
    by_stratum = {}
    for part in PARTS:
        subset = [c for c in cases if mapping[c["id"]] == part]
        by_stratum[part] = {name: _stats(subset, variants[name]) for name in VARIANTS}
    by_source = {}
    for source in sorted({c["domain"] for c in cases}):
        subset = [c for c in cases if c["domain"] == source]
        by_source[source] = {name:_stats(subset,variants[name]) for name in VARIANTS}
    report = {
        "schema": 1, "status": "NERAL_EXT_CEILING_CPU_DIAGNOSTIC_ONLY",
        "total": len(cases),
        "by_stratum": by_stratum,
        "by_domain": by_source,
        "overall": {name:_stats(cases,variants[name]) for name in VARIANTS},
        "diagnostic_grade_method": (
            "Conservative terminal answer/boxed/explicit-sentence parser; "
            "does NOT infer arbitrary last numbers; numerical exactness checked."
        ),
        "completed_but_ungraded_separated_from_incomplete": True,
        "last_number_is_unverified_diagnostic_only": True,
        "incomplete_raw_erased_in_original_run": True,
        "python_functional_tests_executed": False,
        "semantically_independent_test_proven": False,
        "independent_external_benchmark": False,
        "general_improvement_proven": False,
        "release_validated": False, "weights_promoted": False,
        "limitations": (
            "Exploratory regrade on CONSUMED 400 suite. Cannot reconstruct 126/12/2 "
            "incomplete raw outputs; no sandbox grading for Python. New parser "
            "does not prove Qwen ability ceiling and cannot be used for release promotion."
        ),
    }
    review = []
    for name in VARIANTS:
        part_a = [c for c in cases if mapping[c["id"]] == "A"]
        review.extend(_private_samples(part_a, name, variants[name], "A", review_samples))
    return report, review


def audit_files(cases: Path, manifest: Path, base: Path, few: Path, neral: Path,
                scalar: Path, private_review: Path, *,
                case_sha256: str, expected_adapter_sha256: str,
                review_samples: int = MAX_SAMPLES) -> dict:
    if (not isinstance(case_sha256, str) or not SHA.fullmatch(case_sha256)
            or not isinstance(expected_adapter_sha256, str)
            or not SHA.fullmatch(expected_adapter_sha256)):
        raise ValueError("Unpinned benchmark identity")
    scalar, private_review = Path(scalar), Path(private_review)
    root = Path(__file__).resolve().parents[1]
    for output in (scalar, private_review):
        if (output.is_symlink() or output.exists()
                or not output.parent.is_dir() or output.parent.is_symlink()
                or output.resolve().is_relative_to(root)):
            raise ValueError("Write reports privately outside GitHub repository, create-only")
    if scalar.resolve() == private_review.resolve():
        raise ValueError("Private review and scalar paths must differ")
    tests = read_cases(Path(cases), case_sha256)
    raw_manifest = Path(manifest).read_bytes()
    if len(raw_manifest) > 3_000_000 or not raw_manifest:
        raise ValueError("Missing or oversized pinned local manifest")
    metadata = json.loads(raw_manifest)
    if (metadata.get("dataset_sha256") != case_sha256
            or not isinstance(metadata.get("cases"),list)):
        raise ValueError("Missing pinned stratum manifest")
    raw = {name:list(read_responses(Path(filename)).values())
           for name,filename in zip(VARIANTS,(base,few,neral))}
    # read_responses maps IDs -> scalar answer records; reconstruct expected
    # normalized rows WITHOUT modifying raw data.
    for name,filename in zip(VARIANTS,(base,few,neral)):
        res = read_responses(Path(filename))
        raw[name]=[{"id":key,**value} for key,value in res.items()]
    result, sample = diagnose(tests,metadata["cases"],raw,
                              review_samples=review_samples)
    result["case_sha256"] = case_sha256
    result["adapter_sha256"] = expected_adapter_sha256
    result["saved_response_sha256"] = {
        name:hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for name,filename in zip(VARIANTS,(base,few,neral))
    }
    result["private_review_count_per_variant_A"] = {
        name:sum(item["variant"] == name for item in sample) for name in VARIANTS
    }
    # Never send raw samples to GitHub; private file is only on local runner.
    with private_review.open("x",encoding="utf8") as fp:
        for row in sample:
            fp.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    write_report(scalar,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for field in ("cases","manifest","base","fewshot","neral","scalar",
                  "private-review","case-sha256","adapter-sha256"):
        p.add_argument("--"+field,required=True)
    args=p.parse_args(argv)
    result=audit_files(
        Path(args.cases),Path(args.manifest),
        Path(args.base),Path(args.fewshot),Path(args.neral),
        Path(args.scalar),Path(args.private_review),
        case_sha256=args.case_sha256,
        expected_adapter_sha256=args.adapter_sha256)
    print(json.dumps({
        "status":result["status"],
        "total":result["total"],
        "private_review_count_per_variant_A":result["private_review_count_per_variant_A"],
        "release_validated":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
