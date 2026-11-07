"""NERAL-EXT phase 1: CPU-only, conservative paired holdout scorer.

No model weights, downloads, generated-code execution, training or GPU use.
External/novel task provenance is not independently proven by this tool.
All inputs and model responses remain local; export contains counts only.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

CATEGORIES = frozenset({"integer", "rational", "eur"})
SHAS = re.compile(r"[0-9a-f]{64}\Z")
REV = re.compile(r"[0-9a-f]{40}\Z")
VALUE = re.compile(r"[+-]?\d{1,20}(?:[.,]\d{1,20})?(?:/\d{1,20})?\Z")
EURO = re.compile(r"([+-]?\d{1,20})[.,](\d{2})\sEUR\Z")
FINAL = re.compile(r"(?:FINAL_ANSWER|Answer|Antwort|Ergebnis)\s*:\s*(.+)", re.IGNORECASE)
WORDS = re.compile(r"\w+", re.UNICODE)


def _no_dupes(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON object keys")
        obj[key] = value
    return obj


def _bounded_file(path: Path, max_size: int) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_size:
        raise ValueError("Missing, linked, empty or oversized input")
    return path.read_bytes()


def _records(raw: bytes, max_rows: int) -> list[dict]:
    try:
        text = raw.decode("utf-8")
        lines = text.splitlines()
        if not 1 <= len(lines) <= max_rows or any(not s.strip() for s in lines):
            raise ValueError("Invalid JSONL row count")
        rows = [json.loads(line, object_pairs_hook=_no_dupes) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSON objects")
    return rows


def read_cases(path: Path, expected_sha: str) -> list[dict]:
    if not isinstance(expected_sha, str) or SHAS.fullmatch(expected_sha) is None:
        raise ValueError("Expected pinned case SHA-256")
    raw = _bounded_file(Path(path), 2_000_000)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Holdout SHA-256 mismatch")
    rows = _records(raw, 500)
    used = set()
    for row in rows:
        if set(row) != {"id", "domain", "kind", "prompt", "answer"}:
            raise ValueError("Unexpected holdout fields")
        key, domain, kind, prompt, answer = (row[z] for z in ("id", "domain", "kind", "prompt", "answer"))
        if (not isinstance(key, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) is None
                or key in used or not isinstance(domain, str) or
                re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", domain) is None or
                not isinstance(kind, str) or kind not in CATEGORIES or
                not isinstance(prompt, str) or
                not 1 <= len(prompt.strip()) <= 2000 or not isinstance(answer, str) or
                not 1 <= len(answer) <= 80):
            raise ValueError("Untrusted, duplicate or invalid holdout case")
        used.add(key)
        if _value(kind, answer) is None:
            raise ValueError("Reference answer is not safely gradeable")
    return rows


def read_responses(path: Path) -> dict[str, dict]:
    rows = _records(_bounded_file(Path(path), 3_000_000), 500)
    result = {}
    for row in rows:
        if set(row) != {"id", "text", "complete"}:
            raise ValueError("Unexpected response fields")
        key, response, complete = row["id"], row["text"], row["complete"]
        if (not isinstance(key, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) is None
                or key in result or not isinstance(response, str) or
                len(response) > 4096 or type(complete) is not bool):
            raise ValueError("Invalid, oversized or duplicated model response")
        result[key] = {"text": response, "complete": complete}
    return result


def _value(kind: str, text: str) -> Fraction | None:
    if not isinstance(text, str) or len(text) > 80:
        return None
    if kind == "eur":
        match = EURO.fullmatch(text.strip())
        if match is None:
            return None
        sign = -1 if match[1].startswith("-") else 1
        return Fraction(sign * (abs(int(match[1])) * 100 + int(match[2])), 100)
    if kind not in ("integer", "rational"):
        return None
    cleaned = text.strip()
    if VALUE.fullmatch(cleaned) is None:
        return None
    if kind == "integer" and re.fullmatch(r"[+-]?\d{1,20}", cleaned) is None:
        return None
    try:
        if "/" in cleaned:
            a, b = cleaned.split("/")
            return Fraction(int(a), int(b))
        return Fraction(Decimal(cleaned.replace(",", ".")))
    except (ZeroDivisionError, InvalidOperation, ValueError, OverflowError):
        return None


def semantic_grade(case: dict, response: str, *, complete: bool = True) -> str:
    """correct/incorrect/ungraded. Never guess last number or execute code."""
    if not complete or not isinstance(response, str) or len(response) > 4096:
        return "ungraded"
    raw = response.strip()
    if not raw:
        return "ungraded"
    lines = raw.splitlines()
    # Only explicitly marked final line or a single direct answer is gradeable.
    if len(lines) == 1:
        match = FINAL.fullmatch(lines[0].strip())
        candidate = match[1].strip() if match else lines[0].strip()
    else:
        if sum(bool(FINAL.fullmatch(s.strip())) for s in lines) != 1:
            return "ungraded"
        match = FINAL.fullmatch(lines[-1].strip())
        if match is None:
            return "ungraded"
        candidate = match[1].strip()
    got = _value(case["kind"], candidate)
    expected = _value(case["kind"], case["answer"])
    if got is None or expected is None:
        return "ungraded"
    return "correct" if got == expected else "incorrect"


def evaluate(cases: list[dict], responses: dict[str, dict]) -> dict[str, dict]:
    if set(responses) != {c["id"] for c in cases}:
        raise ValueError("Response IDs must match exact holdout IDs")
    out = {}
    for case in cases:
        entry = responses[case["id"]]
        if set(entry) != {"text", "complete"} or type(entry["complete"]) is not bool:
            raise ValueError("Invalid response")
        raw = entry["text"]
        if not isinstance(raw, str) or len(raw) > 4096:
            raise ValueError("Invalid response text")
        strict = bool(entry["complete"] and raw.strip() == case["answer"])
        out[case["id"]] = {
            "strict": strict,
            "semantic": semantic_grade(case, raw, complete=entry["complete"]),
        }
    return out


def _stats(data: dict, ids: list[str]) -> dict:
    return {
        "total": len(ids),
        "strict_correct": sum(data[i]["strict"] for i in ids),
        "semantic_correct": sum(data[i]["semantic"] == "correct" for i in ids),
        "semantic_incorrect": sum(data[i]["semantic"] == "incorrect" for i in ids),
        "ungraded": sum(data[i]["semantic"] == "ungraded" for i in ids),
    }


def _mcnemar(better: int, worse: int) -> float:
    total = better + worse
    if not total:
        return 1.0
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(min(better, worse) + 1)) / (2 ** total))


def paired_report(cases: list[dict], base: dict, candidate: dict, *,
                  case_hash: str, base_revision: str, adapter_sha: str,
                  training_overlap_checked: bool = False) -> dict:
    if (not isinstance(case_hash, str) or SHAS.fullmatch(case_hash) is None or
            not isinstance(adapter_sha, str) or SHAS.fullmatch(adapter_sha) is None or
            not isinstance(base_revision, str) or REV.fullmatch(base_revision) is None):
        raise ValueError("Pinned SHA/revision required")
    ids = [c["id"] for c in cases]
    if set(base) != set(ids) or set(candidate) != set(ids):
        raise ValueError("Mismatched paired evaluations")
    improved = sum(base[i]["semantic"] != "correct" and candidate[i]["semantic"] == "correct" for i in ids)
    regressed = sum(base[i]["semantic"] == "correct" and candidate[i]["semantic"] != "correct" for i in ids)
    domains = sorted({c["domain"] for c in cases})
    return {
        "schema": 1, "status": "NERAL_EXT_SCORING_ONLY",
        "case_sha256": case_hash, "base_revision": base_revision,
        "adapter_sha256": adapter_sha, "total": len(ids),
        "base": _stats(base, ids), "candidate": _stats(candidate, ids),
        "per_domain": {domain: {
            "base": _stats(base, [c["id"] for c in cases if c["domain"] == domain]),
            "candidate": _stats(candidate, [c["id"] for c in cases if c["domain"] == domain]),
        } for domain in domains},
        "paired": {"improvements": improved, "regressions": regressed,
                   "net_gain": improved - regressed,
                   "exact_mcnemar_p": _mcnemar(improved, regressed)},
        "training_overlap_checked": training_overlap_checked,
        "external_independence_verified": False,
        "general_improvement_proven": False,
        "release_validated": False,
        "trained_adapter_promoted": False,
        "limitations": "Only local saved responses scored; external dataset provenance, hidden training contamination, decoding parity and coding behavior require separate audit. No model inference or safe execution of generated code.",
    }


def ngram_overlap(train_prompts: list[str], holdout_prompts: list[str], *, n: int = 10) -> bool:
    if not 2 <= n <= 20:
        raise ValueError("Unsupported overlap window")
    train_grams = set()
    for prompt in train_prompts:
        tokens = WORDS.findall(prompt.casefold())
        train_grams.update(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
    for prompt in holdout_prompts:
        tokens = WORDS.findall(prompt.casefold())
        if any(tuple(tokens[i:i+n]) in train_grams for i in range(max(0, len(tokens)-n+1))):
            return True
    return False


def _training_prompts(path: Path) -> list[str]:
    rows = _records(_bounded_file(path, 10_000_000), 100_000)
    prompts = []
    for row in rows:
        value = row.get("prompt")
        if isinstance(value, list):
            users = [z.get("content") for z in value if isinstance(z, dict) and z.get("role") == "user"]
            if len(users) != 1:
                raise ValueError("Invalid training role schema")
            value = users[0]
        if not isinstance(value, str):
            raise ValueError("Invalid training prompt")
        prompts.append(value)
    return prompts


def write_report(path: Path, report: dict) -> None:
    path = Path(path)
    if path.is_symlink() or path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Only new regular report path under an existing directory")
    with path.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, separators=(",", ":"), allow_nan=False)
        out.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("cases", "case-sha256", "base-responses", "neral-responses",
                 "base-revision", "adapter-sha", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--train-jsonl")
    a = p.parse_args(argv)
    cases = read_cases(Path(a.cases), a.case_sha256)
    checked = bool(a.train_jsonl)
    if checked and ngram_overlap(_training_prompts(Path(a.train_jsonl)), [c["prompt"] for c in cases]):
        p.error("Ten-word overlap with training: holdout is contaminated")
    base = evaluate(cases, read_responses(Path(a.base_responses)))
    candidate = evaluate(cases, read_responses(Path(a.neral_responses)))
    result = paired_report(cases, base, candidate, case_hash=a.case_sha256,
                           base_revision=a.base_revision, adapter_sha=a.adapter_sha,
                           training_overlap_checked=checked)
    write_report(Path(a.out), result)
    print("NERAL_EXT_SCORING_ONLY: aggregate report written; no GPU, no raw export")


if __name__ == "__main__":
    main()
