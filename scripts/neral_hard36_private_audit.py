"""Read-only, aggregate-only audit of a previous PRIVATE hard36 run.

Runs on the owner's Windows runner. Never emits raw prompts, model responses,
private paths, tokens or individual case identifiers to GitHub logs.
No Ollama inference, network calls, Docker execution or GPU allocations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from scripts.neral_same_base_ab_score import ARMS, _correct

_MARKED = re.compile(r"^FINAL(?:_ANSWER)?\s*:\s*([+-]?\d{1,15})\s*$", re.I)
_INTEGER = re.compile(r"^[+-]?\d{1,15}$")


def _numeric_candidate(text: str) -> int | None:
    """Diagnostic only: never relax the production grader implicitly."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = [m.group(1) for line in lines
               if (m := _MARKED.fullmatch(line)) is not None]
    if len(markers) == 1:
        return int(markers[0])
    if not markers and lines and _INTEGER.fullmatch(lines[-1]):
        return int(lines[-1])
    return None


def _outcome(case: dict, obs: dict) -> str:
    if not obs["complete"]:
        return "failed_inference"
    if _correct(case["answer"], obs["text"], True):
        return "strict_correct"
    candidate = _numeric_candidate(obs["text"])
    if candidate is None:
        return "no_unambiguous_numeric_final"
    return "format_only_failure" if candidate == case["answer"] else "numeric_answer_wrong"


def summarize_previous_private(private: dict) -> dict:
    payload = private.get("scorer_input")
    if (not isinstance(payload, dict)
            or not isinstance(payload.get("cases"), list)
            or len(payload["cases"]) != 36
            or not isinstance(payload.get("observations"), list)
            or len(payload["observations"]) != 36):
        raise ValueError("Not the complete original hard36 private result")
    by_id = {c["id"]: c for c in payload["cases"]}
    if len(by_id) != 36 or {r["id"] for r in payload["observations"]} != set(by_id):
        raise ValueError("Unpaired private result")
    results = {a: {"by_domain": {}, "reasoning": {}} for a in ARMS}
    domains = sorted({c["domain"] for c in payload["cases"]})
    if set(domains) != {"math", "reasoning", "code_trace"}:
        raise ValueError("Invalid development domains")
    for arm in ARMS:
        for domain in domains:
            subset = [(by_id[row["id"]], row["arms"][arm])
                      for row in payload["observations"]
                      if by_id[row["id"]]["domain"] == domain]
            results[arm]["by_domain"][domain] = {
                "cases": len(subset),
                "failed_inference": sum(bool(obs["http_error"]) for _, obs in subset),
                "completed": sum(bool(obs["complete"]) for _, obs in subset),
                "strict_correct": sum(bool(_correct(c["answer"], obs["text"], obs["complete"]))
                                      for c, obs in subset),
            }
            results[arm]["by_domain"][domain]["outcomes"] = dict(Counter(
                _outcome(case, obs) for case, obs in subset))
        reasoning = [(by_id[row["id"]], row["arms"][arm])
                     for row in payload["observations"]
                     if by_id[row["id"]]["domain"] == "reasoning"]
        out = Counter()
        for case, obs in reasoning:
            if not obs["complete"]:
                out["failed_inference"] += 1
                continue
            out["completed_responses"] += 1
            if _correct(case["answer"], obs["text"], obs["complete"]):
                out["strict_correct"] += 1
                continue
            candidate = _numeric_candidate(obs["text"])
            if candidate is None:
                out["no_unambiguous_numeric_final"] += 1
            elif candidate == case["answer"]:
                out["format_only_failure"] += 1
            else:
                out["numeric_answer_wrong"] += 1
        results[arm]["reasoning"] = dict(out)
    paired_losses = {}
    for domain in domains:
        counts = Counter()
        for row in payload["observations"]:
            case = by_id[row["id"]]
            if case["domain"] != domain:
                continue
            raw = _outcome(case, row["arms"]["raw_qwen35"])
            tir = _outcome(case, row["arms"]["neral_high_router"])
            if raw == "strict_correct" and tir != "strict_correct":
                counts[tir] += 1
        paired_losses[domain] = dict(counts)
    return {
        "source": "supplied_private_hard36",
        "read_only": True, "raw_answers_uploaded": False,
        "tir_losses_vs_raw_by_domain": paired_losses,
        "independent_holdout_verified": False,
        "note": ("Legacy http_error also counts raw local Ollama errors; "
                 "older files do not contain exact error codes. "
                 "Lenient parse is DIAGNOSTIC ONLY, not a score change."),
        "arms": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-json", type=Path, required=True)
    args = parser.parse_args()
    path = args.private_json
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
        raise ValueError("Private result missing or invalid")
    payload = json.loads(path.read_text(encoding="utf8"))
    print(json.dumps(summarize_previous_private(payload), sort_keys=True))


if __name__ == "__main__":
    main()

