"""Offline scorer for same-qwen3.5 NERAL A/B/C evidence; no GPU or internet.

An external, separately approved Windows runner must collect paired live
observations using the SAME Qwen3.5 model digest and original user questions.
This file consumes local PRIVATE raw observations but emits ONLY scalar metrics.
No independent blind-holdout status or model-weight improvement is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import re
import statistics

from neral_numeric_final import extract_numeric_final

ARMS = ("raw_qwen35", "neral_high", "neral_high_router")
MODEL = "qwen3.5:9b"
PROFILE = "measured_nonthinking_2048"
SCHEMA = "NERAL_SAME_BASE_3ARM_DEVELOPMENT_DIAGNOSTIC"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")
INTEGER = re.compile(r"[+-]?\d{1,15}\Z")
FINAL = re.compile(r"FINAL(?:_ANSWER)?\s*:\s*([+-]?\d{1,15})\Z", re.I)
TOOLS = frozenset({None, "neral_math_exact", "neral_logic_exact",
                   "neral_python_trace_static", "neral_quant_exact",
                   "neral_python_sandbox"})

class SameBaseError(ValueError):
    """Never include prompt text or model responses in external errors."""


def _mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(min(wins, losses) + 1)) / 2**n) if n else 1.0


def _correct(gold: int, text: str, complete: bool) -> bool:
    """Score a bounded unambiguous integer final, identically for every arm.

    Never use the gold answer to select an arbitrary number from model prose.
    Missing/truncated or contradictory answers remain incorrect.
    """
    if type(complete) is not bool or not complete:
        return False
    candidate = extract_numeric_final(text)
    return candidate is not None and INTEGER.fullmatch(candidate) is not None and int(candidate) == gold


def _p95(values: list[float]) -> float:
    sorted_values = sorted(values)
    return round(sorted_values[math.ceil(len(values) * 0.95) - 1], 3)


def score_private_pairs(private: dict) -> dict:
    """Return aggregate-only, 36-case pilot or 120-case development results.

    Input is not an independently certified holdout. Observed correctness
    derives from an external CPU gold integer, never an LLM's own judgment.
    """
    if type(private) is not dict or set(private) != {"manifest", "cases", "observations"}:
        raise SameBaseError("Invalid private benchmark envelope")
    meta, cases, observations = (private[k] for k in ("manifest", "cases", "observations"))
    if (type(meta) is not dict or set(meta) != {"model", "model_digest", "profile", "token_budget"}
            or meta["model"] != MODEL or meta["profile"] != PROFILE
            or meta["token_budget"] != 2048 or type(meta["model_digest"]) is not str
            or DIGEST.fullmatch(meta["model_digest"]) is None):
        raise SameBaseError("Incorrect or unpinned same-model setup")
    if (type(cases) is not list or len(cases) not in (36, 120)
            or type(observations) is not list or len(observations) != len(cases)):
        raise SameBaseError("Only complete 36-case pilot or 120-case gate supported")
    case_by_id, domains, families = {}, set(), set()
    for case in cases:
        if (type(case) is not dict or set(case) != {"id", "domain", "family", "prompt", "answer"}
                or type(case["id"]) is not str or CASE_ID.fullmatch(case["id"]) is None
                or case["id"] in case_by_id or type(case["prompt"]) is not str
                or not 1 <= len(case["prompt"].encode("utf8")) <= 4096
                or type(case["domain"]) is not str or not 1 <= len(case["domain"]) <= 80
                or type(case["family"]) is not str or not 1 <= len(case["family"]) <= 80
                or type(case["answer"]) is not int):
            raise SameBaseError("Malformed or duplicate private case")
        case_by_id[case["id"]] = case
        domains.add(case["domain"])
        families.add((case["domain"], case["family"]))
    distribution = Counter(c["domain"] for c in cases)
    expected_distribution = ({"math": 12, "reasoning": 12, "code_trace": 12}
                             if len(cases) == 36 else
                             {"math": 30, "reasoning": 30, "code": 40, "tools": 20})
    if len(families) < 6 or dict(distribution) != expected_distribution:
        raise SameBaseError("Required balanced domain/family coverage missing")
    successes = {arm: set() for arm in ARMS}
    elapsed = {arm: [] for arm in ARMS}
    errors = {arm: 0 for arm in ARMS}
    tools = {arm: 0 for arm in ARMS}
    by_domain = {arm: {domain: {"count": 0, "correct": 0} for domain in sorted(domains)} for arm in ARMS}
    seen = set()
    for row in observations:
        if (type(row) is not dict or set(row) != {"id", "arms"} or row["id"] not in case_by_id
                or row["id"] in seen or type(row["arms"]) is not dict or set(row["arms"]) != set(ARMS)):
            raise SameBaseError("Unpaired or duplicate model observations")
        seen.add(row["id"])
        case = case_by_id[row["id"]]
        digest = hashlib.sha256(case["prompt"].encode("utf8")).hexdigest()
        for arm in ARMS:
            obs = row["arms"][arm]
            if (type(obs) is not dict or set(obs) != {"text", "complete", "http_error", "seconds", "tool_used", "model", "model_digest", "profile", "prompt_sha256"}
                    or obs["model"] != MODEL or obs["model_digest"] != meta["model_digest"]
                    or obs["profile"] != PROFILE or obs["prompt_sha256"] != digest
                    or type(obs["complete"]) is not bool or type(obs["http_error"]) is not bool
                    or obs["complete"] == obs["http_error"]
                    or type(obs["text"]) is not str or len(obs["text"].encode("utf8")) > 32768
                    or (not obs["complete"] and obs["text"])
                    or type(obs["seconds"]) not in (int, float) or not math.isfinite(obs["seconds"])
                    or not 0 <= obs["seconds"] <= 900 or obs["tool_used"] not in TOOLS
                    or (arm == "raw_qwen35" and obs["tool_used"] is not None)):
                raise SameBaseError("Untrustworthy provenance, timing or answer observation")
            elapsed[arm].append(obs["seconds"])
            errors[arm] += int(obs["http_error"])
            tools[arm] += int(obs["tool_used"] is not None)
            by_domain[arm][case["domain"]]["count"] += 1
            if _correct(case["answer"], obs["text"], obs["complete"]):
                successes[arm].add(row["id"])
                by_domain[arm][case["domain"]]["correct"] += 1
    if seen != set(case_by_id):
        raise SameBaseError("Missing case observations")
    n = len(cases)
    scores = {arm: {
        "correct": len(successes[arm]), "accuracy_percent": round(100 * len(successes[arm]) / n, 3),
        "http_errors": errors[arm], "tool_calls": tools[arm],
        "median_seconds": round(statistics.median(elapsed[arm]), 3),
        "mean_seconds": round(statistics.mean(elapsed[arm]), 3),
        "p95_seconds": _p95(elapsed[arm]), "by_domain": by_domain[arm],
    } for arm in ARMS}
    paired = {}
    for left, right in (("raw_qwen35", "neral_high"), ("raw_qwen35", "neral_high_router"), ("neral_high", "neral_high_router")):
        wins = len(successes[right] - successes[left])
        losses = len(successes[left] - successes[right])
        paired[f"{right}_vs_{left}"] = {
            "wins": wins, "losses": losses, "delta_pp": round(100 * (wins - losses) / n, 3),
            "exact_mcnemar_p": round(_mcnemar(wins, losses), 9),
        }
    return {
        "schema": 1, "status": SCHEMA, "cases": n, "domains": sorted(domains),
        "family_count": len(families), "model": MODEL, "model_digest": meta["model_digest"],
        "profile": PROFILE, "token_budget": 2048, "arms": scores, "paired": paired,
        "private_prompts_or_answers_uploaded": False,
        "independent_holdout_verified": False,
        "release_validated": False, "router_superiority_proven": False,
        "weight_improvement_proven": False, "automatic_promotion_authorized": False,
        "limitations": "A diagnostic score of externally collected answers, not a certified blind holdout."
                       " The collector must independently establish identity, decoder settings,"
                       " randomization, VRAM, prompt equality and tool execution."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-input", required=True, type=Path)
    parser.add_argument("--aggregate-output", required=True, type=Path)
    args = parser.parse_args()
    target = args.aggregate_output
    if not target.parent.is_dir() or target.exists() or target.is_symlink():
        raise SameBaseError("Create-only scalar report path required")
    with args.private_input.open("r", encoding="utf8") as src:
        private = json.load(src)
    report = score_private_pairs(private)
    with target.open("x", encoding="utf8") as out:
        json.dump(report, out, sort_keys=True)
        out.write("\n")
    print(json.dumps({"status": report["status"], "cases": report["cases"],
                      "independent_holdout_verified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
