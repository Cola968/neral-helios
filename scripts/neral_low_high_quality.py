"""Pairwise LOW-versus-HIGH development quality gate for local NERAL 1.0.

Runs the SAME questions through the same local Ollama model via /generate
and /think. Grading uses deterministic reference integers from the repository's
existing 72-case DEVELOPMENT suite. No external AI, downloads, GPU training,
untrusted code execution, or publication of raw questions/responses.

This is *not* an unseen release holdout: the suite is already published in
the repository. A positive result is a development signal, not proof of
general reasoning superiority and not permission to publish a stable release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from scripts.neral_ext import semantic_grade
from scripts.neral_v7_foundation_tournament import generate_cases, OllamaLocal

_MODEL = "qwen3:8b"
_ORIGIN = "local_foundation_not_neral_adapter"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOOLS = frozenset({None, "neral_math_exact", "neral_logic_exact",
                    "neral_python_trace_static", "neral_percentage_exact",
                    "neral_quantitative_exact", "neral_model_provenance"})


class QualityGateError(ValueError):
    """No private prompts, model text, or system paths in error messages."""


def make_prompt(case: dict) -> str:
    """Fix the same grading instruction for both modes before model calls."""
    if not isinstance(case, dict) or not isinstance(case.get("prompt"), str):
        raise QualityGateError("Invalid case")
    text = case["prompt"]
    if not 1 <= len(text) <= 2000:
        raise QualityGateError("Unbounded case")
    return text + "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."


def _validate(cases: list[dict], responses: list[dict], digest: str) -> dict:
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise QualityGateError("Pinned local Ollama digest required")
    if not isinstance(cases, list) or not isinstance(responses, list) or not 1 <= len(cases) <= 600:
        raise QualityGateError("Invalid paired suite size")
    keys = set()
    for case in cases:
        if (not isinstance(case, dict) or
                not {"id", "domain", "family", "kind", "answer", "prompt"} <= case.keys() or
                not isinstance(case["id"], str) or
                not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"]) or
                case["id"] in keys or case["kind"] != "integer" or
                not isinstance(case["answer"], int) or
                type(case["answer"]) is not int or
                not isinstance(case["domain"], str) or
                not isinstance(case["family"], str) or
                len(case["family"]) > 80):
            raise QualityGateError("Invalid or duplicate benchmark case")
        make_prompt(case)
        keys.add(case["id"])
    if len(responses) != len(cases):
        raise QualityGateError("Missing model observations")
    indexed = {}
    for row in responses:
        if (not isinstance(row, dict) or set(row) != {"id", "low", "high"} or
                not isinstance(row["id"], str) or
                row["id"] not in keys or row["id"] in indexed):
            raise QualityGateError("Unpaired or duplicate observation")
        for mode, key in (("LOW", "low"), ("HIGH", "high")):
            item = row[key]
            if (not isinstance(item, dict) or
                    set(item) != {"text", "complete", "mode", "model", "model_origin",
                                  "adapter_loaded", "seconds", "tool_used"} or
                    item["mode"] != mode or item["model"] != _MODEL or
                    item["model_origin"] != _ORIGIN or
                    item["adapter_loaded"] is not False or
                    type(item["complete"]) is not bool or
                    not isinstance(item["text"], str) or
                    len(item["text"].encode("utf8")) > 32768 or
                    type(item["seconds"]) not in (int, float) or
                    not math.isfinite(item["seconds"]) or
                    item["seconds"] < 0 or item["seconds"] > 86400 or
                    item["tool_used"] not in _TOOLS):
                raise QualityGateError("Invalid mode, origin or observation")
            if not item["complete"] and item["text"]:
                raise QualityGateError("Incomplete output must not be scored")
        indexed[row["id"]] = row
    return indexed


def _mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k)
                   for k in range(min(wins, losses) + 1)) / 2**n)


def score_paired(cases: list[dict], responses: list[dict], *,
                 model_digest: str) -> dict:
    """Count only independently gradeable exact results; errors score zero."""
    indexed = _validate(cases, responses, model_digest)
    low_correct = high_correct = wins = losses = tool_count = 0
    domains = {}
    latencies = {"low": [], "high": []}
    for case in cases:
        pair = indexed[case["id"]]
        grades = []
        for mode in ("low", "high"):
            item = pair[mode]
            # The independent exact grader rejects missing finals, long
            # outputs, errors, and unsupported syntax. Never extract an
            # arbitrary number from a model's answer.
            grade = semantic_grade(case, item["text"], complete=item["complete"])
            grades.append(grade == "correct")
            latencies[mode].append(item["seconds"])
        low_ok, high_ok = grades
        low_correct += int(low_ok)
        high_correct += int(high_ok)
        wins += int(high_ok and not low_ok)
        losses += int(low_ok and not high_ok)
        tool_count += int(pair["high"]["tool_used"] is not None)
        d = domains.setdefault(case["domain"], {"count": 0,
                             "low_correct": 0, "high_correct": 0})
        d["count"] += 1
        d["low_correct"] += int(low_ok)
        d["high_correct"] += int(high_ok)

    count = len(cases)
    methods = {(c["domain"], c["family"]) for c in cases}
    pp = 100.0 * (high_correct - low_correct) / count
    pvalue = _mcnemar(wins, losses)
    # The gate is conservative and pre-fixed: sufficient family/domain
    # coverage, a substantial positive net gain, no domain lost, and
    # paired discordant significance. A 12-case pilot must never pass it.
    coverage = count >= 72 and len(domains) >= 3 and len(methods) >= 12
    no_domain_regression = all(
        x["high_correct"] >= x["low_correct"] for x in domains.values())
    passed = bool(coverage and pp >= 5.0 and wins > losses and
                  no_domain_regression and pvalue <= 0.05)
    suite_bytes = json.dumps(cases, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf8")
    return {
        "schema": 1,
        "status": "NERAL_LOW_HIGH_PAIRED_DEVELOPMENT_DIAGNOSTIC",
        "model": _MODEL, "model_digest": model_digest,
        "case_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "case_count": count, "family_count": len(methods),
        "low_correct": low_correct, "high_correct": high_correct,
        "difference_pp": round(pp, 3),
        "paired": {"wins": wins, "losses": losses,
                   "exact_mcnemar_p": round(pvalue, 8)},
        "by_domain": domains,
        "high_tool_assisted": tool_count,
        "mean_seconds": {k: round(sum(v) / len(v), 3)
                         for k, v in latencies.items()},
        "development_gate_passed": passed,
        "neural_improvement_proven": False,
        "general_reasoning_superiority_proven": False,
        "independent_holdout_verified": False,
        "release_validated": False,
        "limitations": "Public development cases; system-level LOW/HIGH comparison, not a blinded release holdout or a neural-weight comparison.",
    }


def run_local_probe(*, count: int, output: Path) -> dict:
    """Read-only local inference; only a scalar report is written to output."""
    if type(count) is not int or count not in (12, 24, 72):
        raise QualityGateError("Allowed study sizes: 12, 24 or 72")
    output = Path(output)
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.parent.is_symlink() or output.suffix != ".json"):
        raise QualityGateError("Fresh local report file required")
    installed = OllamaLocal().inventory()
    digest = installed.get(_MODEL)
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise QualityGateError("Pinned local model not installed")
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_preview_server import create_preview_app

    app = create_preview_app(OllamaLocalBackend(_MODEL, timeout=90))
    client = app.test_client()
    base_url = "http://127.0.0.1:9690"
    if client.get("/", base_url=base_url).status_code != 200:
        raise QualityGateError("Local preview session unavailable")
    session = client.get("/api/neral/preview/session", base_url=base_url)
    if session.status_code != 200:
        raise QualityGateError("Local preview session refused")
    headers = {"X-Neral-CSRF": session.json["csrf"]}
    cases = generate_cases()[:count]
    rows = []

    for index, case in enumerate(cases):
        prompt = make_prompt(case)
        result = {"id": case["id"]}
        # Alternating order controls first/second request ordering bias.
        order = ("low", "high") if index % 2 == 0 else ("high", "low")
        for name in order:
            start = time.monotonic()
            response = client.post(
                "/api/neral/chat/generate" if name == "low"
                else "/api/neral/chat/think", json={"prompt": prompt},
                headers=headers, base_url=base_url)
            seconds = round(time.monotonic() - start, 3)
            valid = response.status_code == 200 and isinstance(response.json, dict)
            packet = response.json if valid else {}
            correct_origin = valid and (packet.get("model") == _MODEL and
                packet.get("model_origin") == _ORIGIN and
                packet.get("adapter_loaded") is False)
            if valid and not correct_origin:
                raise QualityGateError("Unexpected model origin; abort measurement")
            value = packet.get("text") if valid else ""
            complete = bool(isinstance(value, str) and value.strip())
            result[name] = {
                "text": value if complete else "", "complete": complete,
                "mode": name.upper(), "model": _MODEL,
                "model_origin": _ORIGIN, "adapter_loaded": False,
                "seconds": seconds,
                "tool_used": packet.get("tool_used") if valid else None,
            }
        rows.append(result)

    if OllamaLocal().inventory().get(_MODEL) != digest:
        raise QualityGateError("Local model digest changed during measurement")
    report = score_paired(cases, rows, model_digest=digest)
    with output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"],
                      "case_count": report["case_count"],
                      "low_correct": report["low_correct"],
                      "high_correct": report["high_correct"],
                      "development_gate_passed": report["development_gate_passed"]},
                     sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(12, 24, 72), default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_local_probe(count=args.count, output=args.out)


if __name__ == "__main__":
    main()
