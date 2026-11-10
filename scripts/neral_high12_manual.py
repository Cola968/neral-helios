"""Manual two-arm HIGH12: 4 math + 4 reasoning + 4 code_trace.

This is a development-only nonregression smoke, NOT a blind holdout,
guaranteed quality floor, release approval, or training process.
The workflow must pin PR #503 HEAD, the real local model digest and an
exclusive self-hosted GPU. No raw prompts, answers or model traces are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import secrets
import statistics
import time

MODEL = "qwen3.5:9b"
PROFILE = "measured_nonthinking_2048"
ARMS = ("raw_qwen35", "neral_high")
DOMAINS = ("math", "reasoning", "code_trace")
AUTHORIZATION = "NERAL_HIGH12_GPU_MANUAL_ONLY"
TARGET = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


# Only sanitized error codes are exported, never prompts or model text.
_FAILURE_REASONS = {
    "token_limit": "length_truncated",
    "timeout": "timeout",
    "empty_answer": "empty_response",
    "incomplete": "incomplete_response",
    "invalid_response": "invalid_response",
    "http_error": "http_error",
    "transport_error": "transport_error",
    "model_error": "model_error",
    "internal_error": "internal_error",
    "server_503": "http_error",
}
_ERROR_NAMES = tuple(sorted(set(_FAILURE_REASONS.values()) | {
    "missing_final_answer", "wrong_answer"
}))
_TECHNICAL = frozenset(_ERROR_NAMES) - {
    "length_truncated", "missing_final_answer", "wrong_answer"
}


class High12GateError(ValueError):
    """Sanitized errors: no private question, answer, code or user path."""


def select_cases(seed: int) -> list[dict]:
    """Select two fresh variants of each Hard36 family, then mix case order."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise High12GateError("Seed must be a 64-bit unsigned integer")
    from scripts.neral_tir_hard_live import generate_hard_cases
    all_cases = generate_hard_cases(seed)
    if len(all_cases) != 36 or len({c["id"] for c in all_cases}) != 36:
        raise High12GateError("Hard36 case generator returned an invalid batch")
    selected = []
    for domain in DOMAINS:
        rows = [c for c in all_cases if c["domain"] == domain]
        families = sorted({c["family"] for c in rows})
        if len(rows) != 12 or len(families) != 2:
            raise High12GateError("Unexpected Hard36 domain or family distribution")
        for family in families:
            candidates = [c for c in rows if c["family"] == family]
            if len(candidates) != 6:
                raise High12GateError("Unexpected case family size")
            family_seed = hashlib.sha256(
                (str(seed) + ":" + domain + ":" + family).encode("utf8")
            ).digest()
            random.Random(int.from_bytes(family_seed[:8], "big")).shuffle(candidates)
            selected.extend(candidates[:2])
    random.Random(seed ^ 0xA5D71026).shuffle(selected)
    if (len(selected) != 12 or len({c["id"] for c in selected}) != 12
            or Counter(c["domain"] for c in selected) != dict.fromkeys(DOMAINS, 4)):
        raise High12GateError("Failed exact 12-case balanced selection")
    return selected


def score_pairs(cases: list[dict], observations: list[dict], *,
                model_digest: str) -> dict:
    """Fail-closed paired ratchet; the same scorer parses both models' replies."""
    if type(model_digest) is not str or DIGEST.fullmatch(model_digest) is None:
        raise High12GateError("Unpinned model digest")
    if (type(cases) is not list or len(cases) != 12
            or type(observations) is not list or len(observations) != 12):
        raise High12GateError("Expected exactly twelve paired observations")
    if Counter(c.get("domain") for c in cases) != dict.fromkeys(DOMAINS, 4):
        raise High12GateError("Missing balanced domains")
    if len({c["id"] for c in cases}) != 12:
        raise High12GateError("Duplicate case")
    from scripts.neral_low_high_quality import make_prompt
    from scripts.neral_same_base_ab_score import _correct, TOOLS
    by_id = {c["id"]: c for c in cases}
    good = {arm: set() for arm in ARMS}
    failures = {arm: 0 for arm in ARMS}
    errors = {arm: Counter({name: 0 for name in _ERROR_NAMES}) for arm in ARMS}
    errors_by_domain = {
        domain: {arm: Counter({name: 0 for name in _ERROR_NAMES})
                 for arm in ARMS}
        for domain in DOMAINS
    }
    time_values = {arm: [] for arm in ARMS}
    by_domain = {
        domain: {"count": 4, "base_correct": 0, "high_correct": 0,
                 "wins": 0, "losses": 0}
        for domain in DOMAINS
    }
    seen = set()
    for row in observations:
        if (type(row) is not dict or set(row) != {"id", "prompt_sha256", "arms"}
                or row["id"] not in by_id or row["id"] in seen
                or type(row["arms"]) is not dict
                or set(row["arms"]) != set(ARMS)):
            raise High12GateError("Unpaired, missing or repeated result")
        seen.add(row["id"])
        case = by_id[row["id"]]
        if row["prompt_sha256"] != hashlib.sha256(
                make_prompt(case).encode("utf8")).hexdigest():
            raise High12GateError("Mismatched raw and HIGH question hash")
        outcomes = {}
        for arm in ARMS:
            obs = row["arms"][arm]
            if (type(obs) is not dict or
                    set(obs) != {"text", "complete", "failed", "seconds",
                                 "model", "model_digest", "profile", "tool_used",
                                 "failure_reason"}
                    or type(obs["text"]) is not str
                    or len(obs["text"].encode("utf8")) > 32768
                    or obs["model"] != MODEL
                    or obs["model_digest"] != model_digest
                    or obs["profile"] != PROFILE
                    or type(obs["complete"]) is not bool
                    or type(obs["failed"]) is not bool
                    or obs["complete"] == obs["failed"]
                    or ((obs["failure_reason"] is None) != obs["complete"])
                    or (obs["failed"] and obs["failure_reason"] not in _FAILURE_REASONS)
                    or (not obs["complete"] and obs["text"])
                    or type(obs["seconds"]) not in (int, float)
                    or not math.isfinite(obs["seconds"])
                    or not 0 <= obs["seconds"] <= 900
                    or obs["tool_used"] not in TOOLS
                    or (arm == "raw_qwen35" and obs["tool_used"] is not None)):
                raise High12GateError("Invalid paired observation provenance")
            failures[arm] += int(obs["failed"])
            time_values[arm].append(obs["seconds"])
            correct = _correct(case["answer"], obs["text"], obs["complete"])
            if obs["failed"]:
                error_kind = _FAILURE_REASONS[obs["failure_reason"]]
            elif not correct:
                from neral_numeric_final import extract_numeric_final
                error_kind = ("missing_final_answer"
                              if extract_numeric_final(obs["text"]) is None
                              else "wrong_answer")
            else:
                error_kind = None
            if error_kind is not None:
                errors[arm][error_kind] += 1
                errors_by_domain[case["domain"]][arm][error_kind] += 1
            outcomes[arm] = correct
            if correct:
                good[arm].add(case["id"])
                by_domain[case["domain"]][
                    "base_correct" if arm == "raw_qwen35" else "high_correct"] += 1
        if outcomes["neral_high"] and not outcomes["raw_qwen35"]:
            by_domain[case["domain"]]["wins"] += 1
        elif outcomes["raw_qwen35"] and not outcomes["neral_high"]:
            by_domain[case["domain"]]["losses"] += 1
    if seen != set(by_id):
        raise High12GateError("Incomplete paired observation matrix")
    wins = len(good["neral_high"] - good["raw_qwen35"])
    losses = len(good["raw_qwen35"] - good["neral_high"])
    technical = {
        arm: sum(errors[arm][name] for name in _TECHNICAL)
        for arm in ARMS
    }
    format_failures = {
        arm: errors[arm]["length_truncated"] +
             errors[arm]["missing_final_answer"]
        for arm in ARMS
    }
    # The baseline can itself truncate. Real technical errors on either side
    # still fail; extra HIGH length/format failures are regressions.
    # This is NOT a blind holdout or automatic release authorization.
    pass_gate = (len(good["raw_qwen35"]) >= 3
                 and not any(technical.values())
                 and format_failures["neral_high"] <= format_failures["raw_qwen35"]
                 and len(good["neral_high"]) >= len(good["raw_qwen35"]))
    return {
        "schema": 1,
        "status": "NERAL_HIGH12_PAIRED_DEVELOPMENT_ONLY",
        "cases": 12, "domains": list(DOMAINS), "family_count": 6,
        "model": MODEL, "model_digest": model_digest,
        "profile": PROFILE, "token_budget": 2048,
        "raw_correct": len(good["raw_qwen35"]),
        "high_correct": len(good["neral_high"]),
        "paired_wins": wins, "paired_losses": losses,
        "paired_delta_pp": round((wins - losses) * 100 / 12, 3),
        "failed_calls": failures,
        "error_counts": {arm: dict(errors[arm]) for arm in ARMS},
        "technical_failures": technical,
        "format_failures": format_failures,
        "errors_by_domain": {
            domain: {arm: dict(errors_by_domain[domain][arm]) for arm in ARMS}
            for domain in DOMAINS
        },
        "seed_option_present": False,
        "median_seconds": {arm: round(statistics.median(time_values[arm]), 3)
                           for arm in ARMS},
        "by_domain": by_domain,
        "development_gate_passed": pass_gate,
        "release_approval": False,
        "independent_holdout_verified": False,
        "model_weights_improved": False,
        "automatic_merge_allowed": False,
        "private_prompts_or_answers_uploaded": False,
        "limitations": "Tiny 12-case same-run diagnostic; no release or holdout certification.",
    }


def run_gpu(*, authorize: str, target_sha: str, output: Path) -> bool:
    """One exclusive pinned 4070 inference run, without raw response exports."""
    if authorize != AUTHORIZATION or type(target_sha) is not str or not TARGET.fullmatch(target_sha):
        raise High12GateError("Manual GPU consent or exact target SHA missing")
    output = Path(output)
    if (output.exists() or output.is_symlink() or output.suffix != ".json"
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise High12GateError("Fresh scalar-only output file is required")
    import os
    if os.environ.get("NERAL_EXPERIMENTAL_TIR") != "0":
        raise High12GateError("Experimental TIR must be disabled")

    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_same_base_live import _high_client, require_model, observation
    from scripts.neral_tir_hard_live import _run_one
    from scripts.neral_low_high_quality import make_prompt
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from neral_agent_autonomy import OllamaLocalBackend

    digest = require_model(OllamaLocal(timeout=120).inventory())
    if digest != ORIGINAL_DIGESTS[MODEL]:
        raise High12GateError("Wrong Qwen3.5 model digest")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    client, headers = _high_client(backend)

    # One hidden seed, one shared batch. No seed or question is uploaded.
    seed = secrets.randbits(64)
    cases = select_cases(seed)
    rows = []
    try:
        for index, case in enumerate(cases):
            prompt = make_prompt(case)
            order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            row = {"id": case["id"], "prompt_sha256":
                   hashlib.sha256(prompt.encode("utf8")).hexdigest(), "arms": {}}
            for arm in order:
                answer, complete, failed, seconds, tool, _, _, reason = _run_one(
                    arm, prompt, backend, client, headers)
                if failed and reason == "model_error":
                    # Record real local HTTP/transport evidence when available.
                    diag = backend.last_inference_diagnostic
                    if isinstance(diag, dict):
                        status = diag.get("ollama_http_status")
                        kind = diag.get("ollama_error_type")
                        if type(status) is int and status >= 400:
                            reason = "http_error"
                        elif kind == "transport_error":
                            reason = "transport_error"
                row["arms"][arm] = {
                    "text": answer if complete else "",
                    "complete": complete, "failed": failed,
                    "seconds": round(seconds, 3),
                    "model": MODEL, "model_digest": digest,
                    "profile": PROFILE, "tool_used": tool,
                    "failure_reason": reason if failed else None,
                }
            rows.append(row)
            if (index+1) % 2 == 0:
                print(f"HIGH12_PAIR_PROGRESS {(index+1)}/12", flush=True)
    finally:
        backend.unload()
    if require_model(OllamaLocal(timeout=120).inventory()) != digest:
        raise High12GateError("Local model digest changed during comparison")
    result = score_pairs(cases, rows, model_digest=digest)
    result["target_sha"] = target_sha
    result["seed_commitment_sha256"] = hashlib.sha256(str(seed).encode("ascii")).hexdigest()
    result["elapsed_type"] = "actual_local_inference_only"
    with output.open("x", encoding="utf8") as out:
        json.dump(result, out, ensure_ascii=False, sort_keys=True, allow_nan=False)
        out.write("\n")
    # Print only counts; no raw task, code, answer or prompt.
    print(json.dumps({
        "status": result["status"], "raw_correct": result["raw_correct"],
        "high_correct": result["high_correct"], "paired_wins": result["paired_wins"],
        "paired_losses": result["paired_losses"],
        "development_gate_passed": result["development_gate_passed"],
    }, sort_keys=True), flush=True)
    return result["development_gate_passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        passed = run_gpu(authorize=args.authorize, target_sha=args.target_sha, output=args.out)
    except Exception as exc:
        # Preserve private model text and local paths in errors and tracebacks.
        print("HIGH12_ABORT: " + type(exc).__name__, flush=True)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
