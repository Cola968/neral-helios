"""Measure REAL free answers from the unchanged own 350M weights, CPU only.

Eight prepublished development prompts; deliberately NOT an independent holdout.
Compare greedy raw vs guarded with identical checkpoint, text and token budget.
Actual responses and per-case details stay on the owner's private local disk.
No training, GPU, network requests, model/checkpoint writes or artifact upload.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import helios350_preview as raw
import helios_guarded_preview as guarded
from helios_domain_dev_probe import PUBLIC_DEV_CASES, validate_public_cases
from helios_quality_diagnostic import as_prompt, output_target
from phoenix_continuity_cpu_probe import CONTINUITY_SHA, EXPECTED_TOKENS

BUDGET = 16
# Exactly one German and one English public case in each domain, fixed in code
# before running this diagnostic. Do NOT reuse these questions as a blind holdout.
CASE_IDS = ("m01", "m05", "w01", "w05", "s01", "s03", "a01", "a03")


def selected_cases() -> tuple[tuple, ...]:
    validate_public_cases()
    lookup = {row[0]: row for row in PUBLIC_DEV_CASES}
    if len(set(CASE_IDS)) != 8 or any(case_id not in lookup for case_id in CASE_IDS):
        raise ValueError("public development selection invalid")
    rows = tuple(lookup[case_id] for case_id in CASE_IDS)
    for domain in ("arithmetic", "everyday", "astronomy", "aviation"):
        selected = [row for row in rows if row[1] == domain]
        if len(selected) != 2 or {row[2] for row in selected} != {"de", "en"}:
            raise ValueError("domain/language coverage changed")
    return rows


def instruction(question: str, language: str) -> str:
    if language == "de":
        return "Antworte mit genau einer Ziffer. " + question
    if language == "en":
        return "Answer with exactly one digit. " + question
    raise ValueError("unsupported language")


def score(answer: str, expected: str) -> dict:
    """Report strict correctness separately from weaker, diagnostic digit prefix."""
    if not isinstance(answer, str) or expected not in "0123456789" or len(expected) != 1:
        raise ValueError("invalid free-answer score input")
    stripped = answer.strip()
    first = re.match(r"^([0-9])(?=\D|$)", stripped)
    return {"strict_exact": stripped == expected,
            "starts_with_correct_digit": bool(first and first.group(1) == expected),
            "nonempty": bool(stripped)}


def summarize(rows: list[dict]) -> dict:
    if len(rows) != len(CASE_IDS) or {row["id"] for row in rows} != set(CASE_IDS):
        raise ValueError("incomplete development set")
    result = {}
    for decoder in ("raw", "guarded"):
        result[decoder] = {
            "strict_exact": sum(row[decoder]["score"]["strict_exact"] for row in rows),
            "starts_with_correct_digit": sum(
                row[decoder]["score"]["starts_with_correct_digit"] for row in rows),
            "nonempty": sum(row[decoder]["score"]["nonempty"] for row in rows),
        }
    result["guarded"]["finish_reasons"] = {
        reason: sum(row["guarded"]["finish_reason"] == reason for row in rows)
        for reason in ("eos", "role_boundary", "repetition_guard", "max_tokens")
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-local-checkpoint", action="store_true")
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--private-dir", type=Path, required=True)
    p.add_argument("--output-name", required=True)
    args = p.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        p.error("explicit trusted-own-checkpoint opt-in and CPU-only isolation required")
    cases = selected_cases()
    destination = output_target(args.private_dir, args.output_name)
    model, tok, bos, eos, processed, torch = guarded.load_continuity_cpu(
        args.assets_root, args.checkpoint)
    if processed != EXPECTED_TOKENS or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("wrong original checkpoint or device")
    rows = []
    for case_id, domain, language, question, expected, _ in cases:
        prompt = instruction(question, language)
        as_prompt(tok, bos, prompt)  # Same exact tokenization used in live preview.
        raw_answer, raw_used = raw.generate(model, tok, bos, eos, torch, prompt, BUDGET)
        guarded_answer, guarded_used, finish = guarded.guarded_generate(
            model, tok, bos, eos, torch, prompt, BUDGET)
        rows.append({
            "id": case_id, "domain": domain, "language": language,
            "prompt": prompt, "expected": expected,
            "raw": {"answer": raw_answer, "non_eos_tokens": raw_used,
                    "score": score(raw_answer, expected)},
            "guarded": {"answer": guarded_answer, "generation_steps": guarded_used,
                        "finish_reason": finish, "score": score(guarded_answer, expected)},
        })
    aggregate = summarize(rows)
    report = {
        "classification": "PUBLIC_DEVELOPMENT_FREE_ANSWER_NOT_INDEPENDENT_HOLDOUT",
        "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed,
        "device": "cpu", "max_new_tokens_per_decoder": BUDGET,
        "same_checkpoint_same_prompt_same_budget": True, "cases": rows,
        "denominator": len(rows), "aggregate": aggregate,
        "strict_exact_definition": "entire generated text stripped equals the single target digit",
        "digit_prefix_is_not_answer_correctness": True,
        "training_started": False, "weights_changed": False,
        "independent_holdout": False, "quality_gain_verified": False,
    }
    # Only private results contain answers. Never log rows or model responses.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"event": "public_own_weight_free_answer_baseline_complete",
                      "checkpoint_sha256": CONTINUITY_SHA, "cases": len(rows),
                      "aggregate": aggregate, "private_report": str(destination),
                      "training_started": False, "independent_holdout": False,
                      "quality_gain_verified": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
