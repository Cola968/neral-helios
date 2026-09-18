"""24 explicitly PUBLIC development probes for Neral 350M: not an independent holdout.

Check candidate digit likelihood under comparable chat/QA prompts for four domains.
All per-question scores remain on the owner's PC in a new immutable private JSON.
Aggregates only are printed to Actions logs. No training, weights writes, GPU or upload.
A top-ranked digit is a forced-choice measure, NOT actual free-answer correctness.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from helios_guarded_preview import load_continuity_cpu
from helios_prompt_ablation import ranked_digits
from helios_quality_diagnostic import output_target
from phoenix_continuity_cpu_probe import CONTINUITY_SHA, EXPECTED_TOKENS

# These 24 familiar questions are published directly in source, NOT private test data.
# id, domain, language, short question, expected single digit, optional equation.
PUBLIC_DEV_CASES = (
    ("m01", "arithmetic", "de", "Wie viel ist 3 + 4?", "7", "3 + 4 ="),
    ("m02", "arithmetic", "de", "Wie viel ist 8 - 3?", "5", "8 - 3 ="),
    ("m03", "arithmetic", "de", "Wie viel ist 6 / 2?", "3", "6 / 2 ="),
    ("m04", "arithmetic", "de", "Wie viel ist 2 * 3?", "6", "2 * 3 ="),
    ("m05", "arithmetic", "en", "What is 9 - 8?", "1", "9 - 8 ="),
    ("m06", "arithmetic", "en", "What is 2 + 5?", "7", "2 + 5 ="),
    ("m07", "arithmetic", "en", "What is 4 * 2?", "8", "4 * 2 ="),
    ("m08", "arithmetic", "en", "What is 7 - 2?", "5", "7 - 2 ="),
    ("w01", "everyday", "de", "Wie viele Beine hat ein Mensch normalerweise?", "2", None),
    ("w02", "everyday", "de", "Wie viele Seiten hat ein Dreieck?", "3", None),
    ("w03", "everyday", "de", "Wie viele Beine hat eine Spinne?", "8", None),
    ("w04", "everyday", "de", "Wie viele Tage hat eine Woche?", "7", None),
    ("w05", "everyday", "en", "How many wheels does a bicycle have?", "2", None),
    ("w06", "everyday", "en", "How many fingers does a typical hand have, including the thumb?", "5", None),
    ("w07", "everyday", "en", "How many sides does a square have?", "4", None),
    ("w08", "everyday", "en", "How many wheels does a tricycle have?", "3", None),
    ("s01", "astronomy", "de", "Wie viele Planeten hat unser Sonnensystem?", "8", None),
    ("s02", "astronomy", "de", "Wie viele Sterne hat unser Sonnensystem?", "1", None),
    ("s03", "astronomy", "en", "How many natural moons does Earth have?", "1", None),
    ("s04", "astronomy", "en", "How many stars form a binary star system?", "2", None),
    ("a01", "aviation", "de", "Wie viele Haupttriebwerke hat ein Airbus A380?", "4", None),
    ("a02", "aviation", "de", "Wie viele Haupttriebwerke hat ein Airbus A320?", "2", None),
    ("a03", "aviation", "en", "How many main engines does a Boeing 747 have?", "4", None),
    ("a04", "aviation", "en", "How many main engines does a Boeing 787 have?", "2", None),
)


def prompt_styles(question: str, language: str, equation: str | None) -> dict[str, str]:
    if language == "de":
        instructed = "Antworte mit genau einer Ziffer. " + question
        qa = "Frage: " + question + "\nAntwort:"
    else:
        instructed = "Answer with exactly one digit. " + question
        qa = "Question: " + question + "\nAnswer:"
    result = {"instruction_chat": "User: " + instructed + "\nAssistant:",
              "minimal_chat": "User: " + question + "\nAssistant:",
              "qa": qa}
    if equation is not None:
        result["equation"] = equation
    return result


def validate_public_cases() -> None:
    if len(PUBLIC_DEV_CASES) != 24 or len({row[0] for row in PUBLIC_DEV_CASES}) != 24:
        raise ValueError("public development suite identity changed")
    domains = {"arithmetic": 8, "everyday": 8, "astronomy": 4, "aviation": 4}
    for domain, count in domains.items():
        values = [row for row in PUBLIC_DEV_CASES if row[1] == domain]
        if len(values) != count or sum(row[2] == "de" for row in values) != count // 2:
            raise ValueError("domain or language stratification changed")
    if any(expected not in "0123456789" or len(expected) != 1 or
           (domain == "arithmetic") != (equation is not None)
           for _, domain, _, _, expected, equation in PUBLIC_DEV_CASES):
        raise ValueError("invalid single-digit answer or equation mapping")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trusted-local-checkpoint", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or os.getenv("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("explicit own-weight opt-in and CPU-only isolation required")
    validate_public_cases()
    output = output_target(args.private_dir, args.output_name)
    model, tok, bos, _, processed, torch = load_continuity_cpu(args.assets_root, args.checkpoint)
    if processed != EXPECTED_TOKENS or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("pinned own checkpoint or device mismatch")
    detailed = []
    for case_id, domain, language, question, expected, equation in PUBLIC_DEV_CASES:
        variants = prompt_styles(question, language, equation)
        rankings = {name: ranked_digits(model, tok, bos, torch, prefix, expected)
                    for name, prefix in variants.items()}
        detailed.append({"id": case_id, "domain": domain, "language": language,
                         "question": question, "expected": expected, "rankings": rankings})
    domain_summary: dict[str, dict] = {}
    for domain in ("arithmetic", "everyday", "astronomy", "aviation"):
        rows = [row for row in detailed if row["domain"] == domain]
        labels = tuple(rows[0]["rankings"])
        domain_summary[domain] = {"cases": len(rows),
                                  "de": sum(row["language"] == "de" for row in rows),
                                  "en": sum(row["language"] == "en" for row in rows),
                                  "formats": {name: {"correct_digit_top1": sum(
                                      row["rankings"][name]["correct_top1"] for row in rows),
                                      "mean_correct_digit_rank": round(sum(
                                          row["rankings"][name]["correct_rank"] for row in rows) / len(rows), 3)}
                                      for name in labels}}
    report = {"classification": "PUBLIC_DEVELOPMENT_FORCED_CHOICE_NOT_CHAT_ACCURACY",
              "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed,
              "device": "cpu", "cases": detailed, "domain_summary": domain_summary,
              "denominator": len(detailed), "training_started": False,
              "weights_changed": False, "independent_holdout": False,
              "general_quality_improvement_proven": False,
              "notes": "All prompts and expected answers are public, formats have different token contexts; no free-generation or independent capability claim."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"event": "pinned_own_weight_domain_dev_complete", "cases": 24,
                      "domain_summary": domain_summary, "private_output": str(output),
                      "training_started": False, "independent_holdout": False,
                      "general_quality_improvement_proven": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
