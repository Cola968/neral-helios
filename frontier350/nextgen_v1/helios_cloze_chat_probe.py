"""Public development-only cloze-vs-chat probe on immutable original Neral weights.

This checks whether the model can finish a factual sentence at all, rather
than pretending the existing 0/8 chatbot result is a complete diagnosis.
Different prompts/tasks confound causal conclusions. No weights or datasets
are written, no internet access, no GPU; detailed generated text stays LOCAL.
This is NOT an independent holdout, SFT, model improvement or quality claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from helios_guarded_preview import load_continuity_cpu
from helios_prompt_ablation import native_greedy
from helios_quality_diagnostic import output_target
from phoenix_continuity_cpu_probe import CONTINUITY_SHA, EXPECTED_TOKENS

# Fixed, intentionally public BEFORE evaluation. No private holdout use.
# case ID, language, unambiguous sentence prefix, corresponding chat question,
# accepted FIRST WORDS (not a full semantic scoring rubric).
PUBLIC_CLOZE_CASES = (
    ("de_capital", "de", "Die Hauptstadt von Deutschland ist", "Was ist die Hauptstadt von Deutschland?", ("berlin",)),
    ("en_capital", "en", "The capital of France is", "What is the capital of France?", ("paris",)),
    ("de_orbit", "de", "Die Erde umkreist die", "Was umkreist die Erde?", ("sonne",)),
    ("en_orbit", "en", "Earth orbits the", "What does Earth orbit?", ("sun",)),
    ("de_shape", "de", "Ein Quadrat hat", "Wie viele Seiten hat ein Quadrat?", ("vier", "4")),
    ("en_shape", "en", "A triangle has", "How many sides does a triangle have?", ("three", "3")),
    ("de_add", "de", "Das Ergebnis von 2 + 2 ist", "Was ist 2 + 2?", ("vier", "4")),
    ("en_add", "en", "The result of 1 + 1 is", "What is 1 + 1?", ("two", "2")),
)
BUDGET = 16


def first_word(text: str) -> str | None:
    """Deliberately strict first lexical item; an answer appearing later is NO hit."""
    match = re.match(r"^\s*[^\w]*([\w]+)", text, flags=re.UNICODE)
    return match.group(1).casefold() if match else None


def validate_public_cases() -> None:
    if len(PUBLIC_CLOZE_CASES) != 8 or len({c[0] for c in PUBLIC_CLOZE_CASES}) != 8:
        raise ValueError("expected exactly eight distinct publicly locked cases")
    if sum(c[1] == "de" for c in PUBLIC_CLOZE_CASES) != 4 or sum(c[1] == "en" for c in PUBLIC_CLOZE_CASES) != 4:
        raise ValueError("language split unexpectedly changed")
    if any(not c[2].strip() or not c[3].strip() or not c[4] or
           any(not a or a != a.casefold() or re.search(r"\s", a) for a in c[4])
           for c in PUBLIC_CLOZE_CASES):
        raise ValueError("malformed public cloze case")


def summarize(rows: list[dict]) -> dict:
    if len(rows) != 8 or len({row["id"] for row in rows}) != 8:
        raise ValueError("incomplete or duplicate development results")
    summary = {}
    for mode in ("document_completion", "chat_question"):
        summary[mode] = {
            "first_word_matches_predeclared_answers": sum(row[mode]["first_word_match"] for row in rows),
            "nonempty": sum(bool(row[mode]["answer"].strip()) for row in rows),
            "native_eos": sum(row[mode]["finish_reason"] == "eos" for row in rows),
            "denominator": len(rows),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trusted-local-checkpoint", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("explicit own-checkpoint opt-in and CPU isolation required")
    validate_public_cases()
    output = output_target(args.private_dir, args.output_name)
    model, tok, bos, eos, processed, torch = load_continuity_cpu(args.assets_root, args.checkpoint)
    if processed != EXPECTED_TOKENS or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("wrong checkpoint/device; reject measurements")
    rows = []
    for case_id, language, document, question, answers in PUBLIC_CLOZE_CASES:
        variants = {"document_completion": document,
                    "chat_question": "User: " + question + "\nAssistant:"}
        results = {}
        for mode, prompt in variants.items():
            generated = native_greedy(model, tok, bos, eos, torch, prompt)
            word = first_word(generated["text"])
            results[mode] = {"answer": generated["text"],
                             "first_word": word,
                             "first_word_match": word in answers,
                             "finish_reason": generated["finish_reason"],
                             "generated_non_eos_tokens": generated["generated_non_eos_tokens"]}
        rows.append({"id": case_id, "language": language, "document_prefix": document,
                     "chat_question": question, "accepted_first_words": answers,
                     **results})
    summary = summarize(rows)
    report = {"classification": "PUBLIC_DEVELOPMENT_CLOZE_VS_CHAT_NOT_INDEPENDENT_BENCHMARK",
              "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed, "device": "cpu",
              "generation_policy": "identical raw greedy argmax, max 16 new tokens, BOS + public prompt",
              "cases": rows, "summary": summary,
              "comparison_limit": "Different prompt/task distributions; first-word lexical checks are not general semantic answer correctness and cannot establish a causal root cause.",
              "weights_modified": False, "training_started": False,
              "independent_holdout": False, "general_quality_improvement_proven": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    # Generated text and per-case answers are never uploaded or logged.
    print(json.dumps({"event": "public_cloze_chat_diagnostic_completed",
                      "checkpoint_sha256": CONTINUITY_SHA, "cases": len(rows),
                      "aggregate": summary, "private_output": str(output),
                      "training_started": False, "independent_holdout": False,
                      "general_quality_improvement_proven": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
