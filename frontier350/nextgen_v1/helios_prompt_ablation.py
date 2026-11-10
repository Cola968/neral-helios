"""CPU-only own-weight development ablation: prompt format vs arithmetic recall.

Not an independent benchmark, training run, quality upgrade or general-chat claim.
Answers, prompts and per-candidate scores stay in an existing PRIVATE local file.
Only aggregate development metrics may be printed to CI. No network or CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from helios_guarded_preview import load_continuity_cpu
from helios_quality_diagnostic import DEV_CASES, as_prompt, output_target
from phoenix_continuity_cpu_probe import CONTINUITY_SHA, EXPECTED_TOKENS

# Publicly visible development cases, deliberately excluded from any private holdout.
MINIMAL = ("Was ist 2 + 2?", "What is 1 + 1?", "Was ist 7 - 4?")
EQUATIONS = ("2 + 2 =", "1 + 1 =", "7 - 4 =")
DIGITS = tuple(str(n) for n in range(10))
BUDGET = 16


def prompt_variants(question: str, minimal: str, equation: str) -> dict[str, str]:
    answer_label = "Antwort:" if minimal.startswith("Was") else "Answer:"
    question_label = "Frage:" if minimal.startswith("Was") else "Question:"
    return {
        "current_chat_instruction": "User: " + question + "\nAssistant:",
        "minimal_chat": "User: " + minimal + "\nAssistant:",
        "minimal_qa": question_label + " " + minimal + "\n" + answer_label,
        "bare_question": minimal + "\n" + answer_label,
        "equation": equation,
    }


def ranked_digits(model, tokenizer, bos: int, torch, prefix: str, expected: str) -> dict:
    context = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
    candidates = [tokenizer.encode(" " + digit, add_special_tokens=False).ids for digit in DIGITS]
    lengths = {len(ids) for ids in candidates}
    if not context or len(context) > 384 or len(lengths) != 1 or 0 in lengths:
        raise ValueError("unsupported context or unequal candidate lengths")
    if len({tuple(ids) for ids in candidates}) != 10 or len(context) + max(lengths) > model.cfg.max_seq_len:
        raise ValueError("candidate ambiguity or context overflow")
    scores = []
    with torch.inference_mode():
        # Efficient and mathematically identical to teacher forcing for one-token
        # candidates. For longer BPE spellings, score every continuation token.
        if next(iter(lengths)) == 1:
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            scores = [float(logp[ids[0]].item()) for ids in candidates]
        else:
            for ids in candidates:
                all_input = context + ids[:-1]
                logits, _ = model(torch.tensor([all_input], dtype=torch.long))
                logp = torch.log_softmax(logits[0, len(context) - 1:].float(), dim=-1)
                scores.append(float(logp.gather(-1, torch.tensor(ids)[:, None]).sum().item()))
    expected_index = int(expected)
    ranking = sorted(range(10), key=lambda i: scores[i], reverse=True)
    return {"prompt": prefix, "prompt_tokens_including_bos": len(context),
            "candidate_tokens": next(iter(lengths)), "top_digit": DIGITS[ranking[0]],
            "correct_rank": ranking.index(expected_index) + 1,
            "correct_top1": ranking[0] == expected_index,
            "correct_vs_best_wrong_logprob_margin": scores[expected_index] - max(
                scores[i] for i in range(10) if i != expected_index),
            "log_likelihoods": dict(zip(DIGITS, scores))}


def native_greedy(model, tokenizer, bos: int, eos: int, torch, prefix: str) -> dict:
    ids = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
    if len(ids) > 384:
        raise ValueError("prompt exceeds fixed development input ceiling")
    generated = []
    finish = "token_cap"
    with torch.inference_mode():
        for _ in range(BUDGET):
            logits, _ = model(torch.tensor([ids], dtype=torch.long))
            next_id = int(logits[0, -1].float().argmax().item())
            if next_id == eos:
                finish = "eos"
                break
            generated.append(next_id)
            ids.append(next_id)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_non_eos_tokens": len(generated), "finish_reason": finish}


def leading_digit(text: str) -> str | None:
    # A conservative format indicator. Not a semantic correctness judgment.
    match = re.match(r"^\s*([0-9])(?![0-9])", text)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trusted-local-checkpoint", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("explicit own-checkpoint opt-in and CPU isolation required")
    target = output_target(args.private_dir, args.output_name)
    previous = args.previous_report
    if previous.is_symlink() or not previous.is_file() or previous.resolve(strict=True).parent != target.parent:
        raise ValueError("previous real run must be an existing regular file in the same private directory")
    if previous.name != "helios_quality_4b9e3902e0a8.json":
        raise ValueError("expected the previous immutable, named 2B development run")
    earlier = json.loads(previous.read_text(encoding="utf-8"))
    if (earlier.get("checkpoint_sha256") != CONTINUITY_SHA or
            earlier.get("processed_tokens") != EXPECTED_TOKENS or
            len(earlier.get("cases", [])) != len(DEV_CASES)):
        raise ValueError("prior diagnostic does not match the exact own checkpoint/cases")
    model, tokenizer, bos, eos, processed, torch = load_continuity_cpu(args.assets_root, args.checkpoint)
    if processed != EXPECTED_TOKENS or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("wrong weights or execution device")
    rows = []
    for index, (case_id, question, answer) in enumerate(DEV_CASES):
        if earlier["cases"][index].get("id") != case_id or earlier["cases"][index].get("question") != question:
            raise ValueError("previous run does not contain exactly the same public development cases")
        as_prompt(tokenizer, bos, question)
        variants = prompt_variants(question, MINIMAL[index], EQUATIONS[index])
        ranks = {label: ranked_digits(model, tokenizer, bos, torch, prefix, answer)
                 for label, prefix in variants.items()}
        generated = {label: native_greedy(model, tokenizer, bos, eos, torch, variants[label])
                     for label in ("current_chat_instruction", "minimal_chat", "equation")}
        if generated["current_chat_instruction"]["text"] != earlier["cases"][index]["raw"]["answer"]:
            raise RuntimeError("raw replay changed: fail closed before reporting format effects")
        for item in generated.values():
            item["strict_exact"] = item["text"].strip() == answer
            item["leading_digit_correct"] = leading_digit(item["text"]) == answer
            item["expected_digit_appears"] = bool(re.search(r"(?<!\d)" + re.escape(answer) + r"(?!\d)", item["text"]))
        rows.append({"id": case_id, "expected": answer, "ranking": ranks, "generation": generated})
    names = tuple(prompt_variants(DEV_CASES[0][1], MINIMAL[0], EQUATIONS[0]))
    summary = {name: {"correct_top1": sum(r["ranking"][name]["correct_top1"] for r in rows),
                      "correct_ranks": [r["ranking"][name]["correct_rank"] for r in rows]}
               for name in names}
    generation_summary = {name: {"strict_exact": sum(r["generation"][name]["strict_exact"] for r in rows),
                                 "leading_digit_correct": sum(r["generation"][name]["leading_digit_correct"] for r in rows),
                                 "contains_digit_not_semantic": sum(r["generation"][name]["expected_digit_appears"] for r in rows),
                                 "native_eos": sum(r["generation"][name]["finish_reason"] == "eos" for r in rows)}
                          for name in ("current_chat_instruction", "minimal_chat", "equation")}
    report = {"classification": "PUBLIC_DEV_FORMAT_ABLATION_NOT_INDEPENDENT_BENCHMARK",
              "checkpoint_sha256": CONTINUITY_SHA, "processed_tokens": processed, "device": "cpu",
              "source_diagnostic": previous.name, "cases": rows, "digit_ranking": summary,
              "free_generation": generation_summary, "denominator": len(rows),
              "training_started": False, "weights_modified": False,
              "general_quality_improvement_proven": False,
              "interpretation": "Changes in syntax/question wording are confounded; small public development examples cannot establish a causal root or chat competence."}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"event": "own_weight_prompt_ablation_complete", "cases": len(rows),
                      "digit_rankings": summary, "free_generation": generation_summary,
                      "replay_identical": True, "private_report": str(target),
                      "training_started": False, "general_quality_improvement_proven": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
