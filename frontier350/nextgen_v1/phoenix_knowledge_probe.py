"""Read-only exploratory factual-choice diagnostic for Neral's pinned 2.073B checkpoint.

PUBLIC author-written development examples, NOT independent holdout, not NIS2, not
proof of safety/quality. No training, answers, private prompts, GPU or model writes.
Compare normalized next-token log likelihood of two explicit alternatives under
both chat and plain QA prompts. No prompt is optimized automatically from results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from helios_guarded_preview import load_continuity_cpu

# Two equally important languages and subjects. Deliberately basic PUBLIC diagnostic
# material; never use these questions for training or the independent holdout.
CASES = (
    ("en", "What is two plus two?", " 4", " 7"),
    ("en", "What is the chemical formula of water?", " H2O", " CO2"),
    ("en", "Which planet is closest to the Sun?", " Mercury", " Neptune"),
    ("en", "What is the capital of France?", " Paris", " Rome"),
    ("en", "What is Earth's natural satellite?", " Moon", " Sun"),
    ("en", "How many degrees are in a right angle?", " 90", " 45"),
    ("de", "Wie viel ist drei mal drei?", " 9", " 6"),
    ("de", "Was ist die Hauptstadt von Italien?", " Rom", " Wien"),
    ("de", "Welches chemische Symbol steht fuer Sauerstoff?", " O", " N"),
    ("de", "Wie heisst die Kraft, die Massen anzieht?", " Gravitation", " Reibung"),
    ("de", "Welcher Planet ist als roter Planet bekannt?", " Mars", " Venus"),
    ("de", "Wie viele Sekunden hat eine Minute?", " 60", " 30"),
)
FORMATS = ("chat", "qa")


def prompt_for(language: str, question: str, variant: str) -> str:
    if variant == "chat":
        return "User: " + question + "\nAssistant:"
    if variant == "qa":
        return ("Question: " + question + "\nAnswer:" if language == "en"
                else "Frage: " + question + "\nAntwort:")
    raise ValueError("unknown prompt format")


def validate_cases(cases=CASES):
    if len(cases) != 12 or {row[0] for row in cases} != {"de", "en"}:
        raise ValueError("public diagnosis requires twelve balanced questions")
    if sum(row[0] == "en" for row in cases) != 6:
        raise ValueError("language distribution changed")
    prompts = set()
    for row in cases:
        if len(row) != 4 or not all(isinstance(x, str) for x in row):
            raise ValueError("invalid case")
        lang, question, correct, incorrect = row
        if lang not in {"de", "en"} or not 10 <= len(question) <= 200:
            raise ValueError("invalid language or question")
        if question.casefold() in prompts or not correct.startswith(" ") or not incorrect.startswith(" ") or correct == incorrect:
            raise ValueError("duplicate or malformed candidate")
        prompts.add(question.casefold())
    return True


def normalized_pair(correct: float, incorrect: float) -> tuple[int, float]:
    if not all(math.isfinite(x) for x in (correct, incorrect)):
        raise ValueError("nonfinite likelihood")
    delta = correct - incorrect
    return (1 if delta > 1e-7 else 0 if delta < -1e-7 else -1), delta


def score_completion(model, tokenizer, torch, bos: int, prefix: str, completion: str) -> float:
    """Teacher-forced token-average log P(answer | prefix), not generated output.

    Tokenize prefix/answer separately to match actual generation token boundaries.
    EOS and a length prior are intentionally not scored; this remains exploratory.
    """
    pre = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
    answer = list(tokenizer.encode(completion, add_special_tokens=False).ids)
    if not answer or len(pre) + len(answer) > 384:
        raise ValueError("empty alternative or overly long diagnostic sequence")
    ids = pre + answer[:-1]
    with torch.inference_mode():
        logits, _ = model(torch.tensor([ids], device="cpu", dtype=torch.long))
        selected = logits[0, len(pre) - 1:, :].float()
        targets = torch.tensor(answer, device="cpu", dtype=torch.long)
        log_probs = torch.log_softmax(selected, dim=-1)
        value = float(log_probs.gather(1, targets[:, None]).mean().item())
    if not math.isfinite(value):
        raise ValueError("nonfinite model score")
    return value


def evaluate(model, tokenizer, torch, bos: int) -> dict:
    validate_cases()
    result = {}
    for variant in FORMATS:
        counts = {"overall": [0, 0, 0], "de": [0, 0, 0], "en": [0, 0, 0]}
        margins = []
        for lang, question, answer, foil in CASES:
            prefix = prompt_for(lang, question, variant)
            decision, margin = normalized_pair(
                score_completion(model, tokenizer, torch, bos, prefix, answer),
                score_completion(model, tokenizer, torch, bos, prefix, foil))
            bucket = 0 if decision == 1 else 1 if decision == 0 else 2
            counts["overall"][bucket] += 1
            counts[lang][bucket] += 1
            margins.append(margin)
        result[variant] = {name: {"correct_preferred": values[0],
                                  "incorrect_preferred": values[1], "tie": values[2],
                                  "total": sum(values)} for name, values in counts.items()}
        result[variant]["mean_token_normalized_margin"] = round(sum(margins) / len(margins), 5)
    return {"classification": "public_development_factual_choice_probe_NOT_independent_quality",
            "checkpoint_tokens": 2_073_161_216,
            "prompt_formats": result,
            "cases": len(CASES), "pairs_per_format": len(CASES),
            "cases_sha256": hashlib.sha256(json.dumps(CASES, ensure_ascii=False).encode()).hexdigest(),
            "interpretation": "Only forced-choice preference under two public templates; neither free-answer correctness nor learned factual inventory.",
            "training_started": False, "gpu_used": False, "weights_modified": False,
            "candidate_promoted": False, "independent_holdout": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-local-checkpoint", action="store_true")
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    if not args.execute or not args.trusted_local_checkpoint:
        ap.error("explicit execute and own-checkpoint trust required")
    validate_cases()
    model, tok, bos, _, _, torch = load_continuity_cpu(args.assets_root, args.checkpoint)
    print(json.dumps(evaluate(model, tok, torch, bos), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
