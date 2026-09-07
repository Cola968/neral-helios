"""Neral model-only GSM8K direct-answer diagnostic.

This harness is intentionally separate from NIS and the frozen internal promotion suites. It
uses the reserved GSM8K *test* split prepared by ``prepare_gsm8k_holdout50.py`` and performs
greedy raw-model generation with no SOLVE, Helios, retrieval or external model. The score is
an exact numeric final-answer rate under Neral's explicit prompt/extraction contract; it is not
claimed to be an official Hugging Face/Inspect/Lighteval leaderboard score.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch

from experiment import TOKENIZER, amp, load_model, sha

ANSWER_RE = re.compile(r"(?i)answer\s*:\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)")
NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")


def canonical_number(value: str):
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace("$", "")
    text = text.strip(" .;:!?)\]}\n\r\t")
    try:
        if "/" in text:
            return Fraction(text)
        return Fraction(Decimal(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_prediction(text: str):
    matches = ANSWER_RE.findall(text)
    if matches:
        raw = matches[-1]
        return raw, canonical_number(raw), "answer_marker"
    numbers = NUMBER_RE.findall(text)
    if numbers:
        raw = numbers[-1]
        return raw, canonical_number(raw), "last_number_fallback"
    return None, None, "no_number"


def read_holdout(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("dataset_id") != "openai/gsm8k" or manifest.get("training_allowed") is not False:
        raise ValueError("GSM8K holdout manifest is not evaluation-only")
    data = root / manifest["prepared_file"]
    rows = [json.loads(line) for line in data.read_text(encoding="utf8").splitlines() if line.strip()]
    if len(rows) != int(manifest["examples"]):
        raise ValueError("GSM8K holdout row count mismatch")
    return manifest, rows


def prompt_for(question: str):
    return (
        "User: Solve this math problem carefully. Give a concise calculation and finish with exactly "
        "'Answer: <number>'.\nProblem: " + question.strip() + "\nAssistant:"
    )


@torch.inference_mode()
def generate(model, device, prompt: str, max_new_tokens: int):
    ids = TOKENIZER.encode(prompt, bos=True)
    if len(ids) + max_new_tokens > model.cfg.max_seq_len:
        max_new_tokens = max(1, model.cfg.max_seq_len - len(ids))
    x = torch.tensor([ids], device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with amp(device):
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0, top_k=0, eos_id=TOKENIZER.eos_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    generated = out[0, len(ids):].tolist()
    return TOKENIZER.decode(generated), generated, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="public_eval50_data/gsm8k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    if args.limit < 0 or args.max_new_tokens <= 0:
        raise ValueError("invalid evaluation budget")

    manifest, rows = read_holdout(Path(args.data))
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(4)
    torch.manual_seed(968_800)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()

    cases = []
    total_seconds = 0.0
    for i, row in enumerate(rows):
        text, token_ids, seconds = generate(model, device, prompt_for(row["question"]), args.max_new_tokens)
        raw_pred, pred, extraction = extract_prediction(text)
        ref = canonical_number(row["final_answer"])
        correct = pred is not None and ref is not None and pred == ref
        eos = TOKENIZER.eos_id in token_ids
        total_seconds += seconds
        cases.append({
            "id": row["id"],
            "correct": correct,
            "reference": row["final_answer"],
            "prediction": raw_pred,
            "extraction": extraction,
            "eos": eos,
            "generated_tokens": len(token_ids),
            "seconds": seconds,
            "output": text,
        })
        if (i + 1) % 100 == 0:
            print(json.dumps({"completed": i + 1, "exact_pct": 100 * sum(x["correct"] for x in cases) / len(cases)}, ensure_ascii=False), flush=True)

    n = len(cases)
    report = {
        "benchmark": "GSM8K main test - Neral direct-answer harness",
        "definition": "External held-out model-only exact numeric generation diagnostic; not an official leaderboard score.",
        "dataset_manifest": manifest,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "parameters": model.parameter_report(),
        "step": payload.get("step"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "evaluated": n,
        "summary": {
            "exact": sum(x["correct"] for x in cases),
            "exact_pct": 100 * sum(x["correct"] for x in cases) / max(1, n),
            "extracted_pct": 100 * sum(x["prediction"] is not None for x in cases) / max(1, n),
            "answer_marker_pct": 100 * sum(x["extraction"] == "answer_marker" for x in cases) / max(1, n),
            "eos_pct": 100 * sum(x["eos"] for x in cases) / max(1, n),
            "mean_generated_tokens": statistics.mean(x["generated_tokens"] for x in cases) if cases else 0.0,
            "total_seconds": total_seconds,
            "mean_tokens_per_second": sum(x["generated_tokens"] for x in cases) / max(total_seconds, 1e-9),
        },
        "solve_used": False,
        "helios_used": False,
        "external_model_used": False,
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"saved": str(out), **report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
