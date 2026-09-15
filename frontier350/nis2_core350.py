from __future__ import annotations

"""Arctus 350M adapter for the frozen NIS2 chat-core v2 suite.

V2 expands the model-only diagnostic to 140 cases while preserving the original v1
benchmark file unchanged for historical comparisons. This adapter still does not emit a
full NIS2 score: canonical NIS2 requires the separate language/math/code/science/
bilingual/context lineage reports in addition to chat-core evidence.
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_nis2_chat140 import (  # noqa: E402
    EXACT_GENERATION_CASES,
    GENERATION_HEALTH_CASES,
    PAIR_CASES,
    SUITE_VERSION,
    suite_sha256,
)
from model350 import FRONTIER_350, Neral350M  # noqa: E402

PARAMETERS = 349_825_152


def _ids(tok: Tokenizer, text: str) -> list[int]:
    return [int(x) for x in tok.encode(text, add_special_tokens=False).ids]


def conditional_mean_nll(
    model: Neral350M,
    tok: Tokenizer,
    prefix: str,
    continuation: str,
    device: torch.device,
    bos_id: int,
) -> float:
    prefix_ids = _ids(tok, prefix)
    cont_ids = _ids(tok, continuation)
    if not cont_ids:
        raise ValueError("empty continuation")
    seq = [bos_id, *prefix_ids, *cont_ids]
    if len(seq) > FRONTIER_350.max_seq_len:
        raise ValueError("NIS2 case exceeds Arctus context")
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(cont_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x)
        start = len(prefix_ids)
        pred = logits[0, start:start + len(cont_ids)]
        if pred.size(0) != y.numel():
            raise RuntimeError("conditional-loss span mismatch")
        loss = F.cross_entropy(pred.float(), y, reduction="mean")
    return float(loss)


def pair_eval(model: Neral350M, tok: Tokenizer, device: torch.device, bos_id: int) -> dict:
    rows = []
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for i, (category, prefix, preferred, distractor) in enumerate(PAIR_CASES):
        good = conditional_mean_nll(model, tok, prefix, preferred, device, bos_id)
        bad = conditional_mean_nll(model, tok, prefix, distractor, device, bos_id)
        margin = bad - good
        correct = margin > 0.0
        row = {
            "id": f"nis2-v2-pair-{i:03d}",
            "category": category,
            "correct": bool(correct),
            "preferred_mean_nll": good,
            "distractor_mean_nll": bad,
            "margin_nats_per_bpe": margin,
        }
        rows.append(row)
        by_category[category].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    categories = {}
    for category, items in sorted(by_category.items()):
        margins = [float(x["margin_nats_per_bpe"]) for x in items]
        categories[category] = {
            "correct": sum(bool(x["correct"]) for x in items),
            "total": len(items),
            "accuracy_pct": 100.0 * sum(bool(x["correct"]) for x in items) / len(items),
            "mean_margin_nats_per_bpe": statistics.fmean(margins),
            "median_margin_nats_per_bpe": statistics.median(margins),
            "near_tie_pct": 100.0 * sum(abs(x) <= 0.05 for x in margins) / len(margins),
        }
    margins = [float(r["margin_nats_per_bpe"]) for r in rows]
    return {
        "overall_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "mean_margin_nats_per_bpe": statistics.fmean(margins),
        "median_margin_nats_per_bpe": statistics.median(margins),
        "near_tie_pct": 100.0 * sum(abs(x) <= 0.05 for x in margins) / len(margins),
        "categories": categories,
        "cases": rows,
    }


def repeated_ngram_ratio(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


def max_token_run(ids: list[int]) -> int:
    if not ids:
        return 0
    best = cur = 1
    for a, b in zip(ids, ids[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def generate(
    model: Neral350M,
    tok: Tokenizer,
    prompt: str,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_new: int,
) -> dict:
    seq = [bos_id, *_ids(tok, prompt)]
    generated: list[int] = []
    eos = False
    with torch.inference_mode():
        for _ in range(max_new):
            if len(seq) >= FRONTIER_350.max_seq_len:
                break
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            seq.append(nxt)
            if nxt == eos_id:
                eos = True
                break
            generated.append(nxt)
    text = tok.decode(generated, skip_special_tokens=True)
    unique_ratio = len(set(generated)) / max(1, len(generated))
    return {
        "raw_output": text,
        "generated_tokens": len(generated),
        "eos_reached": eos,
        "hit_max_new": (not eos and len(generated) >= max_new),
        "repeated_trigram_ratio": repeated_ngram_ratio(generated, 3),
        "unique_token_ratio": unique_ratio,
        "max_identical_token_run": max_token_run(generated),
        "utf8_ok": True,
    }


def exact_generation_eval(
    model: Neral350M,
    tok: Tokenizer,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_new: int,
) -> dict:
    rows = []
    for case_id, prompt, accepted in EXACT_GENERATION_CASES:
        result = generate(model, tok, prompt, device, bos_id, eos_id, max_new)
        text = result["raw_output"].strip()
        correct = text in accepted
        row = {"id": case_id, "correct": bool(correct), "accepted": accepted, "output": text, **result}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {
        "accuracy_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "eos_pct": 100.0 * sum(r["eos_reached"] for r in rows) / len(rows),
        "hit_max_new_pct": 100.0 * sum(r["hit_max_new"] for r in rows) / len(rows),
        "mean_repeated_trigram_pct": 100.0 * statistics.fmean(r["repeated_trigram_ratio"] for r in rows),
        "mean_unique_token_pct": 100.0 * statistics.fmean(r["unique_token_ratio"] for r in rows),
        "utf8_pct": 100.0 * sum(r["utf8_ok"] for r in rows) / len(rows),
        "cases": rows,
    }


def generation_health_eval(
    model: Neral350M,
    tok: Tokenizer,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_new: int,
) -> dict:
    rows = []
    for case_id, prompt in GENERATION_HEALTH_CASES:
        result = generate(model, tok, prompt, device, bos_id, eos_id, max_new)
        degenerative = result["repeated_trigram_ratio"] >= 0.25 or result["max_identical_token_run"] >= 4
        row = {"id": case_id, "degenerative": bool(degenerative), **result}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {
        "cases_count": len(rows),
        "eos_pct": 100.0 * sum(r["eos_reached"] for r in rows) / len(rows),
        "hit_max_new_pct": 100.0 * sum(r["hit_max_new"] for r in rows) / len(rows),
        "degenerative_pct": 100.0 * sum(r["degenerative"] for r in rows) / len(rows),
        "mean_repeated_trigram_pct": 100.0 * statistics.fmean(r["repeated_trigram_ratio"] for r in rows),
        "mean_unique_token_pct": 100.0 * statistics.fmean(r["unique_token_ratio"] for r in rows),
        "max_identical_token_run": max(r["max_identical_token_run"] for r in rows),
        "cases": rows,
    }


def score_summary(pair: dict, exact: dict, health: dict) -> dict:
    cats = pair["categories"]
    pair_instruction = cats["instruction_following"]["accuracy_pct"]
    category_scores = {k: v["accuracy_pct"] for k, v in cats.items()}
    chat_pair_macro = statistics.fmean(category_scores.values())
    return {
        "instruction_following_pct": 0.5 * pair_instruction + 0.5 * exact["accuracy_pct"],
        "instruction_pair_pct": pair_instruction,
        "instruction_exact_generation_pct": exact["accuracy_pct"],
        "pair_overall_pct": pair["overall_pct"],
        "pair_macro_pct": chat_pair_macro,
        "category_accuracy_pct": category_scores,
        "pair_near_tie_pct": pair["near_tie_pct"],
        "generation_exact_eos_pct": exact["eos_pct"],
        "generation_exact_repeated_trigram_pct": exact["mean_repeated_trigram_pct"],
        "generation_health_eos_pct": health["eos_pct"],
        "generation_health_degenerative_pct": health["degenerative_pct"],
        "generation_health_repeated_trigram_pct": health["mean_repeated_trigram_pct"],
        "generation_health_unique_token_pct": health["mean_unique_token_pct"],
        "generation_utf8_pct": exact["utf8_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, required=True)
    ap.add_argument("--token-tolerance", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--health-max-new", type=int, default=48)
    args = ap.parse_args()

    if args.expected_tokens <= 0 or args.token_tolerance < 0:
        raise ValueError("invalid expected token contract")
    if args.max_new <= 0 or args.health_max_new <= 0:
        raise ValueError("generation budgets must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    torch.manual_seed(92020)
    torch.cuda.manual_seed_all(92020)
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint = Path(args.checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", -1))
    if not args.expected_tokens <= processed <= args.expected_tokens + args.token_tolerance:
        raise RuntimeError(
            f"expected {args.expected_tokens:,}..{args.expected_tokens + args.token_tolerance:,} tokens, got {processed:,}"
        )

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise RuntimeError("tokenizer vocabulary mismatch")
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("missing BOS/EOS")

    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("parameter contract mismatch")

    pair = pair_eval(model, tok, device, int(bos_id))
    exact = exact_generation_eval(model, tok, device, int(bos_id), int(eos_id), args.max_new)
    health = generation_health_eval(model, tok, device, int(bos_id), int(eos_id), args.health_max_new)
    scores = score_summary(pair, exact, health)
    report = {
        "benchmark": SUITE_VERSION,
        "suite_sha256": suite_sha256(),
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_processed_tokens": processed,
        "checkpoint_step": int(ck.get("step", -1)),
        "case_counts": {
            "pair_preference": len(PAIR_CASES),
            "exact_generation": len(EXACT_GENERATION_CASES),
            "generation_health": len(GENERATION_HEALTH_CASES),
            "total": len(PAIR_CASES) + len(EXACT_GENERATION_CASES) + len(GENERATION_HEALTH_CASES),
        },
        "foreign_model_or_tool_assistance": False,
        "used_for_checkpoint_selection": False,
        "tokenization_note": "Arctus native 16K BPE; pair continuations tokenized separately for stable conditional comparison.",
        "pair_preference": pair,
        "exact_generation": exact,
        "generation_health": health,
        "scores": scores,
        "full_nis2": {
            "score": None,
            "coverage_pct": None,
            "status": "not_yet_computable",
            "reason": "Canonical full NIS2 still requires separate Arctus-compatible language, math-transfer/OOD, code, science, bilingual, context and generation-health lineage reports. Chat-core v2 is a larger diagnostic, not a substitute for those frozen components.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scores": scores, "case_counts": report["case_counts"], "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
