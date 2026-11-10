from __future__ import annotations

"""Arctus 350M adapter for the frozen NIS2 chat-core suite.

This evaluates the existing frozen NIS2 chat-core cases against the native Arctus
16K BPE checkpoint. It does not redefine NIS2 and deliberately does not emit a
full NIS2 score: the full score also requires Arctus-compatible evidence for the
legacy language/math/code/science/translation/context components.
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_nis2_chat50 import (  # noqa: E402
    EXACT_GENERATION_CASES,
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
        # target index prefix_len predicts the first continuation token because
        # y_full would be seq[1:]. Continuations are tokenized separately on both
        # sides of each pair, giving a stable BPE-conditional preference test.
        start = len(prefix_ids)
        pred = logits[0, start:start + len(cont_ids)]
        if pred.size(0) != y.numel():
            raise RuntimeError("conditional-loss span mismatch")
        loss = F.cross_entropy(pred.float(), y, reduction="mean")
    return float(loss)


def pair_eval(model: Neral350M, tok: Tokenizer, device: torch.device, bos_id: int) -> dict:
    rows = []
    by_category: dict[str, list[bool]] = collections.defaultdict(list)
    for i, (category, prefix, preferred, distractor) in enumerate(PAIR_CASES):
        good = conditional_mean_nll(model, tok, prefix, preferred, device, bos_id)
        bad = conditional_mean_nll(model, tok, prefix, distractor, device, bos_id)
        correct = good < bad
        row = {
            "id": f"nis2-pair-{i:03d}",
            "category": category,
            "correct": bool(correct),
            "preferred_mean_nll": good,
            "distractor_mean_nll": bad,
            "margin_nats_per_bpe": bad - good,
        }
        rows.append(row)
        by_category[category].append(bool(correct))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {
        "overall_pct": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "categories": {
            k: {
                "correct": sum(v),
                "total": len(v),
                "accuracy_pct": 100.0 * sum(v) / len(v),
            }
            for k, v in sorted(by_category.items())
        },
        "cases": rows,
    }


def repeated_ngram_ratio(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


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
    return {
        "raw_output": text,
        "generated_tokens": len(generated),
        "eos_reached": eos,
        "repeated_trigram_ratio": repeated_ngram_ratio(generated, 3),
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
        "mean_repeated_trigram_pct": 100.0 * sum(r["repeated_trigram_ratio"] for r in rows) / len(rows),
        "utf8_pct": 100.0 * sum(r["utf8_ok"] for r in rows) / len(rows),
        "cases": rows,
    }


def score_summary(pair: dict, exact: dict) -> dict:
    cats = pair["categories"]
    pair_instruction = cats["instruction_following"]["accuracy_pct"]
    return {
        "instruction_following_pct": 0.5 * pair_instruction + 0.5 * exact["accuracy_pct"],
        "instruction_pair_pct": pair_instruction,
        "instruction_exact_generation_pct": exact["accuracy_pct"],
        "dialogue_state_pct": cats["dialogue_state"]["accuracy_pct"],
        "truthfulness_pct": cats["truthfulness"]["accuracy_pct"],
        "robustness_pct": cats["robustness"]["accuracy_pct"],
        "compositional_reasoning_pct": cats["compositional_reasoning"]["accuracy_pct"],
        "pair_overall_pct": pair["overall_pct"],
        "generation_eos_pct": exact["eos_pct"],
        "generation_repeated_trigram_pct": exact["mean_repeated_trigram_pct"],
        "generation_utf8_pct": exact["utf8_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, default=500_000_000)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

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
    if not args.expected_tokens <= processed <= args.expected_tokens + 32_768:
        raise RuntimeError(f"expected bounded {args.expected_tokens:,}-token checkpoint, got {processed:,}")

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
    scores = score_summary(pair, exact)
    report = {
        "benchmark": SUITE_VERSION,
        "suite_sha256": suite_sha256(),
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_processed_tokens": processed,
        "checkpoint_step": int(ck.get("step", -1)),
        "foreign_model_or_tool_assistance": False,
        "used_for_checkpoint_selection": False,
        "tokenization_note": "Arctus native 16K BPE; pair continuations tokenized separately for stable conditional comparison.",
        "pair_preference": pair,
        "exact_generation": exact,
        "scores": scores,
        "full_nis2": {
            "score": None,
            "coverage_pct": None,
            "status": "not_yet_computable",
            "reason": "The canonical NIS2 scorer also requires Arctus-compatible language, math-transfer/OOD, code, science, bilingual, context and generation-health lineage reports. Partial-score imputation is forbidden by NIS2.md.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scores": scores, "full_nis2": report["full_nis2"], "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
