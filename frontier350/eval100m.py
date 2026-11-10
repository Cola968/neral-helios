from __future__ import annotations

"""Fixed-protocol evaluation for the bounded Arctus 350M 100M-token checkpoint.

This is a pretraining checkpoint evaluation, not a chat benchmark. It verifies checkpoint
integrity, computes held-out validation loss from a fixed RNG seed, and records deterministic
greedy completion probes plus generation-health metrics. It never modifies a checkpoint.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model350 import FRONTIER_350, Neral350M

PARAMETERS = 349_825_152

PROMPTS = [
    {"id": "en_capital", "prompt": "The capital of Germany is", "expected": ["Berlin", "berlin"]},
    {"id": "de_capital", "prompt": "Berlin ist die Hauptstadt von", "expected": ["Deutschland", "deutschland"]},
    {"id": "math_12x7", "prompt": "The result of 12 * 7 is", "expected": ["84"]},
    {"id": "de_science", "prompt": "Die Erde umkreist", "expected": ["Sonne", "sonne"]},
    {"id": "en_science", "prompt": "Water freezes at", "expected": ["0", "zero", "Celsius", "celsius"]},
    {"id": "code_add", "prompt": "def add(a, b):\n    return", "expected": ["a + b", "a+b", "a +b", "a+ b"]},
    {"id": "en_geometry", "prompt": "A triangle has", "expected": ["three", "3"]},
    {"id": "en_general", "prompt": "Machine learning is", "expected": []},
]


def load_arrays(root: Path) -> list[np.memmap]:
    paths = sorted(root.glob("*.bin"))
    if not paths:
        raise FileNotFoundError(f"no validation shards in {root}")
    return [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]


def fixed_validation(model: Neral350M, arrays: list[np.memmap], device: torch.device,
                     *, seed: int, seq: int, batch: int, batches: int) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            rows = np.empty((batch, seq + 1), dtype=np.int64)
            for i in range(batch):
                arr = rng.choice(arrays)
                if len(arr) < seq + 1:
                    raise ValueError("validation shard too short")
                start = rng.randrange(0, len(arr) - seq - 1)
                rows[i] = arr[start:start + seq + 1]
            if rows.max(initial=0) >= FRONTIER_350.vocab_size:
                raise ValueError("validation token outside locked vocabulary")
            whole = torch.from_numpy(rows).to(device, non_blocking=True)
            _, loss = model(whole[:, :-1], whole[:, 1:])
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            losses.append(float(loss))
    return sum(losses) / len(losses), losses


def repetition_ratio(ids: list[int], n: int = 4) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def distinct_ratio(ids: list[int], n: int = 2) -> float:
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(grams)) / len(grams)


def generate(model: Neral350M, tok: Tokenizer, prompt: str, device: torch.device,
             *, max_new: int, eos_id: int, bos_id: int) -> dict:
    body = tok.encode(prompt, add_special_tokens=False).ids
    ids = [bos_id, *map(int, body)]
    generated: list[int] = []
    eos = False
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new):
            if len(ids) >= FRONTIER_350.max_seq_len:
                break
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            ids.append(nxt)
            if nxt == eos_id:
                eos = True
                break
            generated.append(nxt)
    text = tok.decode(generated, skip_special_tokens=True)
    return {
        "text": text,
        "generated_tokens": len(generated),
        "eos_reached": eos,
        "repetition_4gram": repetition_ratio(generated, 4),
        "distinct_2gram": distinct_ratio(generated, 2),
        "empty": len(text.strip()) == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=350100)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_path = Path(args.checkpoint)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", -1))
    if not 100_000_000 <= processed <= 100_100_000:
        raise RuntimeError(f"expected bounded 100M checkpoint, got {processed}")

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise RuntimeError("tokenizer vocabulary mismatch")
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("missing BOS/EOS token")

    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("parameter contract mismatch")

    arrays = load_arrays(Path(args.valid_data))
    val_mean, val_losses = fixed_validation(
        model, arrays, device, seed=args.seed, seq=args.seq,
        batch=args.batch, batches=args.eval_batches,
    )

    completions = []
    for item in PROMPTS:
        g = generate(model, tok, item["prompt"], device, max_new=args.max_new,
                     eos_id=int(eos_id), bos_id=int(bos_id))
        hit = any(x in g["text"] for x in item["expected"]) if item["expected"] else None
        completions.append({**item, **g, "expected_hit": hit})
        print(json.dumps(completions[-1], ensure_ascii=False), flush=True)

    scored = [x for x in completions if x["expected_hit"] is not None]
    report = {
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_processed_tokens": processed,
        "checkpoint_step": int(ck.get("step", -1)),
        "checkpoint_best_validation": float(ck.get("best_validation", float("nan"))),
        "protocol": {
            "fixed_seed": args.seed,
            "validation_seq": args.seq,
            "validation_batch": args.batch,
            "validation_batches": args.eval_batches,
            "generation": "greedy deterministic completion, pretraining checkpoint; not chat/SFT evaluation",
            "max_new_tokens": args.max_new,
        },
        "fixed_validation_loss": val_mean,
        "fixed_validation_loss_min": min(val_losses),
        "fixed_validation_loss_max": max(val_losses),
        "all_validation_finite": all(math.isfinite(x) for x in val_losses),
        "completion_keyword_hit_rate": (sum(bool(x["expected_hit"]) for x in scored) / len(scored)) if scored else None,
        "eos_rate": sum(bool(x["eos_reached"]) for x in completions) / len(completions),
        "mean_repetition_4gram": sum(float(x["repetition_4gram"]) for x in completions) / len(completions),
        "mean_distinct_2gram": sum(float(x["distinct_2gram"]) for x in completions) / len(completions),
        "empty_completion_rate": sum(bool(x["empty"]) for x in completions) / len(completions),
        "completions": completions,
        "interpretation_guardrail": "Loss and completion probes at 100M tokens measure early pretraining progress only; they do not establish chatbot readiness or NIS2 capability.",
    }
    if not report["all_validation_finite"]:
        raise RuntimeError("non-finite fixed evaluation")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "completions"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
