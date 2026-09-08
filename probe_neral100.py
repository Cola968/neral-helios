from __future__ import annotations

"""Short, non-training CUDA probe for Neral-100M Helix.

It never saves weights and never touches training corpora. It measures architecture viability,
prefill/decode speed and one backward pass with activation checkpointing.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from architectures100 import HELIX_100_EXPECTED_PARAMETERS, validate_helix100_contract
from model100 import HELIX_100, Neral100M


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup: int = 1, repeats: int = 3):
    for _ in range(warmup):
        fn()
    sync()
    vals = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); sync(); vals.append(time.perf_counter() - t)
    return min(vals), sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="neral100_probe.json")
    ap.add_argument("--prefill-lengths", default="256,512,1024")
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--backward-seq", type=int, default=512)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Helix probe requires BF16 CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = Neral100M(HELIX_100).to(device=device, dtype=torch.bfloat16)
    if model.parameter_report()["total"] != HELIX_100_EXPECTED_PARAMETERS:
        raise ValueError("parameter contract mismatch")
    model.eval()

    report = {
        "architecture": validate_helix100_contract(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "bf16": True,
        "training_performed": False,
        "weights_saved": False,
        "prefill": {},
    }

    for seq in [int(x) for x in args.prefill_lengths.split(",") if x.strip()]:
        if not 0 < seq <= HELIX_100.max_seq_len:
            raise ValueError("invalid probe sequence")
        x = torch.randint(0, HELIX_100.vocab_size, (1, seq), device=device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            best, mean = timed(lambda: model(x)[0], warmup=1, repeats=3)
        report["prefill"][str(seq)] = {
            "best_seconds": best,
            "mean_seconds": mean,
            "best_tokens_per_second": seq / best,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del x

    prompt_len = min(512, HELIX_100.max_seq_len - args.decode_steps - 1)
    prompt = torch.randint(0, HELIX_100.vocab_size, (1, prompt_len), device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, cache = model.forward_cached(prompt)
        sync(); start = time.perf_counter()
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(args.decode_steps):
            logits, cache = model.forward_cached(token, cache)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sync(); seconds = time.perf_counter() - start
    report["decode"] = {
        "prompt_length": prompt_len,
        "steps": args.decode_steps,
        "seconds": seconds,
        "tokens_per_second": args.decode_steps / seconds,
        "cache_heads": cache[0][0].size(1),
        "cache_length": cache[0][0].size(-2),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del prompt, logits, cache, token

    model.train(); model.set_gradient_checkpointing(True)
    seq = args.backward_seq
    x = torch.randint(0, HELIX_100.vocab_size, (1, seq), device=device)
    y = torch.randint(0, HELIX_100.vocab_size, (1, seq), device=device)
    torch.cuda.reset_peak_memory_stats(); model.zero_grad(set_to_none=True); sync(); start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward(); sync(); seconds = time.perf_counter() - start
    finite = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    report["backward_checkpointed"] = {
        "sequence_length": seq,
        "batch_size": 1,
        "loss": float(loss.detach()),
        "finite": finite,
        "seconds": seconds,
        "tokens_per_second": seq / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if not finite:
        raise FloatingPointError("non-finite Helix probe backward")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
