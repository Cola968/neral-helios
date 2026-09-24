from __future__ import annotations

"""Phase-C RTX-4070 training-path benchmark for Neral 600M candidates.

This is a real optimizer-inclusive synthetic benchmark. It does not measure model
quality. All candidates use identical seq length and effective tokens/update.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from architecture_search600 import CANDIDATES, PHASE_C
from model600 import make_model


def one_update(model, train_model, opt, *, seq: int, microbatch: int, grad_accum: int, vocab: int, device):
    opt.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(grad_accum):
        x = torch.randint(0, vocab, (microbatch, seq), device=device, dtype=torch.long)
        y = torch.randint(0, vocab, (microbatch, seq), device=device, dtype=torch.long)
        _, loss = train_model(x, y)
        if loss is None:
            raise RuntimeError("missing loss")
        (loss / grad_accum).backward()
        loss_sum += loss.detach().float()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss_sum / grad_accum, grad


def run_variant(candidate: str, *, seq: int, microbatch: int, tokens_per_update: int,
                warmup: int, measured: int, compile_mode: str, checkpoint_every: int) -> dict:
    if tokens_per_update % (seq * microbatch):
        raise ValueError("tokens/update must be divisible by seq*microbatch")
    grad_accum = tokens_per_update // (seq * microbatch)
    cfg = CANDIDATES[candidate]
    device = torch.device("cuda")
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(600_000 + hash((candidate, microbatch, checkpoint_every)) % 10_000)

    model = make_model(candidate, checkpoint_every=checkpoint_every).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1, fused=True)
    train_model = model
    if compile_mode != "none":
        train_model = torch.compile(model, backend="inductor", mode=compile_mode,
                                    fullgraph=False, dynamic=False)

    model.train()
    for _ in range(warmup):
        loss, grad = one_update(model, train_model, opt, seq=seq, microbatch=microbatch,
                                grad_accum=grad_accum, vocab=16_384, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    last_loss = last_grad = None
    for _ in range(measured):
        last_loss, last_grad = one_update(model, train_model, opt, seq=seq, microbatch=microbatch,
                                          grad_accum=grad_accum, vocab=16_384, device=device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "candidate": candidate,
        "config": cfg.to_dict(),
        "seq": seq,
        "microbatch": microbatch,
        "grad_accum": grad_accum,
        "tokens_per_update": tokens_per_update,
        "checkpoint_every": checkpoint_every,
        "compile_mode": compile_mode,
        "measured_updates": measured,
        "measured_tokens": measured * tokens_per_update,
        "elapsed_s": elapsed,
        "tokens_per_second": measured * tokens_per_update / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "last_loss": float(last_loss),
        "last_grad_norm": float(last_grad),
        "finite": bool(torch.isfinite(last_loss) and torch.isfinite(last_grad)),
        "gpu": torch.cuda.get_device_name(0),
    }
    del train_model, opt, model
    torch.cuda.empty_cache(); gc.collect()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=list(PHASE_C))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatches", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--tokens-per-update", type=int, default=16_384)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=4)
    ap.add_argument("--compile-mode", choices=("none", "default", "max-autotune"), default="default")
    ap.add_argument("--out", default="frontier600_report/phase_c_speed.json")
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    rows = []
    for candidate in args.candidates:
        if candidate not in CANDIDATES:
            raise KeyError(candidate)
        for mb in args.microbatches:
            # First try the fastest profile. If it OOMs, retry with selective block checkpointing.
            for checkpoint_every in (0, 2):
                try:
                    row = run_variant(candidate, seq=args.seq, microbatch=mb,
                                      tokens_per_update=args.tokens_per_update,
                                      warmup=args.warmup, measured=args.measured,
                                      compile_mode=args.compile_mode,
                                      checkpoint_every=checkpoint_every)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    rows.append({
                        "candidate": candidate, "seq": args.seq, "microbatch": mb,
                        "checkpoint_every": checkpoint_every, "oom": True,
                        "error": str(exc)[:500],
                    })
                    print(json.dumps(rows[-1]), flush=True)
                    torch.cuda.empty_cache(); gc.collect()
            else:
                continue

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "Neral600 Phase-C RTX-4070 benchmark",
        "quality_claim": False,
        "reference_arctus_350_peak_tps_measured_elsewhere": 19_239.0,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
