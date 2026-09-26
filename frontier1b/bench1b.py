from __future__ import annotations

"""Fail-closed BF16 forward/backward benchmark for Nova 1B.

This is an architecture feasibility benchmark, not a training-quality benchmark. It does
not use an optimizer so optimizer-memory experiments remain a separate attribution axis.
"""

import argparse
import json
import time

import torch

from architecture1b import CANDIDATES, NOVA_1B
from model1b import make_model


def run_case(candidate: str, seq: int, batch: int, warmup: int, steps: int, compile_model: bool) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    if candidate not in CANDIDATES:
        raise KeyError(candidate)
    cfg = CANDIDATES[candidate]
    if seq > cfg.max_seq_len:
        raise ValueError("sequence length exceeds architecture contract")

    torch.manual_seed(1_000_350)
    torch.cuda.manual_seed_all(1_000_350)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = make_model(candidate, checkpoint_blocks=True).cuda().to(dtype=torch.bfloat16)
    model.train()
    param_actual = sum(p.numel() for p in model.parameters())
    param_expected = cfg.parameter_breakdown()["total"]
    if param_actual != param_expected:
        raise RuntimeError(f"parameter mismatch {param_actual} != {param_expected}")

    if compile_model:
        model = torch.compile(model, mode="default", fullgraph=False)

    x = torch.randint(0, 16_384, (batch, seq), device="cuda", dtype=torch.long)
    y = torch.randint(0, 16_384, (batch, seq), device="cuda", dtype=torch.long)

    def iteration() -> float:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite or missing loss")
        loss.backward()
        return float(loss.detach())

    losses = []
    for _ in range(warmup):
        losses.append(iteration())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        losses.append(iteration())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = batch * seq * steps
    return {
        "suite": "nova1b-bf16-fwd-bwd-v1",
        "candidate": candidate,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seq": seq,
        "batch": batch,
        "steps": steps,
        "compile": compile_model,
        "checkpoint_blocks": True,
        "parameters": param_actual,
        "loss_last": losses[-1],
        "finite": all(torch.isfinite(torch.tensor(v)).item() for v in losses),
        "elapsed_s": elapsed,
        "tokens_per_s_fwd_bwd": tokens / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "optimizer_included": False,
        "notes": [
            "This number is not end-to-end training throughput because no optimizer step is included.",
            "Promotion requires an optimizer-inclusive matched pilot after the architecture fits safely.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=NOVA_1B.name, choices=sorted(CANDIDATES))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-peak-gib", type=float, default=11.75)
    args = ap.parse_args()
    report = run_case(args.candidate, args.seq, args.batch, args.warmup, args.steps, args.compile)
    print(json.dumps(report, indent=2))
    if report["peak_reserved_gib"] > args.max_peak_gib:
        raise SystemExit(
            f"Nova VRAM gate failed: {report['peak_reserved_gib']:.2f} GiB > {args.max_peak_gib:.2f} GiB"
        )


if __name__ == "__main__":
    main()
