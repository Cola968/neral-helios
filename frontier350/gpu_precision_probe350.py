"""Tiny, fail-closed CUDA precision probe while Roblox is running.

This is NOT Arctus model training or checkpoint inference. It allocates only
small synthetic parameter tensors and reports bfloat16-vs-float32 AdamW
rounding. Never reads checkpoint bytes, datasets or tokenizer assets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROBE_NAME = "synthetic_cuda_optimizer_precision_only"


def parse_gpu_memory_csv(value: str) -> tuple[int, int]:
    """One GPU only; refuse unknown / N/A telemetry rather than guessing."""
    if not isinstance(value, str):
        raise ValueError("GPU telemetry must be text")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one GPU memory row")
    match = re.fullmatch(r"(\d+)\s*,\s*(\d+)", lines[0])
    if match is None:
        raise ValueError("unparseable NVIDIA VRAM telemetry")
    total, free = (int(match.group(i)) for i in (1, 2))
    if total < 1 or not 0 <= free <= total:
        raise ValueError("inconsistent NVIDIA VRAM telemetry")
    return total, free


def assess_free_vram(total_mib: int, free_mib: int, *,
                     min_free_mib: int = 4096) -> tuple[bool, str]:
    """Never oversubscribe a shared gaming GPU using a guessed free value."""
    if (type(total_mib) is not int or type(free_mib) is not int
        or type(min_free_mib) is not int or total_mib <= 0
        or free_mib < 0 or free_mib > total_mib or min_free_mib <= 0):
        raise ValueError("invalid VRAM admission inputs")
    if free_mib < min_free_mib:
        return False, "insufficient_free_vram"
    return True, "admitted"


def _sample_adam_updates(torch, dtype, initial: float, steps: int = 8) -> dict:
    """Measure quantized AdamW changes for 4096 constant synthetic weights."""
    param = torch.nn.Parameter(torch.full((4096,), initial, dtype=dtype, device="cuda"))
    before = param.detach().clone()
    opt = torch.optim.AdamW([param], lr=1.5e-5, betas=(.9, .95),
                             weight_decay=.03, eps=1e-8, fused=True)
    for _ in range(steps):
        param.grad = torch.full_like(param, 0.01)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    changed = (param.detach() != before).float().mean().item()
    mean_shift = (param.detach().float() - before.float()).abs().mean().item()
    del opt, param, before
    return {"changed_fraction":float(changed), "mean_abs_shift":float(mean_shift)}


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-free-mib", type=int, default=4096)
    ap.add_argument("--max-torch-reserved-mib", type=int, default=256)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError("do not overwrite an earlier GPU probe")
    report = {"probe":PROBE_NAME, "training_started":False,
              "real_model_loaded":False, "checkpoint_touched":False,
              "status":"skipped", "reason":"not_authorized"}
    # Explicit second guard in addition to the GitHub workflow admission.
    if os.environ.get("NERAL_GPU_PROBE_ALLOW") != "1":
        out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
        return 0
    if args.min_free_mib < 4096 or not 1 <= args.max_torch_reserved_mib <= 256:
        raise ValueError("the admission policy cannot be relaxed via CLI")
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            report["reason"]="cuda_bf16_unavailable"
        else:
            # Calling CUDA mem_get_info can initialize its context. Account for
            # this with a 4 GiB safety headroom, not optimistic free-VRAM claims.
            free_bytes,total_bytes=torch.cuda.mem_get_info(0)
            total_mib,free_mib=total_bytes//2**20,free_bytes//2**20
            admitted,reason=assess_free_vram(total_mib,free_mib,
                                             min_free_mib=args.min_free_mib)
            report["observed_total_mib"]=total_mib
            report["observed_free_mib_before"]=free_mib
            if not admitted:
                report["reason"]=reason
            else:
                report["device"]=torch.cuda.get_device_name(0)
                torch.cuda.reset_peak_memory_stats(0)
                cases={}
                for precision in ("bfloat16","float32"):
                    dtype=getattr(torch,precision)
                    cases[precision]={}
                    for initial,label in ((0.02,"linear_like"),(1.0,"rmsnorm_like")):
                        cases[precision][label]=_sample_adam_updates(torch,dtype,initial)
                peak_alloc=torch.cuda.max_memory_allocated(0)//2**20
                peak_reserved=torch.cuda.max_memory_reserved(0)//2**20
                report.update(
                    status="completed" if peak_reserved <= args.max_torch_reserved_mib else "budget_exceeded",
                    reason="synthetic_only" if peak_reserved <= args.max_torch_reserved_mib else "reserved_vram_exceeded",
                    cases=cases,
                    pytorch_peak_allocated_mib=peak_alloc,
                    pytorch_peak_reserved_mib=peak_reserved,
                    observed_free_mib_after=torch.cuda.mem_get_info(0)[0]//2**20,
                )
    except (ImportError, RuntimeError, OSError) as error:
        # A hardware failure must be visible, not silently called a success.
        report.update(status="failed",reason=type(error).__name__,detail=str(error)[:350])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report["status"] in {"completed","skipped"} else 2


if __name__ == "__main__":
    sys.exit(main())
