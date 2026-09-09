from __future__ import annotations

"""Hardware-in-the-loop benchmark for the next Neral-50M optimization layer.

This intentionally benchmarks isolated model compute with synthetic on-device byte IDs so host
input latency cannot hide attention/kernel improvements. Chat-V2 has a separate end-to-end probe.
Candidates must preserve the exact 50,012,976 trainable parameters and pass output-equivalence
checks before their throughput is considered.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from fastpath50 import upgrade_runtime_inplace
from model50 import Neral50M
from model50_ultra import use_exact_padded_attention_inplace, verify_runtime_equivalence

EXPECTED_PARAMS = 50_012_976
SEED = 968_570


def build_model(kind: str, device: torch.device):
    torch.manual_seed(SEED)
    model = Neral50M(FIFTY).to(device)
    upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
    if kind == "pad56_auto":
        pass
    elif kind == "pad64_auto":
        use_exact_padded_attention_inplace(model, padded_head_dim=64, backend="auto")
    elif kind == "pad64_flash":
        use_exact_padded_attention_inplace(model, padded_head_dim=64, backend="flash")
    elif kind == "pad64_efficient":
        use_exact_padded_attention_inplace(model, padded_head_dim=64, backend="efficient")
    else:
        raise ValueError(kind)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise AssertionError("candidate changed Neral-50M parameter count")
    return model


def one_step(model, opt, x, y, *, foreach_clip: bool):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True, foreach=foreach_clip
    )
    opt.step()
    return loss, grad


def benchmark(kind: str, seq: int, batch: int, *, foreach_clip: bool, warmup: int, repeats: int):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = build_model(kind, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.025, fused=True)
    gen = torch.Generator(device=device).manual_seed(SEED + seq + batch)
    x = torch.randint(0, FIFTY.vocab_size, (batch, seq), device=device, generator=gen)
    y = torch.randint(0, FIFTY.vocab_size, (batch, seq), device=device, generator=gen)
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup):
            one_step(model, opt, x, y, foreach_clip=foreach_clip)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_loss = None
        for _ in range(repeats):
            last_loss, _ = one_step(model, opt, x, y, foreach_clip=foreach_clip)
        end.record(); end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        tokens = seq * batch * repeats
        return {
            "kind": kind, "seq": seq, "batch": batch, "foreach_clip": foreach_clip,
            "tokens": tokens, "seconds": seconds, "tokens_per_second": tokens / seconds,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "last_loss": float(last_loss.detach()), "oom": False,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # Forced SDPA backends legitimately raise RuntimeError when unsupported. Record rather than
        # silently falling back; a backend gets no speed claim without actually running.
        return {
            "kind": kind, "seq": seq, "batch": batch, "foreach_clip": foreach_clip,
            "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
            "unsupported_or_failed": type(exc).__name__, "error": str(exc)[:500],
        }
    finally:
        del opt, model, x, y
        torch.cuda.empty_cache()


def compile_probe(seq: int, batch: int, warmup: int, repeats: int):
    if not hasattr(torch, "compile"):
        return {"supported": False, "reason": "torch.compile unavailable"}
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    eager = build_model("pad64_auto", device).train()
    try:
        compiled = torch.compile(eager, mode="reduce-overhead", fullgraph=False)
        opt = torch.optim.AdamW(compiled.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.025, fused=True)
        gen = torch.Generator(device=device).manual_seed(SEED + 991)
        x = torch.randint(0, FIFTY.vocab_size, (batch, seq), device=device, generator=gen)
        y = torch.randint(0, FIFTY.vocab_size, (batch, seq), device=device, generator=gen)
        for _ in range(warmup): one_step(compiled, opt, x, y, foreach_clip=True)
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record()
        for _ in range(repeats): one_step(compiled, opt, x, y, foreach_clip=True)
        b.record(); b.synchronize(); seconds = a.elapsed_time(b) / 1000.0
        return {"supported": True, "seq": seq, "batch": batch,
                "tokens_per_second": seq * batch * repeats / seconds, "seconds": seconds}
    except Exception as exc:
        return {"supported": False, "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}
    finally:
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/neral50_ultra_probe.json")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neral-50M ultra probe requires BF16 CUDA")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")

    # Numerical equivalence uses the same initialized base model. Backend=auto is required here so
    # unsupported forced backends do not turn a correctness test into a platform test.
    base = Neral50M(FIFTY).to("cuda").eval()
    tokens = torch.randint(0, FIFTY.vocab_size, (2, 128), device="cuda")
    equivalence = verify_runtime_equivalence(base, tokens, padded_head_dim=64, backend="auto")
    if not equivalence["allclose"]:
        raise RuntimeError(f"pad64 equivalence failed: {equivalence}")
    del base, tokens; torch.cuda.empty_cache()

    shapes = [(256, 32), (512, 16), (1024, 8), (2048, 4)]
    kinds = ("pad56_auto", "pad64_auto", "pad64_flash", "pad64_efficient")
    rows = []
    for seq, batch in shapes:
        for kind in kinds:
            row = benchmark(kind, seq, batch, foreach_clip=True, warmup=args.warmup, repeats=args.repeats)
            rows.append(row); print(json.dumps(row), flush=True)

    # Explicitly measure foreach clipping once against the default implementation. This keeps the
    # decision evidence-based; no semantic relaxation of nonfinite checking is included.
    clip_ab = []
    for foreach in (False, True):
        row = benchmark("pad64_auto", 1024, 8, foreach_clip=foreach, warmup=args.warmup, repeats=args.repeats)
        clip_ab.append(row); print(json.dumps({"clip_ab": row}), flush=True)

    compiled = compile_probe(1024, 8, args.warmup, args.repeats)
    print(json.dumps({"compile": compiled}), flush=True)
    usable = [r for r in rows if "tokens_per_second" in r]
    best_by_shape = {}
    for seq, batch in shapes:
        candidates = [r for r in usable if r["seq"] == seq and r["batch"] == batch]
        if candidates:
            best_by_shape[str(seq)] = max(candidates, key=lambda r: r["tokens_per_second"])
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "parameters": EXPECTED_PARAMS, "equivalence": equivalence, "candidates": rows,
        "best_by_shape": best_by_shape, "clip_foreach_ab": clip_ab, "torch_compile": compiled,
        "promotion_rule": "Only exact-weight candidates with equivalence=true and measured end-to-end speedup may enter Chat V2.",
        "no_claim": "Backend names, compile availability and theoretical kernel counts are not speed claims.",
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
