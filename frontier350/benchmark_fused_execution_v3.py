from __future__ import annotations

"""Numerical + throughput probe for Neral 350M fused execution.

The probe converts one promoted canonical checkpoint into model350_fast's QKV/SwiGLU
execution layout. It never saves or promotes model weights. Canonical and fused models
receive identical deterministic batches, train only the same last four blocks, and are
compared after backward. A later guarded canary is still required before adoption.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from model350 import FRONTIER_350, Neral350M
from model350_fast import Neral350MFast, convert_unfused_state_dict
from protection_v9 import configure_trainable
from recovery_identity import file_sha

SEQ = 512
BATCH = 4
PARAMETERS = 349_825_152


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover_anchor(assets: Path) -> tuple[Path, str, int]:
    state = assets / "frontier350_scale_v3_state.json"
    if state.is_file():
        obj = read_json(state)
        path = Path(obj["safe_anchor_path"])
        if path.is_file() and file_sha(path) == obj["safe_anchor_sha256"]:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if int(ck["processed_tokens"]) == int(obj["safe_anchor_tokens"]):
                return path, obj["safe_anchor_sha256"], int(obj["safe_anchor_tokens"])
    rows = []
    for path in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            rows.append((int(ck["processed_tokens"]), path, file_sha(path)))
        except Exception:
            pass
    if not rows:
        raise RuntimeError("no promoted anchor found")
    tokens, path, sha = max(rows, key=lambda x: x[0])
    return path, sha, tokens


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def grad_vector(named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    chunks = []
    for _, p in named:
        if p.grad is None:
            raise RuntimeError("missing gradient")
        chunks.append(p.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks)


def canonicalize_fast_grad(named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    grads = {name: p.grad.detach().float().cpu() for name, p in named}
    out = []
    for layer in range(FRONTIER_350.n_layers - 4, FRONTIER_350.n_layers):
        p = f"blocks.{layer}."
        out.append(grads[p + "n1.weight"].reshape(-1))
        qkv = grads[p + "attn.qkv.weight"]
        qn = FRONTIER_350.d_model
        kvn = FRONTIER_350.kv_dim
        q, k, v = torch.split(qkv, (qn, kvn, kvn), dim=0)
        out.extend([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
        out.append(grads[p + "attn.o.weight"].reshape(-1))
        out.append(grads[p + "attn.qn.weight"].reshape(-1))
        out.append(grads[p + "attn.kn.weight"].reshape(-1))
        out.append(grads[p + "n2.weight"].reshape(-1))
        gu = grads[p + "ffn.gate_up.weight"]
        gate, up = torch.chunk(gu, 2, dim=0)
        out.extend([gate.reshape(-1), up.reshape(-1)])
        out.append(grads[p + "ffn.down.weight"].reshape(-1))
    return torch.cat(out)


def make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(36421)
    whole = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ + 1), generator=gen, dtype=torch.long)
    return whole[:, :-1].to(device), whole[:, 1:].to(device)


def timed_steps(model, named, x, y, warmup: int, repeats: int) -> dict:
    run = torch.compile(model, fullgraph=True, dynamic=False, mode="default")
    rates = []
    losses = []
    for i in range(warmup + repeats):
        for _, p in named:
            p.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = run(x, y)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            rates.append((BATCH * SEQ) / elapsed)
            losses.append(float(loss.detach()))
    return {
        "median_tps": statistics.median(rates),
        "p10_tps": p10(rates),
        "peak_tps": max(rates),
        "loss_median": statistics.median(losses),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    a = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    if not 1 <= a.warmup <= 8 or not 4 <= a.repeats <= 24:
        raise ValueError("bounded probe counts required")

    anchor, sha, tokens = discover_anchor(Path(a.assets))
    ck = torch.load(anchor, map_location="cpu", weights_only=False)
    if ck["config"] != FRONTIER_350.to_dict():
        raise ValueError("architecture mismatch")

    dev = torch.device("cuda")
    torch.manual_seed(36420)
    canonical = Neral350M().to(dev, dtype=torch.bfloat16)
    canonical.load_state_dict(ck["model"], strict=True)
    fast = Neral350MFast().to(dev, dtype=torch.bfloat16)
    fast.load_state_dict(convert_unfused_state_dict(ck["model"]), strict=True)
    if canonical.parameter_report()["total"] != PARAMETERS or fast.parameter_report()["total"] != PARAMETERS:
        raise ValueError("parameter contract changed")
    c_named = configure_trainable(canonical, 4)
    f_named = configure_trainable(fast, 4)
    x, y = make_batch(dev)

    canonical.zero_grad(set_to_none=True)
    fast.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        c_logits, c_loss = canonical(x, y)
        f_logits, f_loss = fast(x, y)
    c_loss.backward()
    f_loss.backward()
    cg = grad_vector(c_named)
    fg = canonicalize_fast_grad(f_named)
    if cg.numel() != fg.numel():
        raise ValueError("gradient layouts differ")
    grad_rel = float((cg - fg).norm() / cg.norm().clamp_min(1e-12))
    grad_max = float((cg - fg).abs().max())
    logit_max = float((c_logits.detach().float() - f_logits.detach().float()).abs().max())
    loss_abs = abs(float(c_loss.detach()) - float(f_loss.detach()))

    del c_logits, f_logits, cg, fg
    canonical.zero_grad(set_to_none=True)
    fast.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    c_speed = timed_steps(canonical, c_named, x, y, a.warmup, a.repeats)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    f_speed = timed_steps(fast, f_named, x, y, a.warmup, a.repeats)

    speedup = f_speed["median_tps"] / c_speed["median_tps"]
    numerical_pass = bool(loss_abs <= 2e-3 and grad_rel <= 5e-3 and math.isfinite(grad_rel))
    memory_pass = bool(f_speed["peak_reserved_gib"] <= 11.0)
    report = {
        "schema": 1,
        "probe": "neral350-fused-execution-v3",
        "anchor": str(anchor),
        "anchor_sha256": sha,
        "anchor_tokens": tokens,
        "parameters": PARAMETERS,
        "same_architecture": True,
        "same_trainable_last_blocks": 4,
        "canonical": c_speed,
        "fused": f_speed,
        "speedup": speedup,
        "numerical": {
            "loss_abs_diff": loss_abs,
            "logit_max_abs_diff": logit_max,
            "gradient_relative_l2": grad_rel,
            "gradient_max_abs_diff": grad_max,
            "pass": numerical_pass,
        },
        "memory_pass": memory_pass,
        "eligible_for_canary": bool(numerical_pass and memory_pass and speedup > 1.0),
        "promotion_allowed": False,
        "requires_guarded_canary": True,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
