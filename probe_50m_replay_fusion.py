from __future__ import annotations

"""RTX probe for eliminating Chat-V2's second Transformer pass on replay steps.

Current Chat V2 computes chat forward/backward and, every replay interval, a second LM replay
forward before one shared backward. This probe compares that baseline with one batch-concatenated
forward using exactly the same model weights and loss coefficients. Only equal-width chat/replay
shapes are tested. The production trainer is not changed by this file.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from fastpath50 import upgrade_runtime_inplace
from mixed_loss50 import mixed_chat_replay_loss
from model50 import Neral50M
from train_instruct50 import assistant_loss
from tokenizer import ByteTokenizer

EXPECTED_PARAMS = 50_012_976
SEED = 968_571
TOKENIZER = ByteTokenizer()


def make_model(device):
    torch.manual_seed(SEED)
    model = Neral50M(FIFTY).to(device)
    upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise AssertionError("parameter drift")
    return model


def data(seq: int, chat_batch: int, replay_batch: int, device):
    g = torch.Generator(device=device).manual_seed(SEED + seq)
    cx = torch.randint(0, FIFTY.vocab_size, (chat_batch, seq), device=device, generator=g)
    cy = torch.full((chat_batch, seq), -100, dtype=torch.long, device=device)
    # Approximate multi-turn SFT: supervise the final quarter plus terminal EOS.
    start = max(1, seq * 3 // 4)
    cy[:, start:] = torch.randint(0, FIFTY.vocab_size, (chat_batch, seq - start), device=device, generator=g)
    cy[:, -1] = TOKENIZER.eos_id
    rx = torch.randint(0, FIFTY.vocab_size, (replay_batch, seq), device=device, generator=g)
    ry = torch.randint(0, FIFTY.vocab_size, (replay_batch, seq), device=device, generator=g)
    return cx, cy, rx, ry


def dual_loss(model, cx, cy, rx, ry, eos_weight, replay_scale):
    chat = assistant_loss(model, cx, cy, eos_weight)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(rx)
    replay = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ry.reshape(-1))
    return chat + replay_scale * replay, chat, replay


def numerical_check(seq: int, chat_batch: int, replay_batch: int, eos_weight: float, replay_scale: float):
    device = torch.device("cuda")
    model_a = make_model(device).eval()
    model_b = copy.deepcopy(model_a).eval()
    cx, cy, rx, ry = data(seq, chat_batch, replay_batch, device)
    with torch.no_grad():
        a, ac, ar = dual_loss(model_a, cx, cy, rx, ry, eos_weight, replay_scale)
        b, bc, br = mixed_chat_replay_loss(
            model_b, cx, cy, rx, ry, eos_weight=eos_weight, replay_scale=replay_scale
        )
    result = {
        "seq": seq,
        "total_abs_delta": float((a - b).abs()),
        "chat_abs_delta": float((ac - bc).abs()),
        "replay_abs_delta": float((ar - br).abs()),
        "dual_total": float(a),
        "mixed_total": float(b),
    }
    del model_a, model_b, cx, cy, rx, ry
    torch.cuda.empty_cache()
    return result


def bench(mode: str, seq: int, chat_batch: int, replay_batch: int, warmup: int, repeats: int):
    device = torch.device("cuda")
    model = make_model(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.025, fused=True)
    cx, cy, rx, ry = data(seq, chat_batch, replay_batch, device)
    eos_weight = 2.25
    replay_scale = ((1.0 - .88) / .88) * 4

    def step():
        opt.zero_grad(set_to_none=True)
        if mode == "dual":
            total, _, _ = dual_loss(model, cx, cy, rx, ry, eos_weight, replay_scale)
        elif mode == "mixed":
            total, _, _ = mixed_chat_replay_loss(
                model, cx, cy, rx, ry, eos_weight=eos_weight, replay_scale=replay_scale
            )
        else:
            raise ValueError(mode)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=True)
        opt.step()
        return total

    try:
        for _ in range(warmup): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); loss = None
        for _ in range(repeats): loss = step()
        end.record(); end.synchronize(); seconds = start.elapsed_time(end) / 1000.0
        # Both modes consume the same useful training tokens per replay step.
        useful = repeats * seq * (chat_batch + replay_batch)
        return {
            "mode": mode, "seq": seq, "chat_batch": chat_batch, "replay_batch": replay_batch,
            "useful_tokens": useful, "seconds": seconds, "useful_tokens_per_second": useful / seconds,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(), "last_loss": float(loss.detach()),
        }
    finally:
        del model, opt, cx, cy, rx, ry
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/neral50_replay_fusion.json")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("replay fusion probe requires BF16 CUDA")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")

    shapes = [(512, 16, 8), (1024, 8, 4)]
    checks = [numerical_check(*shape, 2.25, ((1-.88)/.88)*4) for shape in shapes]
    rows = []
    for seq, cb, rb in shapes:
        for mode in ("dual", "mixed"):
            row = bench(mode, seq, cb, rb, args.warmup, args.repeats)
            rows.append(row); print(json.dumps(row), flush=True)
    speedups = {}
    for seq, _, _ in shapes:
        dual = next(r for r in rows if r["seq"] == seq and r["mode"] == "dual")
        mixed = next(r for r in rows if r["seq"] == seq and r["mode"] == "mixed")
        speedups[str(seq)] = mixed["useful_tokens_per_second"] / dual["useful_tokens_per_second"]

    report = {
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "parameters": EXPECTED_PARAMS,
        "numerical_checks": checks, "rows": rows, "speedup_mixed_over_dual": speedups,
        "promotion_rule": "Require small numerical deltas, measured speedup and unchanged downstream validation before trainer integration.",
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
