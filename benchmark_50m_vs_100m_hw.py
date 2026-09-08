from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from model50 import Neral50M
from model100 import Neral100M


def sync() -> None:
    torch.cuda.synchronize()


def mib(n: int) -> float:
    return n / (1024 ** 2)


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def time_prefill(model, seq: int, repeats: int = 3) -> dict:
    idx = torch.randint(0, model.cfg.vocab_size, (1, seq), device="cuda")
    with torch.inference_mode():
        model(idx)
        sync()
        reset_peak()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(idx)
            sync()
            times.append(time.perf_counter() - start)
    best = min(times)
    return {
        "seq": seq,
        "batch": 1,
        "best_seconds": best,
        "mean_seconds": sum(times) / len(times),
        "best_tokens_per_second": seq / best,
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
    }


def time_decode(model, prompt_len: int = 512, steps: int = 32) -> dict:
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device="cuda")
    with torch.inference_mode():
        logits, cache = model.forward_cached(prompt)
        sync()
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        # One warmup continuation, excluded from timing.
        logits, cache = model.forward_cached(nxt, cache)
        sync()
        reset_peak()
        start = time.perf_counter()
        for _ in range(steps):
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            logits, cache = model.forward_cached(nxt, cache)
        sync()
        elapsed = time.perf_counter() - start
    return {
        "prompt_len": prompt_len,
        "steps": steps,
        "seconds": elapsed,
        "tokens_per_second": steps / elapsed,
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        "cache_len": int(cache[0][0].size(-2)),
        "cache_heads": int(cache[0][0].size(1)),
    }


def time_backward(model, seq: int = 512, batch: int = 8, repeats: int = 2) -> dict:
    model.train()
    idx = torch.randint(0, model.cfg.vocab_size, (batch, seq), device="cuda")
    targets = torch.randint(0, model.cfg.vocab_size, (batch, seq), device="cuda")

    # Warmup.
    model.zero_grad(set_to_none=True)
    _, loss = model(idx, targets)
    loss.backward()
    sync()

    reset_peak()
    times = []
    losses = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        _, loss = model(idx, targets)
        loss.backward()
        sync()
        times.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))

    model.eval()
    best = min(times)
    return {
        "seq": seq,
        "batch": batch,
        "tokens_per_step": seq * batch,
        "best_seconds": best,
        "mean_seconds": sum(times) / len(times),
        "best_tokens_per_second": (seq * batch) / best,
        "loss_mean": sum(losses) / len(losses),
        "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
    }


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def model_static(model) -> dict:
    params = sum(p.numel() for p in model.parameters())
    cfg = model.cfg
    kv_heads = int(getattr(cfg, "n_kv_heads", cfg.n_heads))
    head_dim = int(cfg.head_dim)
    kv_4k_bytes = int(cfg.n_layers * 2 * kv_heads * 4096 * head_dim * 2)
    return {
        "parameters": params,
        "millions": params / 1e6,
        "layers": int(cfg.n_layers),
        "d_model": int(cfg.d_model),
        "heads": int(cfg.n_heads),
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "ffn_hidden": int(cfg.ffn_hidden),
        "max_seq_len": int(cfg.max_seq_len),
        "kv_cache_bf16_4096_mib": mib(kv_4k_bytes),
    }


def run_one(name: str, ctor) -> dict:
    reset_peak()
    model = ctor().to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    out = {
        "name": name,
        "static": model_static(model),
        "prefill": {},
    }
    for seq in (256, 512, 1024, 2048, 4096):
        if seq <= model.cfg.max_seq_len:
            out["prefill"][str(seq)] = time_prefill(model, seq)
    out["decode"] = time_decode(model, 512, 32)
    out["backward"] = time_backward(model, 512, 8, 2)
    del model
    reset_peak()
    return out


def make_comparison(m50: dict, m100: dict) -> dict:
    s50, s100 = m50["static"], m100["static"]
    c = {
        "parameters_pct": pct(s100["parameters"], s50["parameters"]),
        "context_pct": pct(s100["max_seq_len"], s50["max_seq_len"]),
        "kv_cache_4k_pct": pct(s100["kv_cache_bf16_4096_mib"], s50["kv_cache_bf16_4096_mib"]),
        "kv_efficiency_per_parameter_pct": pct(
            s50["kv_cache_bf16_4096_mib"] / s50["parameters"],
            s100["kv_cache_bf16_4096_mib"] / s100["parameters"],
        ),
        "decode_tps_pct": pct(m100["decode"]["tokens_per_second"], m50["decode"]["tokens_per_second"]),
        "decode_vram_pct": pct(m100["decode"]["peak_allocated_mib"], m50["decode"]["peak_allocated_mib"]),
        "backward_tps_pct": pct(m100["backward"]["best_tokens_per_second"], m50["backward"]["best_tokens_per_second"]),
        "backward_vram_pct": pct(m100["backward"]["peak_allocated_mib"], m50["backward"]["peak_allocated_mib"]),
        "prefill_tps_pct": {},
        "prefill_vram_pct": {},
    }
    for seq in sorted(set(m50["prefill"]) & set(m100["prefill"]), key=int):
        c["prefill_tps_pct"][seq] = pct(
            m100["prefill"][seq]["best_tokens_per_second"],
            m50["prefill"][seq]["best_tokens_per_second"],
        )
        c["prefill_vram_pct"][seq] = pct(
            m100["prefill"][seq]["peak_allocated_mib"],
            m50["prefill"][seq]["peak_allocated_mib"],
        )
    return c


def render_md(report: dict) -> str:
    a = report["models"]["50M"]
    b = report["models"]["100M"]
    c = report["comparison_100m_vs_50m_pct"]
    rows = [
        "# Neral 50M vs Helix 100M — RTX 4070 hardware benchmark",
        "",
        "Random-initialized architecture benchmark only; this does **not** compare intelligence or NIS.",
        "Both models use the same GPU, PyTorch/CUDA runtime, BF16 dtype, batch settings and benchmark code.",
        "",
        "| Metric | 50M | 100M | 100M vs 50M |",
        "|---|---:|---:|---:|",
        f"| Parameters | {a['static']['millions']:.3f}M | {b['static']['millions']:.3f}M | {c['parameters_pct']:+.1f}% |",
        f"| Max context | {a['static']['max_seq_len']} | {b['static']['max_seq_len']} | {c['context_pct']:+.1f}% |",
        f"| BF16 KV cache @4K | {a['static']['kv_cache_bf16_4096_mib']:.1f} MiB | {b['static']['kv_cache_bf16_4096_mib']:.1f} MiB | {c['kv_cache_4k_pct']:+.1f}% |",
        f"| KV efficiency / parameter | baseline | — | {c['kv_efficiency_per_parameter_pct']:+.1f}% |",
        f"| Cached decode | {a['decode']['tokens_per_second']:.2f} tok/s | {b['decode']['tokens_per_second']:.2f} tok/s | {c['decode_tps_pct']:+.1f}% |",
        f"| Decode peak VRAM | {a['decode']['peak_allocated_mib']:.1f} MiB | {b['decode']['peak_allocated_mib']:.1f} MiB | {c['decode_vram_pct']:+.1f}% |",
        f"| Backward 512x8 | {a['backward']['best_tokens_per_second']:.1f} tok/s | {b['backward']['best_tokens_per_second']:.1f} tok/s | {c['backward_tps_pct']:+.1f}% |",
        f"| Backward peak VRAM | {a['backward']['peak_allocated_mib']:.1f} MiB | {b['backward']['peak_allocated_mib']:.1f} MiB | {c['backward_vram_pct']:+.1f}% |",
    ]
    for seq in (256, 512, 1024, 2048, 4096):
        k = str(seq)
        if k in a["prefill"] and k in b["prefill"]:
            rows.append(
                f"| Prefill {seq} | {a['prefill'][k]['best_tokens_per_second']:.1f} tok/s | "
                f"{b['prefill'][k]['best_tokens_per_second']:.1f} tok/s | {c['prefill_tps_pct'][k]:+.1f}% |"
            )
    rows += [
        "",
        "## Notes",
        "",
        "- Lower KV-cache and VRAM percentages are better; higher throughput percentages are better.",
        "- The benchmark saves no weights and performs no optimizer step.",
        "- Because Helix currently uses GQA, decode speed is sensitive to the PyTorch attention-kernel path on this Windows build.",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="hw_benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 support required")

    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)

    report = {
        "protocol": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
            "random_seed": 20260908,
            "training_performed": False,
            "optimizer_step_performed": False,
            "weights_saved": False,
        },
        "models": {},
    }
    report["models"]["50M"] = run_one("Neral 50M", Neral50M)
    report["models"]["100M"] = run_one("Neral 100M Helix", Neral100M)
    report["comparison_100m_vs_50m_pct"] = make_comparison(report["models"]["50M"], report["models"]["100M"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "NERAL_50M_VS_100M_HW.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_md(report)
    (out_dir / "NERAL_50M_VS_100M_HW.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    print(md)


if __name__ == "__main__":
    main()
