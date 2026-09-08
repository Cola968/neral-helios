from __future__ import annotations

"""Hardware-in-the-loop BF16 training probe for Neral 50M and Orion.

This is a short architecture/performance experiment, not capability training. It measures real
forward+backward+fused-AdamW throughput on the self-hosted Neral GPU runner and records OOMs
instead of hiding them. Random inputs are used so no benchmark/training corpus is consumed.
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fastpath50 import fuse_swiglu_inplace, linear_gemm_report
from model50 import Neral50M
from orion100 import NeralOrion, ORION_A3, ORION_A4, architecture_report


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_model(name: str):
    if name == "50m-legacy":
        return Neral50M(), {"architecture": "legacy-50M"}
    if name == "50m-fused-mlp":
        model = Neral50M()
        fuse_swiglu_inplace(model)
        return model, linear_gemm_report(model)
    if name == "orion-a3":
        return NeralOrion(ORION_A3), architecture_report(ORION_A3)
    if name == "orion-a4":
        return NeralOrion(ORION_A4), architecture_report(ORION_A4)
    raise ValueError(name)


def one_probe(name: str, seq: int, token_budget: int, warmup: int, steps: int, checkpoint: bool):
    cleanup()
    batch = max(1, token_budget // seq)
    result = {
        "name": name,
        "seq": seq,
        "batch": batch,
        "tokens_per_step": batch * seq,
        "gradient_checkpointing": checkpoint,
    }
    try:
        model, report = make_model(name)
        if hasattr(model, "set_gradient_checkpointing"):
            model.set_gradient_checkpointing(checkpoint)
        model = model.to(device="cuda", dtype=torch.bfloat16).train()
        params = sum(p.numel() for p in model.parameters())
        result.update(parameters=params, architecture_report=report)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
        x = torch.randint(0, 260, (batch, seq), device="cuda", dtype=torch.long)
        y = torch.randint(0, 260, (batch, seq), device="cuda", dtype=torch.long)
        torch.cuda.reset_peak_memory_stats()

        samples = []
        total = warmup + steps
        for i in range(total):
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                samples.append(elapsed)

        mean_s = sum(samples) / len(samples)
        result.update(
            status="ok",
            mean_step_seconds=mean_s,
            tokens_per_second=(batch * seq) / mean_s,
            tokens_per_hour=(batch * seq) / mean_s * 3600.0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            final_loss=float(loss.detach()),
        )
        del x, y, opt, model, logits, loss
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc).splitlines()[0])
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup()
    return result


def best_by_model(rows):
    out = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        old = out.get(row["name"])
        if old is None or row["tokens_per_second"] > old["tokens_per_second"]:
            out[row["name"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf_v2_reports")
    ap.add_argument("--seq", nargs="*", type=int, default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--token-budget", type=int, default=8192, help="Approximate tokens/update; batch=floor(budget/seq)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    if min(args.seq) <= 0 or args.token_budget <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("invalid probe budget")

    torch.manual_seed(100_2026)
    torch.cuda.manual_seed_all(100_2026)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows = []
    for seq in args.seq:
        # 50M checkpointing is intentionally not introduced into the legacy lineage here.
        for name in ("50m-legacy", "50m-fused-mlp"):
            rows.append(one_probe(name, seq, args.token_budget, args.warmup, args.steps, False))
        for name in ("orion-a3", "orion-a4"):
            rows.append(one_probe(name, seq, args.token_budget, args.warmup, args.steps, False))
            # At >=2K also measure the memory-saving checkpoint path.
            if seq >= 2048:
                rows.append(one_probe(name, seq, args.token_budget, args.warmup, args.steps, True))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    best = best_by_model(rows)
    baseline = best.get("50m-legacy", {}).get("tokens_per_second")
    comparisons = {}
    if baseline:
        for name, row in best.items():
            comparisons[name] = {
                "best_tokens_per_second": row["tokens_per_second"],
                "speed_ratio_vs_best_50m_legacy": row["tokens_per_second"] / baseline,
                "speed_percent_vs_best_50m_legacy": (row["tokens_per_second"] / baseline - 1.0) * 100.0,
                "seq": row["seq"],
                "batch": row["batch"],
                "gradient_checkpointing": row["gradient_checkpointing"],
            }

    payload = {
        "definition": "BF16 forward+backward+gradient-clip+fused-AdamW, random byte-token IDs; architecture performance only, not quality evidence.",
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python_platform": platform.platform(),
        },
        "settings": vars(args),
        "rows": rows,
        "best_by_model": best,
        "comparisons": comparisons,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "PERF_V2.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")

    md = [
        "# Neral performance probe v2",
        "",
        payload["definition"],
        "",
        f"GPU: `{payload['system']['gpu']}`  ",
        f"PyTorch: `{payload['system']['torch']}` / CUDA `{payload['system']['cuda']}`",
        "",
        "| Model | Seq | Batch | Checkpoint | tok/s | Peak alloc GiB | Status |",
        "|---|---:|---:|:---:|---:|---:|---|",
    ]
    for r in rows:
        tps = f"{r.get('tokens_per_second', 0):,.0f}" if r.get("status") == "ok" else "-"
        gib = r.get("peak_allocated_bytes", 0) / 2**30
        md.append(f"| {r['name']} | {r['seq']} | {r['batch']} | {r['gradient_checkpointing']} | {tps} | {gib:.2f} | {r['status']} |")
    md += ["", "## Best measured speed ratios", ""]
    for name, c in comparisons.items():
        md.append(f"- **{name}**: {c['speed_ratio_vs_best_50m_legacy']:.2f}x vs best legacy-50M probe ({c['speed_percent_vs_best_50m_legacy']:+.1f}%).")
    md += [
        "",
        "> A speed result does not promote a model. Orion still requires stability and frozen quality/capability gates.",
    ]
    (out / "PERF_V2.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(json.dumps({"comparisons": comparisons, "report": str(out / 'PERF_V2.md')}, indent=2), flush=True)


if __name__ == "__main__":
    main()
