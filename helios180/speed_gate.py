from __future__ import annotations
import argparse, json, time
import torch
from .config import Helios180Config
from .model import Helios180ForCausalLM

def bench(batch: int, seq: int, steps: int, use_compile: bool, native_gqa: bool):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = Helios180Config(native_gqa=native_gqa)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = Helios180ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.train()
    if use_compile:
        try:
            model = torch.compile(model, mode="default")
        except Exception as exc:
            del model
            return {"ok": False, "error": f"compile:{type(exc).__name__}:{exc}"}
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9,0.95),
                            weight_decay=0.1, fused=True)
    x = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda")
    inp, tgt = x[:, :-1], x[:, 1:]
    times = []
    losses = []
    try:
        for i in range(steps + 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out = model(inp, tgt)
            out.loss.backward()
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= 2:
                times.append(dt)
                losses.append(float(out.cross_entropy.detach()))
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "oom": True, "batch": batch, "seq": seq}
    tokens = batch * seq
    mean_s = sum(times) / len(times)
    return {
        "ok": True, "batch": batch, "seq": seq, "compile": use_compile,
        "native_gqa": native_gqa, "tokens_per_second": tokens / mean_s,
        "step_seconds": mean_s, "loss": losses[-1],
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "params": cfg.parameter_count(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, nargs="+", default=[2,4,8])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    rows = []
    for native in (False, True):
        for b in args.batches:
            r = bench(b, args.seq_len, args.steps, args.compile, native)
            rows.append(r)
            print(json.dumps(r), flush=True)
    good = [r for r in rows if r.get("ok")]
    if not good:
        raise SystemExit("no successful benchmark")
    best = max(good, key=lambda r: r["tokens_per_second"])
    summary = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "best": best,
        "all": rows,
        "gate_20k": best["tokens_per_second"] >= 20_000,
        "target_30k": best["tokens_per_second"] >= 30_000,
    }
    print("HELIOS180_SUMMARY=" + json.dumps(summary), flush=True)
    with open("helios180_speed_gate.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
