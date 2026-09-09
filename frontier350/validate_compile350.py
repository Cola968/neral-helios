from __future__ import annotations

"""Numerical/validation A/B gate for checkpoint-native TorchInductor execution.

Runs eager and compiled Neral350M sequentially from the same real checkpoint so the
12 GB RTX 4070 never has to hold both models at once. It compares deterministic
validation losses, sampled logits, one training loss, and representative gradients.
No optimizer step or checkpoint write is performed.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from model350 import FRONTIER_350, Neral350M

PARAMETERS = 349_825_152


class MMapPool:
    def __init__(self, root: Path):
        paths = sorted(root.glob("*.bin"))
        if not paths:
            raise FileNotFoundError(f"no *.bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]

    def cpu_batch(self, batch: int, seq: int, rng: random.Random):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            if len(a) < seq + 1:
                raise ValueError("validation shard too short")
            start = rng.randrange(0, len(a) - seq - 1)
            rows[i] = a[start:start + seq + 1]
        if rows.max(initial=0) >= FRONTIER_350.vocab_size:
            raise ValueError("token id outside locked vocabulary")
        whole = torch.from_numpy(rows.copy())
        return whole[:, :-1], whole[:, 1:]


def selected_gradient_fingerprint(model: Neral350M) -> dict[str, torch.Tensor]:
    wanted = (
        "tok_emb.weight",
        "blocks.0.attn.q.weight",
        "blocks.0.attn.o.weight",
        f"blocks.{FRONTIER_350.n_layers - 1}.ffn.down.weight",
        "norm.weight",
    )
    named = dict(model.named_parameters())
    out: dict[str, torch.Tensor] = {}
    for name in wanted:
        p = named[name]
        if p.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        flat = p.grad.detach().float().reshape(-1)
        # Fixed sparse samples cover different regions without copying huge gradients.
        n = flat.numel()
        idx = torch.tensor(sorted(set((0, n // 7, n // 3, n // 2, (5 * n) // 7, n - 1))), device=flat.device)
        out[name] = flat.index_select(0, idx).cpu()
    return out


def run_variant(state: dict, eval_batches_cpu, train_batch_cpu, *, compiled: bool,
                fullgraph: bool, seed: int) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    raw = Neral350M().to(device=device, dtype=torch.bfloat16)
    if raw.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("Arctus parameter contract changed")
    raw.load_state_dict(state, strict=True)
    raw.set_gradient_checkpointing(False)

    model = raw
    if compiled:
        model = torch.compile(
            raw, backend="inductor", mode="default", fullgraph=fullgraph, dynamic=False
        )

    raw.eval()
    eval_losses = []
    logit_samples = []
    with torch.no_grad():
        for x_cpu, y_cpu in eval_batches_cpu:
            x = x_cpu.to(device, non_blocking=False)
            y = y_cpu.to(device, non_blocking=False)
            logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("evaluation returned no loss")
            eval_losses.append(float(loss))
            # Compare fixed positions and vocabulary slices, not just scalar loss.
            s = torch.cat((
                logits[0, 0, :256],
                logits[0, -1, :256],
                logits[0, logits.size(1) // 2, -256:],
            )).detach().float().cpu()
            logit_samples.append(s)

    raw.train()
    raw.zero_grad(set_to_none=True)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    x_cpu, y_cpu = train_batch_cpu
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    _, train_loss = model(x, y)
    if train_loss is None:
        raise RuntimeError("training returned no loss")
    train_loss.backward()
    grads = selected_gradient_fingerprint(raw)
    train_loss_value = float(train_loss.detach())

    result = {
        "eval_losses": eval_losses,
        "eval_mean": sum(eval_losses) / len(eval_losses),
        "train_loss": train_loss_value,
        "logit_samples": [x.tolist() for x in logit_samples],
        "gradient_samples": {k: v.tolist() for k, v in grads.items()},
    }
    del model, raw
    torch.cuda.empty_cache()
    gc.collect()
    return result


def max_abs_nested(a, b) -> float:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return float((ta - tb).abs().max()) if ta.numel() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=35201)
    ap.add_argument("--fullgraph", action="store_true")
    ap.add_argument("--loss-tol", type=float, default=0.02)
    ap.add_argument("--logit-tol", type=float, default=0.25)
    ap.add_argument("--grad-tol", type=float, default=0.02)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    if min(args.seq, args.batch, args.eval_batches) <= 0:
        raise ValueError("positive batch dimensions required")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture differs from locked Frontier 350M config")
    state = ck["model"]

    pool = MMapPool(Path(args.valid_data))
    rng = random.Random(args.seed)
    eval_batches_cpu = [pool.cpu_batch(args.batch, args.seq, rng) for _ in range(args.eval_batches)]
    train_batch_cpu = pool.cpu_batch(args.batch, args.seq, rng)

    eager = run_variant(
        state, eval_batches_cpu, train_batch_cpu, compiled=False,
        fullgraph=False, seed=args.seed,
    )
    compiled = run_variant(
        state, eval_batches_cpu, train_batch_cpu, compiled=True,
        fullgraph=args.fullgraph, seed=args.seed,
    )

    eval_loss_abs = max(abs(a - b) for a, b in zip(eager["eval_losses"], compiled["eval_losses"]))
    train_loss_abs = abs(eager["train_loss"] - compiled["train_loss"])
    logit_max_abs = max(
        max_abs_nested(a, b)
        for a, b in zip(eager["logit_samples"], compiled["logit_samples"])
    )
    grad_max_abs = 0.0
    grad_by_param = {}
    for name in eager["gradient_samples"]:
        d = max_abs_nested(eager["gradient_samples"][name], compiled["gradient_samples"][name])
        grad_by_param[name] = d
        grad_max_abs = max(grad_max_abs, d)

    passed = (
        eval_loss_abs <= args.loss_tol
        and train_loss_abs <= args.loss_tol
        and logit_max_abs <= args.logit_tol
        and grad_max_abs <= args.grad_tol
    )
    report = {
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_processed_tokens": int(ck.get("processed_tokens", -1)),
        "compile_mode": "default",
        "fullgraph": args.fullgraph,
        "thresholds": {
            "loss_abs": args.loss_tol,
            "logit_max_abs": args.logit_tol,
            "grad_max_abs": args.grad_tol,
        },
        "differences": {
            "eval_loss_max_abs": eval_loss_abs,
            "train_loss_abs": train_loss_abs,
            "logit_max_abs": logit_max_abs,
            "grad_max_abs": grad_max_abs,
            "grad_max_abs_by_parameter": grad_by_param,
        },
        "eager_eval_mean": eager["eval_mean"],
        "compiled_eval_mean": compiled["eval_mean"],
        "eager_train_loss": eager["train_loss"],
        "compiled_train_loss": compiled["train_loss"],
        "passed": passed,
        "promotion_allowed": passed,
        "note": "Numerical gate only. Long-run promotion additionally requires a bounded resumed burn-in with validation and sustained TPS checks.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
