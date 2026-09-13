from __future__ import annotations

"""Numerical/speed probe for replay fusion on Neral 350M.

The production trainer is not modified. This probe uses one unchanged checkpoint and the
same three deterministic replay-shaped batches. It compares:
  A) three batch-4 replay forwards/backwards;
  B) one concatenated batch-12 forward/backward.
Weights are never stepped or saved. A passing result is only permission for a later
own-weight canary, never permission to promote a checkpoint.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from model350 import FRONTIER_350, Neral350M
from protection_v9 import configure_trainable
from recovery_identity import file_sha

SEQ = 512
BATCH = 4
DOMAINS = 3
VOCAB = FRONTIER_350.vocab_size


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


def flatten_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in params])


def run_separate(run, params, batches) -> tuple[torch.Tensor, torch.Tensor]:
    for p in params:
        p.grad = None
    losses = []
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = run(x, y)
        (loss / DOMAINS).backward()
        losses.append(loss.detach().float())
    return torch.stack(losses).mean(), flatten_grads(params)


def run_fused(run, params, batches) -> tuple[torch.Tensor, torch.Tensor]:
    for p in params:
        p.grad = None
    x = torch.cat([a for a, _ in batches], dim=0)
    y = torch.cat([b for _, b in batches], dim=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = run(x, y)
    loss.backward()
    return loss.detach().float(), flatten_grads(params)


def timed(fn, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    if not 5 <= a.repeats <= 30 or not 1 <= a.warmup <= 8:
        raise ValueError("bounded probe limits violated")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")

    torch.manual_seed(46001)
    torch.cuda.manual_seed_all(46001)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    assets = Path(a.assets).resolve()
    anchor, anchor_sha, anchor_tokens = discover_anchor(assets)
    ck = torch.load(anchor, map_location="cpu", weights_only=False)
    if ck["config"] != FRONTIER_350.to_dict():
        raise RuntimeError("architecture mismatch")

    model = Neral350M().to("cuda", dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    named = configure_trainable(model, 4)
    params = [p for _, p in named]
    run = torch.compile(model, fullgraph=True, dynamic=False, mode="default")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(46002)
    batches = []
    for _ in range(DOMAINS):
        z = torch.randint(0, VOCAB, (BATCH, SEQ + 1), generator=gen, device="cuda", dtype=torch.long)
        batches.append((z[:, :-1], z[:, 1:]))

    # Compile/warm both static shapes. No optimizer step occurs.
    for _ in range(a.warmup):
        run_separate(run, params, batches)
        run_fused(run, params, batches)
    torch.cuda.synchronize()

    sep_loss, sep_grad = run_separate(run, params, batches)
    fused_loss, fused_grad = run_fused(run, params, batches)
    diff = fused_grad - sep_grad
    sep_norm = float(torch.linalg.vector_norm(sep_grad))
    diff_norm = float(torch.linalg.vector_norm(diff))
    rel_l2 = diff_norm / max(sep_norm, 1e-12)
    max_abs = float(diff.abs().max())
    loss_abs = abs(float(fused_loss) - float(sep_loss))

    sep_times = timed(lambda: run_separate(run, params, batches), a.repeats)
    fused_times = timed(lambda: run_fused(run, params, batches), a.repeats)
    sep_med = statistics.median(sep_times)
    fused_med = statistics.median(fused_times)
    speedup = sep_med / fused_med

    report = {
        "schema": 1,
        "probe": "neral350-replay-fusion-v2",
        "anchor_path": str(anchor),
        "anchor_sha256": anchor_sha,
        "anchor_tokens": anchor_tokens,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in params),
        "shape_separate": [DOMAINS, BATCH, SEQ],
        "shape_fused": [DOMAINS * BATCH, SEQ],
        "loss_separate": float(sep_loss),
        "loss_fused": float(fused_loss),
        "loss_abs_delta": loss_abs,
        "gradient_relative_l2_delta": rel_l2,
        "gradient_max_abs_delta": max_abs,
        "separate_median_seconds": sep_med,
        "fused_median_seconds": fused_med,
        "speedup": speedup,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "numerical_candidate": bool(math.isfinite(rel_l2) and rel_l2 <= 0.005 and loss_abs <= 0.002),
        "speed_candidate": bool(speedup > 1.0),
        "promotion_allowed": False,
        "next_gate": "If numerical_candidate and speed_candidate are true, implement behind an opt-in trainer flag and run a same-data canary plus unchanged validation/dev/bilingual gates.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
