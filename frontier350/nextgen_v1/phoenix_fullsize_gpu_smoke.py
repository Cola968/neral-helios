"""Real full-size Neral 350M GPU engineering smoke with synthetic owned text.

This is NOT a quality-training candidate. It exists to prove that the exact SHA-pinned
historical 350M weights can execute real BF16 forward/backward/AdamW/checkpointing on
the owner's RTX 4070. Original weights are read-only; output is create-only and separate.
No private corpus, holdout, prompts, or review artifacts are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
EXPECTED_PARAMETERS = 349_825_152
SEQ = 128
ACCUM = 2
OPTIMIZER_STEPS = 4
NEW_TOKENS = SEQ * ACCUM * OPTIMIZER_STEPS
LR = 1e-5
MIN_FREE_DISK_BYTES = 20 * 1024**3

# Authored for this engineering smoke; no external corpus is copied.
SYNTHETIC_DIALOGUES = (
    "User: Hallo!\nAssistant: Hallo! Wie kann ich dir helfen?\n",
    "User: What is two plus three?\nAssistant: Two plus three equals five.\n",
    "User: Erkläre kurz die Gravitation.\nAssistant: Gravitation ist die Anziehung zwischen Massen.\n",
    "User: Name the capital of Germany.\nAssistant: The capital of Germany is Berlin.\n",
    "User: Was ist Wasser?\nAssistant: Wasser besteht aus Wasserstoff und Sauerstoff.\n",
    "User: Continue politely.\nAssistant: Of course. What would you like to know?\n",
    "User: Antworte knapp.\nAssistant: Verstanden.\n",
    "User: What are you?\nAssistant: I am Helios, an experimental Neral language model.\n",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(torch, obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_stream(tokenizer, bos: int, eos: int) -> list[int]:
    stream: list[int] = []
    # Keep enough deterministic data for distinct training and validation windows.
    while len(stream) < 20_000:
        for text in SYNTHETIC_DIALOGUES:
            stream.append(bos)
            stream.extend(tokenizer.encode(text).ids)
            stream.append(eos)
    return stream


def batch_from_stream(torch, stream: list[int], offset: int, device):
    limit = len(stream) - SEQ - 1
    if limit <= 0:
        raise RuntimeError("synthetic stream is too short")
    start = offset % limit
    chunk = stream[start:start + SEQ + 1]
    x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
    return x, y


def eval_loss(torch, model, stream: list[int], device) -> float:
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(2):
            x, y = batch_from_stream(torch, stream, 11_000 + i * 211, device)
            _, loss = model(x, y)
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError("non-finite smoke validation loss")
            vals.append(value)
    model.train()
    return sum(vals) / len(vals)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit execution and trust in the owner's SHA-pinned checkpoint are required")

    from helios350_preview import trusted_asset
    from phoenix_continuity_cpu_probe import require_pinned_checkpoint

    source = require_pinned_checkpoint(args.source_checkpoint)
    root = args.assets_root.resolve(strict=True)
    tokenizer_path = trusted_asset(
        root, root / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA
    )
    out = args.out
    if not out.is_absolute() or out.is_symlink() or out.exists():
        raise ValueError("smoke output must be a new absolute non-symlink path")
    if source == out or source.is_relative_to(out) or out.is_relative_to(source.parent):
        raise ValueError("smoke output must be separate from the historical checkpoint")
    if os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize < MIN_FREE_DISK_BYTES:
        raise RuntimeError("less than 20 GiB free for full-size smoke checkpoint")

    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()

    import torch
    from tokenizers import Tokenizer
    from model350 import FRONTIER_350, Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights

    gpu_guard.require_cuda_ready(torch)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("BOS/EOS mismatch")
    stream = make_stream(tokenizer, int(bos), int(eos))

    historical = _checked_historical_weights(
        source, torch, Neral350M, FRONTIER_350.to_dict()
    )
    device = torch.device("cuda")
    torch.manual_seed(35023)
    torch.cuda.manual_seed_all(35023)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(True)
    if model.parameter_report()["total"] != EXPECTED_PARAMETERS:
        raise RuntimeError("locked parameter count changed")
    model.load_state_dict(historical["model"], strict=True)
    del historical

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True
    )
    if optimizer.state_dict()["state"]:
        raise RuntimeError("fresh smoke optimizer unexpectedly has inherited state")

    before = eval_loss(torch, model, stream, device)
    out.mkdir(parents=False, exist_ok=False)
    metrics = out / "metrics.jsonl"
    start = time.monotonic()
    processed = 0
    model.train()

    for step in range(1, OPTIMIZER_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        weighted = torch.zeros((), dtype=torch.float32, device=device)
        for micro in range(ACCUM):
            x, y = batch_from_stream(
                torch, stream, (step * ACCUM + micro) * (SEQ + 37), device
            )
            _, loss = model(x, y)
            if loss is None or not math.isfinite(float(loss)):
                raise FloatingPointError("non-finite training loss")
            (loss / ACCUM).backward()
            weighted.add_(loss.detach().float() / ACCUM)
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if not math.isfinite(float(grad)):
            raise FloatingPointError("non-finite gradient norm")
        gpu_guard.require_safe_runtime_temperature()
        optimizer.step()
        torch.cuda.synchronize()
        gpu_guard.require_safe_runtime_temperature()
        processed += SEQ * ACCUM
        row = {
            "event": "fullsize_engineering_smoke_step",
            "step": step,
            "new_tokens": processed,
            "loss": float(weighted),
            "grad_norm": float(grad),
            "elapsed_s": time.monotonic() - start,
        }
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        print(json.dumps(row, sort_keys=True), flush=True)

    after = eval_loss(torch, model, stream, device)
    checkpoint = out / "smoke_resume.pt"
    state = {
        "schema": "phoenix_fullsize_gpu_smoke_v1",
        "classification": "engineering_smoke_not_quality_candidate",
        "source_sha256": SOURCE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "config": FRONTIER_350.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "new_tokens": processed,
        "optimizer_steps": OPTIMIZER_STEPS,
        "seq": SEQ,
        "grad_accum": ACCUM,
        "lr": LR,
        "synthetic_data_only": True,
    }
    atomic_torch_save(torch, state, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    source_unchanged = sha256_file(source) == SOURCE_SHA
    summary = {
        "event": "phoenix_fullsize_gpu_smoke_finished",
        "classification": "engineering_smoke_not_quality_candidate",
        "parameters": EXPECTED_PARAMETERS,
        "new_tokens": processed,
        "optimizer_steps": OPTIMIZER_STEPS,
        "synthetic_validation_loss_before": before,
        "synthetic_validation_loss_after": after,
        "wall_seconds": time.monotonic() - start,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "checkpoint_sha256": checkpoint_sha,
        "source_unchanged": source_unchanged,
        "quality_improvement_verified": False,
        "private_training_data_used": False,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not source_unchanged:
        raise RuntimeError("historical source checkpoint changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
