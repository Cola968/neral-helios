from __future__ import annotations

"""One-step CPU training smoke test for full-size ORION A5.

Purpose: verify that the exact ~100M architecture can perform a real optimizer update without
CUDA or foreign weights. This is deliberately NOT capability training and must never be used
as quality evidence. It optionally pulls a tiny WikiText-2 sample from Hugging Face; if that
is unavailable it uses a small embedded DE/EN corpus.
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from orion100 import NeralOrion
from orion_a5 import ORION_A5, a5_report


def byte_ids(text: str) -> torch.Tensor:
    # Neral's byte vocabulary reserves 0..3; raw UTF-8 bytes map to 4..259.
    return torch.tensor([b + 4 for b in text.encode("utf-8")], dtype=torch.long)


def load_text(use_hf: bool) -> tuple[str, str]:
    fallback = (
        "Neral Helios ist ein eigenstaendig trainiertes Sprachmodell. "
        "Es soll Mathematik, Programmierung und naturwissenschaftliche Erklaerungen strukturiert loesen.\n"
        "A useful language model should preserve context, reason carefully, avoid repetition, and end answers cleanly.\n"
    )
    if not use_hf:
        return fallback * 8, "embedded-de-en"
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        parts = []
        for row in ds:
            text = (row.get("text") or "").strip()
            if text:
                parts.append(text)
            if sum(len(p) for p in parts) >= 4096:
                break
        joined = "\n".join(parts)
        if len(joined.encode("utf-8")) >= 256:
            return joined, "hf:Salesforce/wikitext:wikitext-2-raw-v1"
    except Exception as exc:
        print(f"HF dataset fallback: {type(exc).__name__}: {exc}")
    return fallback * 8, "embedded-de-en-fallback"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hf", action="store_true")
    args = ap.parse_args()
    if args.seq < 8 or args.seq > 128:
        raise ValueError("CPU smoke seq must stay in [8,128]")

    torch.manual_seed(20260909)
    text, source = load_text(args.hf)
    ids = byte_ids(text)
    if ids.numel() < args.seq + 1:
        raise RuntimeError("smoke corpus too short")
    x = ids[: args.seq].unsqueeze(0)
    y = ids[1 : args.seq + 1].unsqueeze(0)

    model = NeralOrion(ORION_A5).train()
    params = sum(p.numel() for p in model.parameters())
    if params != 99_900_672:
        raise RuntimeError(f"architecture drift: {params}")

    # SGD keeps the CPU smoke memory bounded. Real pretraining continues to use the project
    # optimizer contract; this job only proves forward/backward/update viability.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    with torch.no_grad():
        logits0, _ = model(x)
        before = F.cross_entropy(logits0.reshape(-1, ORION_A5.vocab_size), y.reshape(-1)).item()

    opt.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, ORION_A5.vocab_size), y.reshape(-1))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite pre-update loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()

    with torch.no_grad():
        logits1, _ = model(x)
        after = F.cross_entropy(logits1.reshape(-1, ORION_A5.vocab_size), y.reshape(-1)).item()

    elapsed = time.perf_counter() - start
    result = {
        "candidate": "ORION_A5",
        "parameters": params,
        "source": source,
        "seq": args.seq,
        "optimizer": "SGD-smoke-only",
        "loss_before": before,
        "loss_after": after,
        "loss_delta": after - before,
        "grad_norm_before_clip": grad_norm,
        "elapsed_seconds": elapsed,
        "finite": all(map(torch.isfinite, [torch.tensor(before), torch.tensor(after), torch.tensor(grad_norm)])),
        "foreign_weights": False,
        "quality_evidence": False,
        "architecture": a5_report(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
