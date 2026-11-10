from __future__ import annotations

"""Evaluate a Frontier350 checkpoint on a fixed local BPE validation partition."""

import argparse
import json
import math
from pathlib import Path

import torch

from model350 import FRONTIER_350, Neral350M
from train350_compiled import PersistentMMapPool, evaluate
from recovery_identity import file_sha, object_sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=351337)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 required")
    if args.seq <= 0 or args.seq > FRONTIER_350.max_seq_len:
        raise SystemExit("invalid sequence length")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", 0))

    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    pool = PersistentMMapPool(Path(args.valid_data), FRONTIER_350.vocab_size)
    val = evaluate(model, pool, seq=args.seq, batch=args.batch, batches=args.batches, device=device, seed=args.seed)
    result = {
        "checkpoint_sha256": file_sha(args.checkpoint),
        "tokenizer_sha256": ck.get("tokenizer_sha256"),
        "protocol_sha256": object_sha({"seq": args.seq, "batch": args.batch, "batches": args.batches, "seed": args.seed,
            "shards": [(p.name, file_sha(p)) for p in pool.paths],
            "implementation": {p: file_sha(Path(__file__).parent / p) for p in ('model350.py', 'train350_compiled.py', 'eval_validation_loss350.py')}}),
        "schema": 1,
        "architecture": "frontier350-v1",
        "checkpoint_processed_tokens": processed,
        "validation_loss": val,
        "validation_perplexity": math.exp(val),
        "seq": args.seq,
        "batch": args.batch,
        "batches": args.batches,
        "seed": args.seed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
