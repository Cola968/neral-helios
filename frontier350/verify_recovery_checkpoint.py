from __future__ import annotations

"""Fail-closed verification for bounded Arctus recovery pilots."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-sha256", required=True)
    ap.add_argument("--min-tokens", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args = ap.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"checkpoint missing: {path}")
    if len(args.tokenizer_sha256) != 64:
        raise SystemExit("invalid tokenizer digest length")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise SystemExit("invalid token bounds")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    processed = int(ck.get("processed_tokens", 0))
    if not args.min_tokens <= processed <= args.max_tokens:
        raise SystemExit(
            f"checkpoint token count outside bounds: {processed:,} not in "
            f"[{args.min_tokens:,}, {args.max_tokens:,}]"
        )

    digest = str(ck.get("tokenizer_sha256", ""))
    if digest not in ("", args.tokenizer_sha256):
        raise SystemExit("checkpoint tokenizer digest differs from locked corpus tokenizer")

    cfg = ck.get("config")
    if not isinstance(cfg, dict):
        raise SystemExit("checkpoint config missing")
    required = {"vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads"}
    if not required.issubset(cfg):
        raise SystemExit("checkpoint config incomplete")

    print(
        f"checkpoint_ok processed_tokens={processed} tokenizer_sha256={args.tokenizer_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
