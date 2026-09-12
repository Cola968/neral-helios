from __future__ import annotations

"""Fixed runner for arctus-recovery-dev-v1 using Neral's actual special-token contract."""

import argparse
import collections
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model350 import FRONTIER_350, Neral350M
from eval500m_diagnostic import conditional_mean_nll
from eval_recovery_dev import CASES, SUITE_VERSION
from recovery_identity import file_sha, object_sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, default=1_000_000_000)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    device = torch.device("cuda")
    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise RuntimeError("tokenizer vocabulary mismatch")
    bos_id = tok.token_to_id("<|bos|>")
    if bos_id is None:
        raise RuntimeError("missing Neral <|bos|> token")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", 0))
    if ck.get("tokenizer_sha256") != file_sha(args.tokenizer):
        raise RuntimeError("checkpoint/tokenizer digest mismatch")
    if not args.expected_tokens <= processed <= args.expected_tokens + 32_768:
        raise RuntimeError(f"expected bounded checkpoint near {args.expected_tokens:,}, got {processed:,}")

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    rows = []
    for case_id, cat, prefix, preferred, distractors in CASES:
        options = [preferred, *distractors]
        losses = [conditional_mean_nll(model, tok, prefix, x, device, int(bos_id)) for x in options]
        if not all(math.isfinite(value) for value in losses):
            raise FloatingPointError(f"non-finite option scores: {case_id}")
        chosen = min(range(len(losses)), key=losses.__getitem__)
        ok = chosen == 0
        by_cat[cat].append(ok)
        row = {
            "id": case_id,
            "category": cat,
            "correct": ok,
            "preferred": preferred,
            "chosen": options[chosen],
            "preferred_nll": losses[0],
            "best_distractor_nll": min(losses[1:]),
            "margin_nats_per_bpe": min(losses[1:]) - losses[0],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    categories = {
        k: {"correct": sum(v), "total": len(v), "accuracy_pct": 100.0 * sum(v) / len(v)}
        for k, v in sorted(by_cat.items())
    }
    total_correct = sum(int(x["correct"]) for x in rows)
    summary = {
        "checkpoint_sha256": file_sha(args.checkpoint),
        "tokenizer_sha256": file_sha(args.tokenizer),
        "protocol_sha256": object_sha({"cases": CASES, "implementation": {
            p: file_sha(Path(__file__).parent / p) for p in ('model350.py', 'eval500m_diagnostic.py', 'eval_recovery_dev_v2.py')}}),
        "suite": SUITE_VERSION,
        "checkpoint_processed_tokens": processed,
        "architecture": "frontier350-v1",
        "cases": len(rows),
        "overall_accuracy_pct": 100.0 * total_correct / len(rows),
        "categories": categories,
        "purpose": "development-only recovery diagnostic; not final audit/public benchmark",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
