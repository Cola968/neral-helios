from __future__ import annotations

"""Cheap rotating development screen for Scale V3.

This is NOT a promotion gate. It evaluates 5/20 cases per category (35 total) and is
only used to rank recovery candidates before the immutable full 140-case + bilingual
gates. Rotation prevents repeatedly optimizing against one tiny fixed subset.
"""

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


def select_cases(rotation: int):
    by_cat: dict[str, list[tuple]] = collections.defaultdict(list)
    for row in CASES:
        by_cat[row[1]].append(row)
    selected = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        if len(rows) != 20:
            raise RuntimeError(f"expected 20 cases for {cat}, got {len(rows)}")
        selected.extend(row for i, row in enumerate(rows) if i % 4 == rotation)
    if len(selected) != 35:
        raise RuntimeError(f"expected 35 smoke cases, got {len(selected)}")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, required=True)
    ap.add_argument("--rotation", type=int, choices=range(4), required=True)
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

    selected = select_cases(args.rotation)
    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    rows = []
    for case_id, cat, prefix, preferred, distractors in selected:
        options = [preferred, *distractors]
        losses = [conditional_mean_nll(model, tok, prefix, x, device, int(bos_id)) for x in options]
        if not all(math.isfinite(x) for x in losses):
            raise FloatingPointError(f"non-finite option scores: {case_id}")
        chosen = min(range(len(losses)), key=losses.__getitem__)
        ok = chosen == 0
        margin = min(losses[1:]) - losses[0]
        by_cat[cat].append(ok)
        rows.append({"id": case_id, "category": cat, "correct": ok, "margin_nats_per_bpe": margin})

    categories = {
        cat: {"correct": sum(values), "total": len(values), "accuracy_pct": 100.0 * sum(values) / len(values)}
        for cat, values in sorted(by_cat.items())
    }
    total_correct = sum(int(x["correct"]) for x in rows)
    summary = {
        "checkpoint_sha256": file_sha(args.checkpoint),
        "tokenizer_sha256": file_sha(args.tokenizer),
        "protocol_sha256": object_sha({"suite": SUITE_VERSION, "rotation": args.rotation, "ids": [x[0] for x in selected],
                                        "implementation": file_sha(Path(__file__))}),
        "suite": "arctus-recovery-smoke-v3",
        "rotation": args.rotation,
        "checkpoint_processed_tokens": processed,
        "architecture": "frontier350-v1",
        "cases": len(rows),
        "overall_accuracy_pct": 100.0 * total_correct / len(rows),
        "categories": categories,
        "rows": rows,
        "purpose": "candidate ranking only; never sufficient for promotion",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
