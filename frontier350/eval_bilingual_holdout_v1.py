from __future__ import annotations

"""Evaluate a checkpoint on the runtime-generated ARCTUS bilingual holdout."""

import argparse
import collections
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval500m_diagnostic import conditional_mean_nll
from model350 import FRONTIER_350, Neral350M
from recovery_identity import file_sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-tokens", type=int, required=True)
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

    spec = json.loads(Path(args.cases).read_text(encoding="utf8"))
    if spec.get("suite") != "arctus-bilingual-holdout-v1" or spec.get("training_overlap") is not False:
        raise RuntimeError("wrong/unsafe bilingual holdout")
    cases = spec.get("cases") or []
    if len(cases) != 400:
        raise RuntimeError(f"expected exactly 400 holdout cases, got {len(cases)}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", 0))
    if not args.expected_tokens <= processed <= args.expected_tokens + 32_768:
        raise RuntimeError(f"unexpected checkpoint tokens: {processed}")
    if ck.get("tokenizer_sha256") != file_sha(args.tokenizer):
        raise RuntimeError("checkpoint/tokenizer digest mismatch")

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    by_direction: dict[str, list[bool]] = collections.defaultdict(list)
    rows = []
    for case in cases:
        options = [case["preferred"], *case["distractors"]]
        losses = [
            conditional_mean_nll(model, tok, case["prefix"], option, device, int(bos_id))
            for option in options
        ]
        if not all(math.isfinite(value) for value in losses):
            raise FloatingPointError(f"non-finite score: {case['id']}")
        chosen = min(range(len(losses)), key=losses.__getitem__)
        ok = chosen == 0
        direction = str(case["direction"])
        by_direction[direction].append(ok)
        rows.append(
            {
                "id": case["id"],
                "direction": direction,
                "correct": ok,
                "preferred_nll": losses[0],
                "best_distractor_nll": min(losses[1:]),
                "margin_nats_per_bpe": min(losses[1:]) - losses[0],
            }
        )

    directions = {
        name: {
            "correct": sum(values),
            "total": len(values),
            "accuracy_pct": 100.0 * sum(values) / len(values),
        }
        for name, values in sorted(by_direction.items())
    }
    correct = sum(int(row["correct"]) for row in rows)
    summary = {
        "schema": 1,
        "suite": spec["suite"],
        "checkpoint_sha256": file_sha(args.checkpoint),
        "checkpoint_processed_tokens": processed,
        "tokenizer_sha256": file_sha(args.tokenizer),
        "cases_sha256": file_sha(args.cases),
        "cases": len(rows),
        "overall_accuracy_pct": 100.0 * correct / len(rows),
        "directions": directions,
        "rows": rows,
        "purpose": "development-only independent bilingual recovery check; not final audit/public benchmark",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
