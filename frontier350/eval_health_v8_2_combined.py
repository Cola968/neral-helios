from __future__ import annotations

"""EXPERIMENTAL: smoke + validation from one checkpoint load.

Output/protocol SHA inputs reproduce independent evaluators. Do not enable in
production until a same-checkpoint GPU parity test passes. No training writes.
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
from eval_recovery_smoke_v3 import select_cases
from eval_recovery_dev import SUITE_VERSION
from train350_compiled import PersistentMMapPool, evaluate
from recovery_identity import file_sha, object_sha

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "tokenizer", "valid-data", "smoke-out", "validation-out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-tokens", type=int, required=True)
    p.add_argument("--rotation", type=int, choices=range(4), required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=351337)
    a = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if a.seq <= 0 or a.seq > FRONTIER_350.max_seq_len:
        raise ValueError("invalid sequence length")
    ckpt = Path(a.checkpoint)
    tokenizer_file = Path(a.tokenizer)
    checkpoint_sha = file_sha(ckpt)
    tokenizer_sha = file_sha(tokenizer_file)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    processed = int(ck.get("processed_tokens", 0))
    if not a.expected_tokens <= processed <= a.expected_tokens + 32_768:
        raise RuntimeError("checkpoint token identity mismatch")
    if ck.get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("checkpoint/tokenizer digest mismatch")
    tok = Tokenizer.from_file(str(tokenizer_file))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise RuntimeError("tokenizer vocabulary mismatch")
    bos_id = tok.token_to_id("<|bos|>")
    if bos_id is None:
        raise RuntimeError("missing BOS token")
    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    selected = select_cases(a.rotation)
    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    rows = []
    for case_id, category, prefix, preferred, distractors in selected:
        losses = [conditional_mean_nll(model, tok, prefix, x, device, int(bos_id))
                  for x in [preferred, *distractors]]
        if not all(math.isfinite(x) for x in losses):
            raise FloatingPointError(f"nonfinite smoke losses: {case_id}")
        ok = min(range(len(losses)), key=losses.__getitem__) == 0
        margin = min(losses[1:]) - losses[0]
        by_cat[category].append(ok)
        rows.append({"id": case_id, "category": category,
                     "correct": ok, "margin_nats_per_bpe": margin})
    categories = {
        cat: {"correct": sum(v), "total": len(v), "accuracy_pct": 100.0 * sum(v) / len(v)}
        for cat, v in sorted(by_cat.items())
    }
    smoke = {
        "checkpoint_sha256": checkpoint_sha, "tokenizer_sha256": tokenizer_sha,
        "protocol_sha256": object_sha({"suite": SUITE_VERSION, "rotation": a.rotation,
            "ids": [x[0] for x in selected], "implementation": file_sha(ROOT / "eval_recovery_smoke_v3.py")}),
        "suite": "arctus-recovery-smoke-v3", "rotation": a.rotation,
        "checkpoint_processed_tokens": processed, "architecture": "frontier350-v1",
        "cases": len(rows), "overall_accuracy_pct": 100.0 * sum(int(r["correct"]) for r in rows) / len(rows),
        "categories": categories, "rows": rows,
        "purpose": "candidate ranking only; never sufficient for promotion",
    }

    pool = PersistentMMapPool(Path(a.valid_data), FRONTIER_350.vocab_size)
    val = evaluate(model, pool, seq=a.seq, batch=a.batch, batches=a.batches,
                   device=device, seed=a.seed)
    validation = {
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer_sha256": ck.get("tokenizer_sha256"),
        "protocol_sha256": object_sha({"seq": a.seq, "batch": a.batch,
            "batches": a.batches, "seed": a.seed,
            "shards": [(path.name, file_sha(path)) for path in pool.paths],
            "implementation": {name: file_sha(ROOT / name)
                for name in ("model350.py", "train350_compiled.py", "eval_validation_loss350.py")}}),
        "schema": 1, "architecture": "frontier350-v1",
        "checkpoint_processed_tokens": processed,
        "validation_loss": val, "validation_perplexity": math.exp(val),
        "seq": a.seq, "batch": a.batch, "batches": a.batches, "seed": a.seed,
    }
    if file_sha(ckpt) != checkpoint_sha:
        raise RuntimeError("checkpoint changed during combined read-only evaluation")
    for path, obj in ((Path(a.smoke_out), smoke), (Path(a.validation_out), validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"event": "v82_combined_health", "checkpoint_sha256": checkpoint_sha,
                      "tokens": processed, "smoke_accuracy_pct": smoke["overall_accuracy_pct"],
                      "validation_loss": val}), flush=True)


if __name__ == "__main__":
    main()
