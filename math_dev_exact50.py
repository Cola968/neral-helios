"""Exact-generation evaluator for non-frozen Neral-50M math development rows.

This helper is for checkpoint selection during future training stages. It never loads the
frozen RawMath benchmark. Callers must pass independent held-out rows such as RawMath dev or
a separately generated transfer-dev set. Greedy exact-match is measured in the same answer
format the model must ultimately generate, avoiding the M1 failure mode where NLL improved
substantially but transfer exact-match stayed at zero.
"""
from __future__ import annotations

import random
from collections import defaultdict

from raw_math50_data import CATEGORIES
from tokenizer import ByteTokenizer

TOK = ByteTokenizer()
DEFAULT_SUBSET_SEED = 968_761


def normalize_output(text: str) -> str:
    return text.strip()


def select_balanced_subset(rows: list[dict], per_category: int = 5, seed: int = DEFAULT_SUBSET_SEED) -> list[dict]:
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    rng = random.Random(seed)
    out: list[dict] = []
    seen_ids: set[str] = set()
    for category in CATEGORIES:
        pool = [dict(r) for r in rows if r.get("category") == category]
        if len(pool) < per_category:
            raise ValueError(f"not enough rows for {category}: {len(pool)} < {per_category}")
        rng.shuffle(pool)
        chosen = pool[:per_category]
        for row in chosen:
            identity = str(row.get("canonical", row.get("id", row.get("prompt"))))
            if identity in seen_ids:
                raise ValueError(f"duplicate dev identity: {identity}")
            seen_ids.add(identity)
            out.append(row)
    return out


def prompt_ids(row: dict) -> list[int]:
    return TOK.encode("Question: " + row["prompt"].strip() + "\nAnswer:", bos=True)


def _aggregate(items: list[dict]) -> dict:
    n = len(items)
    return {
        "tasks": n,
        "exact": sum(bool(x["exact"]) for x in items),
        "exact_pct": 100.0 * sum(bool(x["exact"]) for x in items) / n if n else None,
        "eos_pct": 100.0 * sum(bool(x["eos"]) for x in items) / n if n else None,
        "valid_utf8_pct": 100.0 * sum(bool(x["valid_utf8"]) for x in items) / n if n else None,
    }


def evaluate_exact(model, rows: list[dict], device, max_new_tokens: int = 32) -> dict:
    """Run greedy exact-match on caller-supplied independent dev rows."""
    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    was_training = bool(model.training)
    model.eval()
    results: list[dict] = []
    try:
        for row in rows:
            ids = prompt_ids(row)
            x = torch.tensor([ids], device=device)
            with torch.inference_mode():
                out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0, eos_id=TOK.eos_id)
            new = out[0, len(ids):].tolist()
            eos = TOK.eos_id in new
            if eos:
                new = new[:new.index(TOK.eos_id)]
            raw = TOK.decode(new, skip_special=True)
            expected = str(row["answer"]).strip()
            results.append({
                "canonical": row.get("canonical"),
                "category": row.get("category"),
                "language": row.get("language"),
                "expected": expected,
                "output": raw,
                "exact": normalize_output(raw) == expected,
                "eos": eos,
                "valid_utf8": "\ufffd" not in raw,
            })
    finally:
        model.train(was_training)

    by_category = {}
    for category in CATEGORIES:
        chosen = [r for r in results if r.get("category") == category]
        if chosen:
            by_category[category] = _aggregate(chosen)
    by_language = {}
    for language in sorted({str(r.get("language")) for r in results}):
        by_language[language] = _aggregate([r for r in results if str(r.get("language")) == language])
    return {
        "overall": _aggregate(results),
        "by_category": by_category,
        "by_language": by_language,
        "results": results,
        "definition": "Greedy strict exact-match on caller-supplied non-frozen development rows; surrounding whitespace only is ignored.",
        "frozen_benchmark_used": False,
    }
