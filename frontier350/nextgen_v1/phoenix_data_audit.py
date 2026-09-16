"""Local non-approving Phoenix SFT prescreen. NEVER certifies dataset readiness.

Checks exact prompt/holdout collisions, indicative near duplicates and basic
privacy warning patterns. Outputs counts/hashes only, never corpus text.
Manual license, provenance, semantic-leakage and privacy review are mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from phoenix_gate import unique_object
from sft_contract import hex_digest, norm, prompt_hash, sha

EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){10,14}(?!\w)")
MAX_ROWS_PER_SPLIT = 20_000
MAX_CANDIDATES = 500_000
MAX_HASHED_SHINGLES = 32  # Deterministic, bounded work; NOT complete similarity detection.


def local_file(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required local input missing or symlinked")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("all inputs must remain in private asset root")
    return resolved


def shingles(text: str) -> set[str]:
    if len(text) < 32:
        return set()
    sample = text[:512]
    return {sample[i:i + 5] for i in range(len(sample) - 4)}


def simhash(grams: set[str]) -> int:
    """A sampled, deterministic *hint*; do not treat a missed pair as clean."""
    bits = [0] * 64
    for gram in sorted(grams)[:MAX_HASHED_SHINGLES]:
        value = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            bits[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(bits) if score >= 0)


def audit(train: Path, validation: Path, holdout: Path, root: Path) -> dict:
    """Review hints only: no auto-approval, no samples, bounded near-pair work."""
    root = root.resolve(strict=True)
    train, validation, holdout = (local_file(item, root) for item in (train, validation, holdout))
    hashes = json.loads(holdout.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("nonempty frozen holdout hash registry required")
    frozen = {hex_digest(item, "holdout prompt") for item in hashes}
    if len(frozen) != len(hashes):
        raise ValueError("duplicate frozen holdout hash")
    rows = {"train": 0, "validation": 0}
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    index: dict[tuple[int, int], list[int]] = {}
    previous: list[tuple[str, set[str], int]] = []
    exact = near = privacy = format_flags = long = compared = 0
    for split, path in (("train", train), ("validation", validation)):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number > MAX_ROWS_PER_SPLIT:
                    raise ValueError(f"{split}: bounded prescreen row limit exceeded")
                row = json.loads(line, object_pairs_hook=unique_object)
                if not isinstance(row, dict) or set(row) != {"id", "source_id", "user", "assistant"}:
                    raise ValueError(f"{split}:{line_number}: invalid row schema")
                if any(not isinstance(row[name], str) or not row[name].strip() for name in row):
                    raise ValueError(f"{split}:{line_number}: empty or invalid field")
                ident = row["id"]
                if ident in seen_ids:
                    format_flags += 1
                seen_ids.add(ident)
                prompt, answer = row["user"], row["assistant"]
                if len(prompt) > 8192 or len(answer) > 8192:
                    raise ValueError(f"{split}:{line_number}: text exceeds pilot limit")
                fingerprint = prompt_hash(prompt)
                if fingerprint in seen_hashes:
                    exact += 1
                seen_hashes.add(fingerprint)
                if fingerprint in frozen:
                    exact += 1
                if EMAIL.search(prompt) or EMAIL.search(answer) or PHONE.search(prompt) or PHONE.search(answer):
                    privacy += 1  # Indicative, NOT comprehensive PII screening.
                clean = norm(prompt)
                if len(clean) > 512:
                    long += 1  # Unexamined prompt suffix remains manual-review work.
                grams = shingles(clean)
                if grams:
                    fp = simhash(grams)
                    candidates: set[int] = set()
                    for band in range(4):
                        key = band, (fp >> (16 * band)) & 0xffff
                        candidates.update(index.get(key, ()))
                    for prior in candidates:
                        compared += 1
                        if compared > MAX_CANDIDATES:
                            raise ValueError("near-duplicate candidate budget exceeded; manual review required")
                        old_hash, old_grams, old_fp = previous[prior]
                        if fingerprint == old_hash or (fp ^ old_fp).bit_count() > 12:
                            continue
                        overlap = len(grams & old_grams)
                        if overlap / max(1, len(grams | old_grams)) >= 0.82:
                            near += 1
                    pos = len(previous)
                    previous.append((fingerprint, grams, fp))
                    for band in range(4):
                        key = band, (fp >> (16 * band)) & 0xffff
                        index.setdefault(key, []).append(pos)
                rows[split] += 1
        if not rows[split]:
            raise ValueError(f"{split}: no training rows")
    return {"version": 1, "classification": "automated_prescreen_not_approval",
            "approved": False, "manual_review_required": True,
            "train_sha256": sha(train), "validation_sha256": sha(validation),
            "holdout_sha256": sha(holdout), "rows": rows,
            "exact_overlap_count": exact, "near_overlap_count": near,
            "privacy_hits": privacy, "duplicate_id_flags": format_flags,
            "long_prompt_near_screen_incomplete": long,
            "near_candidate_comparisons": compared,
            "note": "Only indicative automated screening. Sampled SimHash covers at most 512 prompt characters and 32 shingles; it can MISS near/semantic duplicates. Email/phone regex can MISS PII. License/source, held-out semantic leakage and privacy require independent manual review. Never flip approved to true automatically."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.assets_root.resolve(strict=True)
    if (args.out.exists() or args.out.is_symlink()
            or not args.out.parent.resolve(strict=True).is_relative_to(root)):
        parser.error("new local-only output path required; no overwrite or symlink")
    report = audit(args.train, args.validation, args.holdout_hashes, root)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
