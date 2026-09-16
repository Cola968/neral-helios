"""Freeze an operator-curated *new* private Phoenix holdout v2. No synthetic benchmark creation.

Does not fetch data, authorize training, score answers or access GPUs. Never prints prompts.
The original lost holdout cannot be reconstructed with this tool.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from phoenix_gate import unique_object
from sft_contract import norm, prompt_hash, sha

IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MIN_PROMPTS = 24
MAX_PROMPTS = 100
MAX_BYTES = 256_000


def _input(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private prompt input must be an existing regular file")
    item = path.resolve(strict=True)
    if not item.is_relative_to(root) or item.stat().st_size > MAX_BYTES:
        raise ValueError("private prompts must stay inside the asset root and below size bound")
    return item


def _output(path: Path, root: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite existing holdout files")
    if not path.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("holdout outputs must remain inside the private asset root")
    return path


def freeze(root: Path, prompts_path: Path, registry_path: Path, receipt_path: Path) -> dict:
    root = root.resolve(strict=True)
    prompts_path = _input(prompts_path, root)
    registry_path = _output(registry_path, root)
    receipt_path = _output(receipt_path, root)
    if len({prompts_path.resolve(), registry_path.resolve(), receipt_path.resolve()}) != 3:
        raise ValueError("all input and output paths must be distinct")
    rows = json.loads(prompts_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(rows, list) or not MIN_PROMPTS <= len(rows) <= MAX_PROMPTS:
        raise ValueError("new holdout must contain 24 to 100 independently reviewed questions")
    ids, hashes = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "prompt"}:
            raise ValueError("each holdout row must have exactly id and prompt")
        identifier, prompt = row["id"], row["prompt"]
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier) or identifier in ids:
            raise ValueError("invalid or duplicated holdout ID")
        if not isinstance(prompt, str) or not 20 <= len(prompt) <= 2000 or not norm(prompt):
            raise ValueError("invalid holdout prompt length or text")
        digest = prompt_hash(prompt)
        if digest in hashes:
            raise ValueError("duplicated holdout prompt after canonical normalization")
        ids.add(identifier)
        hashes.add(digest)
    # A sorted hash-only registry is compatible with the existing SFT and bridge readers.
    registry_bytes = (json.dumps(sorted(hashes), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    receipt = {
        "schema": 1,
        "campaign": "phoenix-holdout-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "raw_private_prompts_sha256": sha(prompts_path),
        "registry_sha256": __import__("hashlib").sha256(registry_bytes).hexdigest(),
        "not_comparable_to_prior_holdout": True,
        "author_review_required": True,
        "training_data_approved": False,
        "status": "new_private_holdout_frozen_not_approved",
        "note": "Original holdout missing; do not reuse old evaluation scores. Freeze before any candidate training, check source overlap and manually review authorship/PII/rubric.",
    }
    created = []
    try:
        with registry_path.open("xb") as stream:
            created.append(registry_path)
            stream.write(registry_bytes)
        with receipt_path.open("x", encoding="utf-8") as stream:
            created.append(receipt_path)
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except Exception:
        # Both files form one freeze. Remove only new files from this failed attempt.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    # No question, answer or private file path is printed or uploaded.
    return {"status": receipt["status"], "question_count": len(rows),
            "not_comparable_to_prior_holdout": True, "training_data_approved": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.assets_root, args.prompts, args.registry, args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
