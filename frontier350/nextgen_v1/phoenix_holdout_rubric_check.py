"""Read-only structural integrity check for an independently authored private holdout v2.

No questions or criteria are generated or sent to logs. This does not judge the
rubrics, establish independence or authorize training. Only local hash/count
checks, entirely CPU-based, with no model or archive access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from phoenix_gate import unique_object
from phoenix_freeze_holdout_v2 import IDENTIFIER, MIN_PROMPTS, MAX_PROMPTS, MAX_BYTES
from sft_contract import prompt_hash, sha

CATEGORIES = {"factual", "reasoning", "math", "instruction", "safety", "writing", "code"}
MAX_RUBRIC_BYTES = 400_000


def private_input(path: Path, root: Path, cap: int) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked private evaluation input")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.stat().st_size > cap:
        raise ValueError("evaluation input outside private root or over size limit")
    return resolved


def inspect(root: Path, prompts: Path, registry: Path, receipt: Path, rubrics: Path) -> dict:
    root = root.resolve(strict=True)
    files = (
        private_input(prompts, root, MAX_BYTES),
        private_input(registry, root, MAX_BYTES),
        private_input(receipt, root, MAX_BYTES),
        private_input(rubrics, root, MAX_RUBRIC_BYTES),
    )
    if len(set(files)) != len(files):
        raise ValueError("all evaluation files must be distinct")
    prompts, registry, receipt, rubrics = files
    question_rows = json.loads(prompts.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    frozen = json.loads(registry.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    metadata = json.loads(receipt.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    rubric_rows = json.loads(rubrics.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(question_rows, list) or not MIN_PROMPTS <= len(question_rows) <= MAX_PROMPTS:
        raise ValueError("invalid private holdout population")
    if not isinstance(metadata, dict) or not all((
            metadata.get("schema") == 1,
            metadata.get("campaign") == "phoenix-holdout-v2",
            metadata.get("question_count") == len(question_rows),
            metadata.get("raw_private_prompts_sha256") == sha(prompts),
            metadata.get("registry_sha256") == sha(registry),
            metadata.get("not_comparable_to_prior_holdout") is True,
            metadata.get("training_data_approved") is False,
            metadata.get("author_review_required") is True)):
        raise ValueError("holdout v2 freeze receipt inconsistent or not unapproved")
    ids, digests = set(), set()
    for row in question_rows:
        if not isinstance(row, dict) or set(row) != {"id", "prompt"}:
            raise ValueError("private question schema invalid")
        ident, prompt = row["id"], row["prompt"]
        if (not isinstance(ident, str) or not IDENTIFIER.fullmatch(ident)
                or ident in ids or not isinstance(prompt, str)
                or not 20 <= len(prompt) <= 2000):
            raise ValueError("invalid or duplicated private question")
        ids.add(ident)
        digests.add(prompt_hash(prompt))
    if len(digests) != len(ids) or not isinstance(frozen, list) or frozen != sorted(digests):
        raise ValueError("frozen prompt hashes differ from private questions")
    if not isinstance(rubric_rows, list) or len(rubric_rows) != len(ids):
        raise ValueError("rubric population differs from frozen questions")
    rubric_ids: set[str] = set()
    languages, categories = Counter(), Counter()
    for row in rubric_rows:
        if not isinstance(row, dict) or set(row) != {"id", "language", "category", "pass_condition"}:
            raise ValueError("rubric schema invalid")
        ident, language, category, condition = (row[key] for key in
                                                ("id", "language", "category", "pass_condition"))
        if not isinstance(ident, str) or ident not in ids or ident in rubric_ids:
            raise ValueError("rubric ID missing, foreign or duplicated")
        if language not in ("de", "en") or category not in CATEGORIES:
            raise ValueError("unsupported rubric language or category")
        if not isinstance(condition, str) or not 20 <= len(condition.strip()) <= 1000:
            raise ValueError("rubric pass condition absent or outside bounds")
        rubric_ids.add(ident)
        languages[language] += 1
        categories[category] += 1
    if rubric_ids != ids or min(languages["de"], languages["en"]) < 4 or len(categories) < 3:
        raise ValueError("rubric coverage or DE/EN skill distribution insufficient")
    return {"status": "v2_freeze_and_rubric_structure_verified_only",
            "campaign": "phoenix-holdout-v2", "questions": len(ids),
            "rubrics": len(rubric_ids), "languages": dict(sorted(languages.items())),
            "categories": dict(sorted(categories.items())),
            "prompt_file_sha256": sha(prompts), "registry_file_sha256": sha(registry),
            "rubric_file_sha256": sha(rubrics),
            "freeze_binding_checked": True, "rubric_id_coverage_checked": True,
            "authorship_verified": False, "semantic_leakage_checked": False,
            "human_rubric_quality_reviewed": False, "answer_quality_evaluated": False,
            "training_approved": False, "training_started": False,
            "raw_text_uploaded": False,
            "note": "This checks format, freeze hashes and rubric coverage only. Rubric adequacy, original authorship, semantics, privacy and training approval require independent human review."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--rubrics", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.assets_root, args.prompts, args.registry, args.receipt, args.rubrics)
    except Exception:
        print(json.dumps({"status": "holdout_v2_rubric_structure_blocked",
                          "training_approved": False, "raw_text_uploaded": False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
