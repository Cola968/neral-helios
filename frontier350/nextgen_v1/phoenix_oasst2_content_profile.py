"""Read-only, non-approving risk counts for the real pinned OASST2 Phoenix selection.

Only aggregate integer counts leave this process. This is NOT a comprehensive PII,
license, semantic leakage, subjective answer-quality or holdout audit. Never export
texts or identifiers, build a training pack, or auto-set any approval flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from phoenix_data_audit import EMAIL, PHONE
from phoenix_oasst2_bridge import UPSTREAM_SHA256, load_archive, local_input, select_tree_disjoint
from fetch_oasst2_sft50 import choose_pairs
from sft_contract import norm

ROLE_MARKER = re.compile(r"(?im)(?:^|\n)\s*(?:user|assistant|system|human|bot)\s*:")


def screen_selected(splits: dict[str, list[dict]], language_by_id: dict[str, str]) -> dict:
    """Count coarse screening flags. Duplicate answers are hints, not leakage proof."""
    if set(splits) != {"train", "validation"}:
        raise ValueError("unexpected split inventory")
    seen_answers: dict[str, str] = {}
    rows: dict[str, int] = {}
    languages: dict[str, dict[str, int]] = {}
    flags = Counter()
    for split in ("train", "validation"):
        inventory = splits[split]
        if not isinstance(inventory, list) or not inventory or len(inventory) > 2000:
            raise ValueError("invalid bounded selected inventory")
        rows[split] = len(inventory)
        languages[split] = {"de": 0, "en": 0}
        for item in inventory:
            if not isinstance(item, dict) or set(item) != {"id", "source_id", "user", "assistant"}:
                raise ValueError("invalid selected row schema")
            ident, question, answer = item["id"], item["user"], item["assistant"]
            if not all(isinstance(value, str) and value.strip() for value in (ident, question, answer)):
                raise ValueError("invalid selected row contents")
            lang = language_by_id.get(ident)
            if lang not in ("de", "en"):
                raise ValueError("source language lineage missing")
            languages[split][lang] += 1
            if EMAIL.search(question) or EMAIL.search(answer):
                flags["email_pattern_rows"] += 1
            if PHONE.search(question) or PHONE.search(answer):
                flags["phone_pattern_rows"] += 1
            if len(answer.strip()) < 40:
                flags["short_answer_rows_under_40_chars"] += 1
            if len(answer) > 2500:
                flags["long_answer_rows_over_2500_chars"] += 1
            if ROLE_MARKER.search(answer):
                flags["role_marker_answer_rows"] += 1
            if len(question) > 512:
                flags["prompt_over_512_chars_near_screen_truncates"] += 1
            normalized_answer = norm(answer)
            if normalized_answer in seen_answers:
                flags["duplicate_normalized_answer_rows"] += 1
                if seen_answers[normalized_answer] != split:
                    flags["answer_shared_across_splits_rows"] += 1
            else:
                seen_answers[normalized_answer] = split
    return {"rows": rows, "languages": languages,
            "flag_counts": {name: flags[name] for name in (
                "email_pattern_rows", "phone_pattern_rows",
                "short_answer_rows_under_40_chars", "long_answer_rows_over_2500_chars",
                "role_marker_answer_rows", "prompt_over_512_chars_near_screen_truncates",
                "duplicate_normalized_answer_rows", "answer_shared_across_splits_rows")}}


def profile_messages(messages: list[dict], max_en: int = 800, max_de: int = 400) -> dict:
    if not 50 <= max_en <= 2000 or not 50 <= max_de <= 2000:
        raise ValueError("sample caps outside bounded range")
    pairs = choose_pairs(messages, max_en=max_en, max_de=max_de)
    language_by_id = {pair["id"]: lang for lang in ("de", "en") for pair in pairs[lang]}
    if len(language_by_id) != sum(len(pairs[lang]) for lang in ("de", "en")):
        raise ValueError("duplicate upstream selection identifier")
    # Empty frozen set is legal ONLY for this memory-only diagnostic. Never export.
    splits, meta = select_tree_disjoint(pairs, messages, frozen=set())
    screened = screen_selected(splits, language_by_id)
    if screened["rows"] != meta["rows"]:
        raise AssertionError("selection counts differ")
    return {"status": "real_selected_content_risk_hints_not_approved",
            "source_archive_sha256": UPSTREAM_SHA256,
            "source_messages": len(messages),
            "chooser_de": len(pairs["de"]), "chooser_en": len(pairs["en"]),
            "conversation_trees": meta["conversation_trees"],
            "tree_disjoint": meta["split_disjoint"],
            "duplicate_prompts_skipped": meta["duplicate_prompts_skipped"],
            **screened,
            "privacy_fully_reviewed": False, "licenses_independently_reviewed": False,
            "answer_quality_human_reviewed": False, "semantic_leakage_checked": False,
            "frozen_holdout_checked": False, "training_approved": False,
            "training_dataset_exported": False, "training_started": False,
            "model_weights_modified": False, "raw_text_uploaded": False,
            "limitations": "Regex PII hits and string duplicates are only warning counts; misses, false positives, unattributed content, unsafe answers and semantic overlaps require independent private human review."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    stage = "private_source_preflight"
    try:
        root = args.assets_root.resolve(strict=True)
        archive = local_input(args.archive, root)
        stage = "pinned_archive_and_content_profile"
        result = profile_messages(load_archive(archive))
    except Exception:
        print(json.dumps({"status": "content_profile_blocked", "stage": stage,
                          "training_approved": False, "raw_text_uploaded": False,
                          "training_started": False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
