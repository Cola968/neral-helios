"""Read-only diagnostic of *actual* pinned OASST2 chooser and tree split.

No holdout is supplied: this is expressly NOT a corpus export, contamination
screen, dataset approval, candidate checkpoint, training or quality measurement.
Only aggregate counts are printed. All source texts remain private in RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from phoenix_oasst2_bridge import (UPSTREAM_SHA256, load_archive, local_input,
                                   select_tree_disjoint)
from fetch_oasst2_sft50 import choose_pairs


def profile_messages(messages: list[dict], max_en: int = 800, max_de: int = 400) -> dict:
    """Run real selectors but do not write, expose, or certify their rows."""
    if not 50 <= max_en <= 2000 or not 50 <= max_de <= 2000:
        raise ValueError("sample caps outside bounded range")
    pairs = choose_pairs(messages, max_en=max_en, max_de=max_de)
    # Empty set is ONLY for in-memory diagnostics, not a frozen holdout substitute.
    # The production `prepare()` continues to require a real existing registry.
    splits, summary = select_tree_disjoint(pairs, messages, frozen=set())
    return {
        "status": "real_source_selector_dry_run_not_approved",
        "original_archive_sha256": UPSTREAM_SHA256,
        "source_messages": len(messages),
        "chooser_output_before_tree_and_prompt_dedup": {
            "de": len(pairs["de"]), "en": len(pairs["en"])},
        "dry_run_rows": summary["rows"],
        "dry_run_conversation_trees": summary["conversation_trees"],
        "duplicate_prompts_skipped": summary["duplicate_prompts_skipped"],
        "incompatible_text_skipped": summary["incompatible_text_skipped"],
        "tree_disjoint": summary["split_disjoint"],
        "frozen_holdout_checked": False,
        "semantic_overlap_checked": False,
        "privacy_and_license_reviewed": False,
        "training_dataset_exported": False,
        "training_approved": False,
        "training_started": False,
        "raw_text_uploaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    stage = "private_source_preflight"
    try:
        root = args.assets_root.resolve(strict=True)
        archive = local_input(args.archive, root)
        stage = "pinned_archive_and_full_selection"
        report = profile_messages(load_archive(archive))
    except Exception:
        # Do not expose paths, IDs, source text or a raw exception to Actions.
        print(json.dumps({"status": "source_selection_dry_run_blocked",
                          "stage": stage, "training_approved": False,
                          "training_started": False, "raw_text_uploaded": False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
