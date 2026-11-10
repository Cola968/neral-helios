"""Read-only original OASST2 comparison of three sampling policies.

Compares the legacy prefix chooser, tree-diverse chooser, and the existing
screened tree-diverse chooser. Produces aggregate metadata only: no holdout,
approval, raw text/IDs in logs, file export, GPU, model loading or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fetch_oasst2_sft50 import choose_pairs
from phoenix_clean_sampler import choose_screened_diverse
from phoenix_diverse_selector import choose_tree_diverse
from phoenix_oasst2_bridge import UPSTREAM_SHA256, load_archive, local_input, select_tree_disjoint
from phoenix_oasst2_content_profile import screen_selected


def summarize(pairs: dict, messages: list[dict]) -> dict:
    splits, meta = select_tree_disjoint(pairs, messages, frozen=set())
    languages = {row["id"]: lang for lang in ("de", "en") for row in pairs[lang]}
    if len(languages) != sum(len(pairs[lang]) for lang in ("de", "en")):
        raise ValueError("ambiguous selection IDs")
    content = screen_selected(splits, languages)
    return {"rows": content["rows"], "languages": content["languages"],
            "conversation_trees": meta["conversation_trees"],
            "distinct_trees_final": sum(meta["conversation_trees"].values()),
            "duplicate_prompts_skipped": meta["duplicate_prompts_skipped"],
            "incompatible_text_skipped": meta["incompatible_text_skipped"],
            "flag_counts": content["flag_counts"], "tree_disjoint": meta["split_disjoint"]}


def compare(messages: list[dict], max_en: int = 800, max_de: int = 400) -> dict:
    if not 50 <= max_en <= 2000 or not 50 <= max_de <= 2000:
        raise ValueError("bounded campaign sizes required")
    baseline = choose_pairs(messages, max_en=max_en, max_de=max_de)
    diverse, diverse_id = choose_tree_diverse(messages, max_en=max_en, max_de=max_de)
    screened, screened_id = choose_screened_diverse(messages, max_en=max_en, max_de=max_de)
    return {
        "status": "exploratory_sample_comparison_not_training_approval",
        "archive_sha256": UPSTREAM_SHA256,
        "source_messages": len(messages),
        "legacy_prefix": summarize(baseline, messages),
        "candidate_tree_diverse": summarize(diverse, messages),
        "candidate_screened_tree_diverse": summarize(screened, messages),
        "tree_diverse_policy": diverse_id["policy"],
        "tree_diverse_inventory_sha256": diverse_id["selected_id_inventory_sha256"],
        "screened_policy": screened_id["policy"],
        "screened_inventory_sha256": screened_id["selected_inventory_sha256"],
        "screened_rejected_upstream_pairs_by_first_reason": screened_id["rejected_upstream_pairs_by_first_reason"],
        "screened_upstream_eligible_rows": screened_id["upstream_eligible_rows"],
        "screened_duplicate_prompts_avoided": screened_id["duplicate_prompts_avoided_during_selection"],
        "holdout_checked": False,
        "semantic_leakage_checked": False,
        "privacy_fully_reviewed": False,
        "license_reviewed": False,
        "human_answer_quality_reviewed": False,
        "training_approved": False,
        "training_dataset_exported": False,
        "training_started": False,
        "model_weights_modified": False,
        "raw_text_uploaded": False,
        "interpretation": "Heuristic screening and tree diversity do not establish answer quality, privacy completeness, legal provenance, semantic independence or training readiness. Existing export path remains unchanged.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args()
    stage = "private_archive_preflight"
    try:
        root = args.assets_root.resolve(strict=True)
        archive = local_input(args.archive, root)
        stage = "pinned_archive_and_three_source_samplers"
        report = compare(load_archive(archive))
    except Exception:
        print(json.dumps({"status": "source_sampler_comparison_blocked", "stage": stage,
                          "training_approved": False, "training_started": False,
                          "raw_text_uploaded": False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
