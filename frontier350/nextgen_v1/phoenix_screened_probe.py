"""Read-only comparison on SHA-pinned OASST2; NO GPU, data export or approval.

Only aggregate counts and hashes are logged. The screened policy is experimental
and its regexes can miss personal data or exclude valuable answer examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phoenix_clean_sampler import choose_screened_diverse
from phoenix_diverse_probe import summarize
from phoenix_diverse_selector import choose_tree_diverse
from phoenix_oasst2_bridge import UPSTREAM_SHA256, load_archive, local_input
from fetch_oasst2_sft50 import choose_pairs


def compare(messages: list[dict], max_de: int = 400, max_en: int = 800) -> dict:
    if not 50 <= max_de <= 2000 or not 50 <= max_en <= 2000:
        raise ValueError("invalid diagnostic caps")
    prior = choose_pairs(messages, max_de=max_de, max_en=max_en)
    diverse, diverse_info = choose_tree_diverse(messages, max_de=max_de, max_en=max_en)
    screened, screened_info = choose_screened_diverse(messages, max_de=max_de, max_en=max_en)
    return {"status": "three_way_source_comparison_not_training_approval",
            "archive_sha256": UPSTREAM_SHA256,
            "messages": len(messages),
            "legacy": summarize(prior, messages),
            "tree_diverse": summarize(diverse, messages),
            "screened_tree_diverse": summarize(screened, messages),
            "diverse_policy": diverse_info["policy"],
            "screened_policy": screened_info,
            "holdout_checked": False, "license_and_privacy_reviewed": False,
            "semantic_leakage_checked": False, "answer_quality_measured": False,
            "training_approved": False, "training_dataset_exported": False,
            "training_started": False, "model_loaded": False,
            "gpu_used": False, "model_weights_modified": False,
            "raw_text_uploaded": False,
            "interpretation": "Length/regex exclusion counts are only heuristics; no model-quality gain or privacy certification. Existing source export remains unchanged."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-root", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    args = ap.parse_args()
    stage = "private_archive_preflight"
    try:
        root = args.assets_root.resolve(strict=True)
        archive = local_input(args.archive, root)
        stage = "pinned_archive_three_way_diagnostic"
        report = compare(load_archive(archive))
    except Exception:
        print(json.dumps({"status": "screened_source_comparison_blocked", "stage": stage,
                          "training_approved": False, "training_started": False,
                          "gpu_used": False, "raw_text_uploaded": False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
