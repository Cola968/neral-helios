"""Aggregate-only diagnosis for Phoenix private review preparation.

No conversation text, IDs, paths, model weights, GPU access or file writes.
This is diagnostic evidence only and never authorizes training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phoenix_clean_sampler import choose_screened_diverse
from phoenix_oasst2_bridge import load_archive, local_input, select_tree_disjoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--max-en", type=int, default=800)
    ap.add_argument("--max-de", type=int, default=400)
    args = ap.parse_args()
    phase = "root"
    try:
        root = args.assets_root.resolve(strict=True)
        if any((parent / ".git").exists() for parent in (root, *root.parents)):
            raise ValueError("private root is in git tree")
        phase = "archive_identity"
        archive = local_input(args.archive, root)
        phase = "archive_load"
        messages = load_archive(archive)
        phase = "screen"
        selected, identity = choose_screened_diverse(messages, max_en=args.max_en, max_de=args.max_de)
        phase = "split"
        splits, summary = select_tree_disjoint(selected, messages, frozen=set())
        language_by_id = {row["id"]: lang for lang in ("de", "en") for row in selected[lang]}
        buckets = {}
        for split in ("train", "validation"):
            for lang in ("de", "en"):
                buckets[f"{split}_{lang}"] = sum(
                    1 for row in splits[split] if language_by_id.get(row["id"]) == lang
                )
        report = {
            "status": "diagnostic_only",
            "messages": len(messages),
            "selected_de": len(selected["de"]),
            "selected_en": len(selected["en"]),
            "tree_disjoint": bool(summary.get("split_disjoint")),
            "buckets": buckets,
            "candidate_policy": identity.get("policy"),
            "training_approved": False,
            "training_started": False,
            "raw_text_logged": False,
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        # Only phase and exception class are exposed. Exception text may contain
        # local paths or source identifiers, so it is intentionally suppressed.
        print(json.dumps({
            "status": "diagnostic_blocked",
            "phase": phase,
            "exception_class": type(exc).__name__,
            "training_approved": False,
            "training_started": False,
            "raw_text_logged": False,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
