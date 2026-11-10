"""Experimental tree-diverse OASST2 selection, NOT the production data pipeline.

The legacy chooser truncates lexicographically ordered prompt IDs. This sampler
compares a deterministic hash-based, one-pair-per-tree alternative. No network,
training, holdout data, dataset export, or source approval. A candidate sampling
scheme must be reviewed BEFORE use in any training campaign.
"""
from __future__ import annotations

import hashlib

from fetch_oasst2_sft50 import choose_pairs

POLICY = "phoenix-tree-diverse-v1-sha256-order-one-pair-per-tree-de-first"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_tree_diverse(messages: list[dict], max_en: int = 800,
                        max_de: int = 400) -> tuple[dict[str, list[dict]], dict]:
    if not 8 <= max_en <= 2000 or not 8 <= max_de <= 2000:
        raise ValueError("bounded sample size required")
    # None disables ONLY the legacy end-slice. All upstream reviewed-human,
    # rank and prompt-answer duplicate rules remain unchanged.
    all_pairs = choose_pairs(messages, max_en=None, max_de=None)
    by_id: dict[str, dict] = {}
    for message in messages:
        ident = message.get("message_id") if isinstance(message, dict) else None
        if not isinstance(ident, str):
            continue
        if ident in by_id:
            raise ValueError("duplicate upstream message ID")
        by_id[ident] = message
    output = {"de": [], "en": []}
    claimed_trees: set[str] = set()
    source_trees = {}
    for language, cap in (("de", max_de), ("en", max_en)):
        grouped: dict[str, list[dict]] = {}
        for pair in all_pairs[language]:
            ident = pair.get("id")
            assistant = by_id.get(ident[len("oasst2-"):] if isinstance(ident, str) and ident.startswith("oasst2-") else "")
            parent = by_id.get(pair.get("upstream_parent_id"))
            if not assistant or not parent or assistant.get("parent_id") != parent.get("message_id"):
                raise ValueError("pair lineage unavailable")
            tree = assistant.get("message_tree_id")
            if not isinstance(tree, str) or not tree.strip() or tree != parent.get("message_tree_id"):
                raise ValueError("missing or inconsistent conversation tree")
            grouped.setdefault(tree, []).append(pair)
        source_trees[language] = len(grouped)
        available = [tree for tree in grouped if tree not in claimed_trees]
        if len(available) < cap:
            raise ValueError("insufficient independent conversation trees for requested sample")
        # SHA ordering avoids lexicographic upstream-ID head bias; choose a stable
        # answer per tree. Across languages, DE claims trees first to avoid reuse.
        for tree in sorted(available, key=lambda item: (digest("tree:" + item), item))[:cap]:
            chosen = min(grouped[tree], key=lambda row: (digest("pair:" + row["id"]), row["id"]))
            output[language].append(chosen)
            claimed_trees.add(tree)
    selected_ids = sorted(pair["id"] for lang in ("de", "en") for pair in output[lang])
    report = {"policy": POLICY, "selected_de": len(output["de"]),
              "selected_en": len(output["en"]),
              "source_distinct_trees": source_trees,
              "distinct_trees_selected": len(claimed_trees),
              "selected_id_inventory_sha256": digest("\n".join(selected_ids)),
              "training_approved": False, "training_started": False,
              "holdout_checked": False, "training_dataset_exported": False}
    return output, report
