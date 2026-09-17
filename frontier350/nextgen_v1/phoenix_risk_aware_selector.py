"""Experimental risk-aware OASST2 selector for Phoenix.

Starts from the existing reviewed-human/rank-filtered OASST2 pairs, rejects only
coarse deterministic warning classes (contact regexes, role markers, extremely
short/long answers), then selects at most one pair per conversation tree by a
stable SHA-256 order. This is a candidate sampler only: no export, holdout,
training approval, model access or GPU work.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTIER350 = HERE.parent
for item in (HERE, FRONTIER350):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

from fetch_oasst2_sft50 import choose_pairs
from phoenix_data_audit import EMAIL, PHONE
from phoenix_oasst2_content_profile import ROLE_MARKER

POLICY = "phoenix-risk-aware-tree-diverse-v1"
MIN_ANSWER_CHARS = 40
MAX_ANSWER_CHARS = 2500


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hard_warning(pair: dict) -> str | None:
    prompt = pair.get("user")
    answer = pair.get("assistant")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise ValueError("pair text missing")
    if EMAIL.search(prompt) or EMAIL.search(answer):
        return "contact_email"
    if PHONE.search(prompt) or PHONE.search(answer):
        return "contact_phone"
    if ROLE_MARKER.search(answer):
        return "role_marker"
    length = len(answer.strip())
    if length < MIN_ANSWER_CHARS:
        return "answer_too_short"
    if length > MAX_ANSWER_CHARS:
        return "answer_too_long"
    return None


def choose_risk_aware(messages: list[dict], max_en: int = 800,
                      max_de: int = 400) -> tuple[dict[str, list[dict]], dict]:
    if not 8 <= max_en <= 2000 or not 8 <= max_de <= 2000:
        raise ValueError("bounded sample size required")
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
    rejected = {"contact_email": 0, "contact_phone": 0, "role_marker": 0,
                "answer_too_short": 0, "answer_too_long": 0}
    claimed_trees: set[str] = set()
    eligible_tree_counts: dict[str, int] = {}

    for language, cap in (("de", max_de), ("en", max_en)):
        grouped: dict[str, list[dict]] = {}
        for pair in all_pairs[language]:
            warning = _hard_warning(pair)
            if warning:
                rejected[warning] += 1
                continue
            ident = pair.get("id")
            upstream_id = ident[len("oasst2-"):] if isinstance(ident, str) and ident.startswith("oasst2-") else ""
            assistant = by_id.get(upstream_id)
            parent = by_id.get(pair.get("upstream_parent_id"))
            if not assistant or not parent or assistant.get("parent_id") != parent.get("message_id"):
                raise ValueError("pair lineage unavailable")
            tree = assistant.get("message_tree_id")
            if not isinstance(tree, str) or not tree.strip() or tree != parent.get("message_tree_id"):
                raise ValueError("missing or inconsistent conversation tree")
            grouped.setdefault(tree, []).append(pair)

        eligible_tree_counts[language] = len(grouped)
        available = [tree for tree in grouped if tree not in claimed_trees]
        if len(available) < cap:
            raise ValueError("insufficient clean independent conversation trees for requested sample")
        for tree in sorted(available, key=lambda item: (digest("tree:" + item), item))[:cap]:
            chosen = min(grouped[tree], key=lambda row: (digest("pair:" + row["id"]), row["id"]))
            output[language].append(chosen)
            claimed_trees.add(tree)

    selected_ids = sorted(row["id"] for lang in ("de", "en") for row in output[lang])
    report = {
        "policy": POLICY,
        "selected_de": len(output["de"]),
        "selected_en": len(output["en"]),
        "distinct_trees_selected": len(claimed_trees),
        "eligible_tree_counts_after_hard_filter": eligible_tree_counts,
        "hard_warning_rejections_before_tree_sampling": rejected,
        "selected_id_inventory_sha256": digest("\n".join(selected_ids)),
        "training_approved": False,
        "training_started": False,
        "holdout_checked": False,
        "training_dataset_exported": False,
    }
    return output, report
