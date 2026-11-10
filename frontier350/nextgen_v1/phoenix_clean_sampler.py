"""Experimental screened OASST2 sampler, NEVER an automatic training approval.

Candidate for comparison only: source texts remain in memory, no file writes, no
model or GPU. Regex flags can miss PII and reject perfectly good examples.
Filtering is not privacy, license or answer-quality certification.
"""
from __future__ import annotations

from collections import Counter

from fetch_oasst2_sft50 import choose_pairs, reviewed_human
from phoenix_data_audit import EMAIL, PHONE
from phoenix_diverse_selector import digest
from phoenix_oasst2_content_profile import ROLE_MARKER
from sft_contract import FORBIDDEN, prompt_hash

POLICY = "phoenix-screened-tree-diverse-v1-no-contact-length-role-flags"


def screen_reason(prompt: str, answer: str) -> str | None:
    """Ordered conservative rejection reasons, not a comprehensive PII detector."""
    if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip():
        return "invalid_text"
    if any(marker in text for text in (prompt, answer) for marker in FORBIDDEN):
        return "reserved_role_tokens"
    if EMAIL.search(prompt) or EMAIL.search(answer):
        return "email_pattern"
    if PHONE.search(prompt) or PHONE.search(answer):
        return "phone_pattern"
    if len(prompt) > 512:
        return "long_prompt"
    if len(answer.strip()) < 40:
        return "short_answer"
    if len(answer) > 2500:
        return "long_answer"
    if ROLE_MARKER.search(answer):
        return "role_marker"
    return None


def choose_screened_diverse(messages: list[dict], max_en: int = 800,
                            max_de: int = 400) -> tuple[dict[str, list[dict]], dict]:
    """Select at most one eligible answer per tree; SHA order, DE then EN.

    Exclusion criteria are *proposals*. Never use these results for training
    without an independent question/answer, licensing and privacy review.
    """
    if not 50 <= max_de <= 2000 or not 50 <= max_en <= 2000:
        raise ValueError("bounded language caps must be 50..2000")
    pairs = choose_pairs(messages, max_en=None, max_de=None)
    source = {}
    for row in messages:
        ident = row.get("message_id") if isinstance(row, dict) else None
        if not isinstance(ident, str) or not ident:
            raise ValueError("missing upstream message ID")
        if ident in source:
            raise ValueError("duplicate upstream message ID")
        source[ident] = row
    groups: dict[str, dict[str, list[dict]]] = {"de": {}, "en": {}}
    rejected = Counter()
    eligible_rows = Counter()
    for lang in ("de", "en"):
        for pair in pairs[lang]:
            ident = pair.get("id")
            if not isinstance(ident, str) or not ident.startswith("oasst2-"):
                raise ValueError("unexpected source pair identity")
            child = source.get(ident[len("oasst2-"):])
            parent = source.get(pair.get("upstream_parent_id"))
            if (child is None or parent is None or child.get("role") != "assistant"
                    or parent.get("role") != "prompter"
                    or child.get("parent_id") != parent.get("message_id")
                    or not reviewed_human(child) or not reviewed_human(parent)
                    or child.get("lang") != lang or parent.get("lang") != lang
                    or not isinstance(child.get("text"), str)
                    or not isinstance(parent.get("text"), str)
                    or child["text"].strip() != pair.get("answer")
                    or parent["text"].strip() != pair.get("prompt")):
                raise ValueError("source pair lineage or reviewed-human identity changed")
            tree = child.get("message_tree_id")
            if not isinstance(tree, str) or not tree.strip() or tree != parent.get("message_tree_id"):
                raise ValueError("invalid conversation tree lineage")
            reason = screen_reason(pair["prompt"], pair["answer"])
            if reason:
                rejected[reason] += 1
                continue
            eligible_rows[lang] += 1
            groups[lang].setdefault(tree, []).append(pair)
    result: dict[str, list[dict]] = {"de": [], "en": []}
    used_trees: set[str] = set()
    used_prompts: set[str] = set()
    collisions = 0
    for lang, cap in (("de", max_de), ("en", max_en)):
        for tree in sorted(groups[lang], key=lambda name: (digest("tree:" + name), name)):
            if tree in used_trees:
                continue
            for pair in sorted(groups[lang][tree], key=lambda row: (digest("pair:" + row["id"]), row["id"])):
                question_hash = prompt_hash(pair["prompt"])
                if question_hash in used_prompts:
                    collisions += 1
                    continue
                used_prompts.add(question_hash)
                used_trees.add(tree)
                result[lang].append(pair)
                break
            if len(result[lang]) == cap:
                break
        if len(result[lang]) != cap:
            raise ValueError("insufficient distinct eligible trees/prompts after conservative screen")
    ids = sorted(pair["id"] for lang in ("de", "en") for pair in result[lang])
    return result, {"policy": POLICY,
                    "selected_de": len(result["de"]), "selected_en": len(result["en"]),
                    "selected_distinct_trees": len(used_trees),
                    "selected_inventory_sha256": digest("\n".join(ids)),
                    "upstream_eligible_rows": dict(eligible_rows),
                    "rejected_upstream_pairs_by_first_reason": dict(sorted(rejected.items())),
                    "duplicate_prompts_avoided_during_selection": collisions,
                    "holdout_checked": False, "privacy_approved": False,
                    "license_approved": False, "answer_quality_reviewed": False,
                    "training_approved": False, "training_dataset_exported": False,
                    "training_started": False}
