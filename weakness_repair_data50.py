"""Deterministic domain-balanced selector for Neral-50M I2 weakness repair.

I2 reuses only the already contamination-screened I1 SFT dataset. It does not create new
answers and never moves prompt groups between train/validation splits. Selection is stable
from immutable pair hashes and deliberately raises the share of math/code while keeping
reasoning, concise-format and general instruction anchors.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

PROFILE = {
    "math": 0.40,
    "code": 0.25,
    "reasoning": 0.10,
    "concise": 0.10,
    "general": 0.15,
}
CONCISE_DOMAINS = {"format", "translation", "summary"}


def bucket_for(row: dict) -> str:
    domain = str(row.get("domain", ""))
    if domain in {"math", "code", "reasoning"}:
        return domain
    if domain in CONCISE_DOMAINS:
        return "concise"
    return "general"


def formatted_training_tokens(row: dict) -> int:
    prompt = str(row["prompt"]).strip()
    answer = str(row["answer"]).strip()
    prefix = ("User: " + prompt + "\nAssistant:").encode("utf8")
    target = (" " + answer + "\n").encode("utf8")
    return 1 + len(prefix) + len(target) + 1


def eligible_rows(rows: list[dict], *, seq: int, min_answer_bytes: int = 2, max_answer_bytes: int = 640) -> list[dict]:
    if seq <= 0 or min_answer_bytes <= 0 or max_answer_bytes < min_answer_bytes:
        raise ValueError("invalid I2 data bounds")
    out = []
    for row in rows:
        if row.get("language") not in {"de", "en"}:
            continue
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
            continue
        pair = row.get("pair_sha256")
        group = row.get("prompt_group_sha256")
        if not isinstance(pair, str) or len(pair) != 64 or not isinstance(group, str) or len(group) != 64:
            continue
        answer_bytes = int(row.get("answer_bytes", len(row["answer"].encode("utf8"))))
        if not min_answer_bytes <= answer_bytes <= max_answer_bytes:
            continue
        if formatted_training_tokens(row) - 1 > seq:
            continue
        out.append(row)
    return out


def _key(row: dict) -> str:
    return hashlib.sha256(("NERAL50:I2:WEAKNESS:" + row["pair_sha256"]).encode()).hexdigest()


def unique_prompt_groups(rows: list[dict]) -> list[dict]:
    """Choose exactly one deterministic pair per normalized prompt group.

    The prepared SFT format intentionally permits multiple acceptable answers for one prompt
    within the same split. I2 samples prompt groups, not answer variants, so one frequent prompt
    cannot receive accidental extra weight and the selected batch stays group-disjoint.
    """
    chosen = {}
    for row in sorted(rows, key=_key):
        chosen.setdefault(row["prompt_group_sha256"], row)
    return list(chosen.values())


def _take_bilingual(rows: list[dict], quota: int, de_fraction: float) -> list[dict]:
    if quota <= 0:
        return []
    de = sorted((r for r in rows if r.get("language") == "de"), key=_key)
    en = sorted((r for r in rows if r.get("language") == "en"), key=_key)
    de_q = min(len(de), round(quota * de_fraction))
    en_q = min(len(en), quota - de_q)
    remain = quota - de_q - en_q
    if remain:
        add_de = min(remain, len(de) - de_q)
        de_q += add_de
        remain -= add_de
    if remain:
        add_en = min(remain, len(en) - en_q)
        en_q += add_en
        remain -= add_en
    return de[:de_q] + en[:en_q]


def select_profile_mix(
    rows: list[dict], *, max_examples: int, min_examples: int = 1,
    desired_de_fraction: float = 0.35, profile: dict[str, float] | None = None,
) -> list[dict]:
    if max_examples <= 0 or min_examples <= 0 or min_examples > max_examples:
        raise ValueError("invalid I2 example bounds")
    if not 0 <= desired_de_fraction <= 1:
        raise ValueError("desired_de_fraction must be 0..1")
    profile = dict(PROFILE if profile is None else profile)
    if set(profile) != set(PROFILE) or any(v < 0 for v in profile.values()) or abs(sum(profile.values()) - 1.0) > 1e-9:
        raise ValueError("invalid I2 profile")

    rows = unique_prompt_groups(rows)
    by = defaultdict(list)
    for row in rows:
        by[bucket_for(row)].append(row)
    target = min(max_examples, sum(len(v) for v in by.values()))
    if target < min_examples:
        raise ValueError(f"Only {target} unique eligible I2 prompt groups; need {min_examples}")

    quotas = {}
    remaining = target
    names = list(PROFILE)
    for i, name in enumerate(names):
        q = remaining if i == len(names) - 1 else round(target * profile[name])
        q = min(q, len(by[name]))
        quotas[name] = q
        remaining -= q

    fill_order = ["math", "code", "reasoning", "concise", "general"]
    while remaining > 0:
        progressed = False
        for name in fill_order:
            spare = len(by[name]) - quotas[name]
            if spare <= 0:
                continue
            take = min(spare, remaining)
            quotas[name] += take
            remaining -= take
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    if remaining:
        raise ValueError(f"Could not fill I2 selection: {remaining} examples missing")

    selected = []
    for name in names:
        selected.extend(_take_bilingual(by[name], quotas[name], desired_de_fraction))
    if len(selected) != target:
        raise AssertionError(f"I2 selected {len(selected)} rows, expected {target}")
    selected.sort(key=_key)
    groups = [r["prompt_group_sha256"] for r in selected]
    if len(groups) != len(set(groups)):
        raise AssertionError("I2 prompt-group deduplication failed")
    return selected


def selection_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for pair in sorted(str(r["pair_sha256"]) for r in rows):
        h.update(pair.encode()); h.update(b"\n")
    return h.hexdigest()


def selection_summary(rows: list[dict]) -> dict:
    languages = Counter(r.get("language") for r in rows)
    domains = Counter(r.get("domain") for r in rows)
    buckets = Counter(bucket_for(r) for r in rows)
    answer_bytes = [int(r.get("answer_bytes", len(str(r.get("answer", "")).encode("utf8")))) for r in rows]
    return {
        "examples": len(rows),
        "prompt_groups": len({r["prompt_group_sha256"] for r in rows}),
        "languages": dict(sorted(languages.items())),
        "domains": dict(sorted(domains.items())),
        "buckets": dict(sorted(buckets.items())),
        "de_fraction": languages["de"] / max(1, len(rows)),
        "answer_bytes_mean": sum(answer_bytes) / max(1, len(answer_bytes)),
        "answer_bytes_max": max(answer_bytes, default=0),
        "selection_sha256": selection_digest(rows),
    }
