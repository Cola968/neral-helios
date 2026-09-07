"""Deterministic data selection for Neral-50M G1 generation repair.

G1 does not invent examples and does not move normalized prompt groups between splits. It only
selects a bounded, answer-length-focused subset from an already validated SFT50 dataset. The
selection is deterministic from immutable pair hashes so reruns use the same rows.
"""
from __future__ import annotations

import hashlib
from collections import Counter


def formatted_training_tokens(row: dict) -> int:
    """Exact byte-token length used by train_instruct50.encode_sft, including BOS/EOS."""
    prompt = str(row["prompt"]).strip()
    answer = str(row["answer"]).strip()
    prefix = ("User: " + prompt + "\nAssistant:").encode("utf8")
    target = (" " + answer + "\n").encode("utf8")
    return 1 + len(prefix) + len(target) + 1


def eligible_rows(rows: list[dict], *, seq: int, min_answer_bytes: int = 12, max_answer_bytes: int = 384) -> list[dict]:
    if seq <= 0 or min_answer_bytes <= 0 or max_answer_bytes < min_answer_bytes:
        raise ValueError("invalid G1 data bounds")
    out=[]
    for row in rows:
        if row.get("language") not in {"de","en"}:continue
        if not isinstance(row.get("prompt"),str) or not isinstance(row.get("answer"),str):continue
        answer_bytes=int(row.get("answer_bytes",len(row["answer"].encode("utf8"))))
        if not min_answer_bytes <= answer_bytes <= max_answer_bytes:continue
        if formatted_training_tokens(row)-1 > seq:continue
        pair=row.get("pair_sha256")
        group=row.get("prompt_group_sha256")
        if not isinstance(pair,str) or len(pair)!=64 or not isinstance(group,str) or len(group)!=64:continue
        out.append(row)
    return out


def _order_key(row: dict) -> str:
    return hashlib.sha256(("NERAL50:G1:"+row["pair_sha256"]).encode()).hexdigest()


def select_language_mix(rows: list[dict], *, max_examples: int, desired_de_fraction: float = .20, min_examples: int = 1) -> list[dict]:
    """Select a stable bounded set, preferring the requested DE fraction when availability permits."""
    if max_examples <= 0 or min_examples <= 0 or min_examples > max_examples:
        raise ValueError("invalid example bounds")
    if not 0 <= desired_de_fraction <= 1:raise ValueError("desired_de_fraction must be 0..1")
    de=sorted((r for r in rows if r.get("language")=="de"),key=_order_key)
    en=sorted((r for r in rows if r.get("language")=="en"),key=_order_key)
    target=min(max_examples,len(de)+len(en))
    if target < min_examples:raise ValueError(f"Only {target} eligible G1 examples; need at least {min_examples}")
    de_quota=min(len(de),round(target*desired_de_fraction))
    en_quota=min(len(en),target-de_quota)
    remaining=target-de_quota-en_quota
    if remaining:
        extra_de=min(remaining,len(de)-de_quota);de_quota+=extra_de;remaining-=extra_de
    if remaining:
        extra_en=min(remaining,len(en)-en_quota);en_quota+=extra_en;remaining-=extra_en
    if remaining:raise AssertionError("G1 selection quota could not be filled")
    selected=de[:de_quota]+en[:en_quota]
    selected.sort(key=_order_key)
    return selected


def selection_digest(rows: list[dict]) -> str:
    h=hashlib.sha256()
    for pair in sorted(str(r["pair_sha256"]) for r in rows):
        h.update(pair.encode());h.update(b"\n")
    return h.hexdigest()


def selection_summary(rows: list[dict]) -> dict:
    counts=Counter(r.get("language") for r in rows)
    domains=Counter(r.get("domain") for r in rows)
    answer_bytes=[int(r.get("answer_bytes",len(str(r.get("answer","")).encode("utf8")))) for r in rows]
    return {
        "examples":len(rows),"de":counts["de"],"en":counts["en"],
        "de_fraction":counts["de"]/max(1,len(rows)),
        "domains":dict(sorted(domains.items())),
        "answer_bytes_mean":sum(answer_bytes)/max(1,len(answer_bytes)),
        "answer_bytes_max":max(answer_bytes,default=0),
        "selection_sha256":selection_digest(rows),
    }
