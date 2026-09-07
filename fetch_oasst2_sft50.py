"""Fetch and convert a pinned human-only OpenAssistant/oasst2 SFT source.

OASST2 contains both human and synthetic messages. Neral deliberately keeps only positively
reviewed, non-deleted, ``synthetic == false`` prompter -> assistant pairs in German/English.
No user IDs are copied into the prepared JSONL. The upstream revision and ready-export archive
SHA-256 are pinned so a future upstream change cannot silently change Neral's training data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
import unicodedata
from pathlib import Path

from hf_source_policy50 import assert_training_allowed
from fetch_oasst1_sft50 import infer_domain

DATASET_ID = "OpenAssistant/oasst2"
UPSTREAM_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
ARCHIVE_NAME = "2023-11-05_oasst2_ready.messages.jsonl.gz"
UPSTREAM_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{UPSTREAM_REVISION}/{ARCHIVE_NAME}"
UPSTREAM_SHA256 = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}/tree/{UPSTREAM_REVISION}"
LICENSE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/blob/{UPSTREAM_REVISION}/README.md"
USER_AGENT = "NeralDataBuilder/2.0 (+local reproducible human-only dataset preparation)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as response, path.open("wb") as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            out.write(chunk)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def reviewed_human(row: dict) -> bool:
    return (
        row.get("deleted") is not True
        and row.get("synthetic") is False
        and row.get("review_result") is True
        and not row.get("model_name")
    )


def choose_pairs(rows, max_en: int, max_de: int):
    messages = {r.get("message_id"): r for r in rows if isinstance(r.get("message_id"), str)}
    grouped = {}
    for row in rows:
        if row.get("role") != "assistant" or row.get("lang") not in {"de", "en"} or not reviewed_human(row):
            continue
        parent = messages.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter" or parent.get("lang") != row.get("lang") or not reviewed_human(parent):
            continue
        prompt, answer = parent.get("text"), row.get("text")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            continue
        prompt, answer = prompt.strip(), answer.strip()
        if not prompt or not answer:
            continue
        rank = row.get("rank")
        rank = 10**9 if rank is None else int(rank)
        key = (row["lang"], parent["message_id"])
        candidate = (rank, row["message_id"], prompt, answer)
        if key not in grouped or candidate[:2] < grouped[key][:2]:
            grouped[key] = candidate

    out = {"de": [], "en": []}
    seen_pairs = set()
    for (lang, parent_id), (rank, message_id, prompt, answer) in sorted(grouped.items()):
        pair_key = (normalize(prompt), normalize(answer))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        out[lang].append({
            "id": f"oasst2-{message_id}",
            "prompt": prompt,
            "answer": answer,
            "domain": infer_domain(prompt, answer),
            "upstream_parent_id": parent_id,
            "upstream_rank": None if rank == 10**9 else rank,
        })
    out["en"] = out["en"][:max_en]
    out["de"] = out["de"][:max_de]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft50_sources")
    ap.add_argument("--max-en", type=int, default=16000)
    ap.add_argument("--max-de", type=int, default=6000)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()
    if min(args.max_en, args.max_de) <= 0:
        raise ValueError("positive language caps required")

    policy = assert_training_allowed(DATASET_ID)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / ARCHIVE_NAME
    if args.force_download or not raw.is_file():
        download(UPSTREAM_URL, raw)
    upstream_sha = sha256_file(raw)
    if upstream_sha != UPSTREAM_SHA256:
        raise ValueError(f"Pinned OASST2 archive SHA-256 mismatch: {upstream_sha}")

    rows = []
    with gzip.open(raw, "rt", encoding="utf8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    pairs = choose_pairs(rows, args.max_en, args.max_de)
    counts = {lang: len(value) for lang, value in pairs.items()}
    if sum(counts.values()) < 3000:
        raise ValueError(f"Too few usable human OASST2 pairs: {counts}")
    if counts.get("de", 0) < 500:
        raise ValueError(f"Too few usable German OASST2 pairs: {counts}")

    safe_train = {lang: max(1, int(n * 0.75)) for lang, n in counts.items() if n}
    safe_total = sum(safe_train.values())
    sources = []
    for lang in ("de", "en"):
        if not pairs[lang]:
            continue
        path = out / f"oasst2_{lang}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs[lang]), encoding="utf8")
        sources.append({
            "id": f"oasst2-human-{lang}",
            "file": path.name,
            "sha256": sha256_file(path),
            "upstream_sha256": upstream_sha,
            "upstream_revision": UPSTREAM_REVISION,
            "license": policy.license,
            "license_url": LICENSE_URL,
            "provenance_url": DATASET_URL,
            "language": lang,
            "domain": "chat",
            "allow_row_domains": True,
            "target_fraction": safe_train[lang] / safe_total,
            "human_only_filter": True,
        })

    manifest = {
        "version": 1,
        "recommended_train_examples": safe_total,
        "dataset_id": DATASET_ID,
        "upstream_download_url": UPSTREAM_URL,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sha256": upstream_sha,
        "filters": "positively reviewed, non-deleted, synthetic=false, no model_name; deterministic best rank per prompt; de/en only",
        "external_model_calls": False,
        "sources": sources,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "download_sha256": upstream_sha,
        "upstream_revision": UPSTREAM_REVISION,
        "pairs": counts,
        "recommended_train_examples": safe_total,
        "domains": {
            lang: {d: sum(r["domain"] == d for r in pairs[lang]) for d in sorted({x["domain"] for x in pairs[lang]})}
            for lang in pairs
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
