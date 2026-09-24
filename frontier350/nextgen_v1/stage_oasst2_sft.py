"""Stage pinned OASST2 human-only DE/EN candidates for Phoenix review.

This script is deliberately non-approving and non-training. It reads only the exact
locally present OASST2 ready export, verifies its pinned archive SHA-256, excludes
synthetic/model-authored/deleted/unreviewed messages, screens exact frozen chat prompts,
and writes a fresh REVIEW_REQUIRED candidate directory. It never downloads data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from stage_oasst1_sft import frozen_chat_hashes, permitted, prompt_hash, sha_file

REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
SOURCE_NAME = "2023-11-05_oasst2_ready.messages.jsonl.gz"
SOURCE_SHA256 = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
SOURCE_URL = f"https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/{REVISION}/{SOURCE_NAME}"
MAX_MESSAGES = 200_000


def select(source: Path, frozen: set[str], *, train_limit: int = 2400,
           valid_limit: int = 320, min_de_train: int = 40,
           min_de_valid: int = 10) -> tuple[dict, dict]:
    if source.name != SOURCE_NAME or not frozen:
        raise ValueError("pinned OASST2 source filename and nonempty frozen registry required")
    if sha_file(source) != SOURCE_SHA256:
        raise ValueError("pinned OASST2 archive SHA-256 mismatch")

    roots: dict[str, dict] = {}
    replies = []
    messages = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            messages += 1
            if messages > MAX_MESSAGES:
                raise ValueError("unexpected OASST2 export length")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("invalid OASST2 record")
            if row.get("parent_id") is None and permitted(row, "prompter"):
                roots[row["message_id"]] = row
            elif (row.get("parent_id") is not None and row.get("rank") == 0
                  and permitted(row, "assistant")):
                replies.append(row)

    by_split = {
        "train": {"de": [], "en": []},
        "validation": {"de": [], "en": []},
    }
    seen_prompts: set[str] = set()
    for reply in sorted(replies, key=lambda row: row["message_id"]):
        parent = roots.get(reply["parent_id"])
        if not parent or reply.get("lang") != parent.get("lang"):
            continue
        if reply.get("message_tree_id") != parent.get("message_tree_id"):
            continue
        prompt = parent["text"].strip()
        digest = prompt_hash(prompt)
        if digest in frozen or digest in seen_prompts:
            continue
        seen_prompts.add(digest)
        tree = str(parent["message_tree_id"])
        split = "validation" if int(hashlib.sha256(tree.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        by_split[split][reply["lang"]].append({
            "id": reply["message_id"],
            "source_id": "oasst2-ready-2023-11-05",
            "user": prompt,
            "assistant": reply["text"].strip(),
        })

    result = {}
    for split, cap, minimum_de in (
        ("train", train_limit, min_de_train),
        ("validation", valid_limit, min_de_valid),
    ):
        de = by_split[split]["de"][: cap // 4]
        en = by_split[split]["en"][: cap - len(de)]
        if len(de) < minimum_de or not en:
            raise ValueError(f"not enough reviewed OASST2 DE/EN {split} candidates")
        rows = sorted([*de, *en], key=lambda r: r["id"])
        if not rows:
            raise ValueError(f"empty OASST2 {split} split")
        result[split] = rows

    train_ids = {row["id"] for row in result["train"]}
    valid_ids = {row["id"] for row in result["validation"]}
    if train_ids & valid_ids:
        raise ValueError("OASST2 train/validation identity leakage")

    counts = {
        split: {
            lang: sum(
                row["id"] in {r["id"] for r in by_split[split][lang]}
                for row in result[split]
            )
            for lang in ("de", "en")
        }
        for split in result
    }
    return result, {
        "source_messages": messages,
        "candidate_counts": counts,
        "frozen_chat_hashes_checked": len(frozen),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--frozen-benchmark", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refuse to reuse any staging directory")

    frozen = frozen_chat_hashes(args.frozen_benchmark)
    selected, stats = select(args.source, frozen)
    args.out.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        with (args.out / f"candidate_{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema": 1,
        "status": "REVIEW_REQUIRED_NOT_APPROVED_NOT_TRAINED",
        "source_id": "OpenAssistant/oasst2",
        "source_revision": REVISION,
        "source_uri": SOURCE_URL,
        "source_filename": args.source.name,
        "source_sha256": SOURCE_SHA256,
        "license_declared_by_publisher": "Apache-2.0",
        "filter": "reviewed ready human-only rank-0 root prompt/assistant, de/en",
        "benchmark_screening": "exact frozen chat prompts only; semantic/public-suite screening still required",
        "independent_privacy_and_license_review": False,
        "approved_for_training": False,
        **stats,
        "output_sha256": {
            split: sha_file(args.out / f"candidate_{split}.jsonl")
            for split in selected
        },
        "warning": "Staging is not approval. Do not create a training manifest until independent review and holdout/public-benchmark screening are complete.",
    }
    (args.out / "STAGING_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "source_uri"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
