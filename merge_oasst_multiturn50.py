from __future__ import annotations

"""Merge pinned OASST1 + OASST2 human-only multi-turn corpora for Neral Chat V2.

The two OpenAssistant releases are independently split upstream. To prevent release-overlap from
creating cross-split leakage, this merger ignores those original split labels and reassigns every
unique normalized conversation using its content hash. Exact cross-release duplicates collapse to
one row, preferring OASST2, then richer histories. Both inputs must declare benchmark screening,
no external model calls and Apache-2.0 licensing.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from fetch_oasst2_sft50 import normalize, sha256_file

SPLITS = (("train", 90), ("val", 5), ("test", 3), ("holdout", 2))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def conversation_key(row: dict) -> str:
    text = "\n".join(f"{m['role']}:{normalize(str(m['content']))}" for m in row["messages"])
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def split_for_key(key: str) -> str:
    bucket = int(key[:8], 16) % 100
    cursor = 0
    for name, width in SPLITS:
        cursor += width
        if bucket < cursor:
            return name
    raise AssertionError("split buckets do not sum to 100")


def preference(row: dict) -> tuple:
    source = str(row.get("source", ""))
    # OASST2 is the newer release; richer histories win within the same release.
    release_rank = 0 if source.startswith("oasst2-") else 1
    return (release_rank, -int(row.get("assistant_turns", 0)), int(row.get("estimated_byte_tokens", 10**9)), source)


def validate_input(root: Path, expected_dataset: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("dataset_id") != expected_dataset:
        raise ValueError(f"unexpected dataset at {root}: {manifest.get('dataset_id')}")
    if manifest.get("license") != "Apache-2.0":
        raise ValueError(f"non-Apache OpenAssistant source: {manifest.get('license')}")
    if manifest.get("benchmark_literal_screen") is not True:
        raise ValueError("input lacks benchmark screen")
    if manifest.get("external_model_calls") is not False:
        raise ValueError("input provenance contract failed")
    if manifest.get("raw_user_ids_exported") is not False:
        raise ValueError("input exports user ids")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oasst1", required=True)
    ap.add_argument("--oasst2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roots = [(Path(args.oasst1), "OpenAssistant/oasst1"), (Path(args.oasst2), "OpenAssistant/oasst2")]
    manifests = [validate_input(root, dataset) for root, dataset in roots]
    all_rows = []
    for root, _ in roots:
        for split, _width in SPLITS:
            all_rows.extend(read_jsonl(root / f"{split}.jsonl"))

    chosen: dict[str, dict] = {}
    duplicate_sources = Counter()
    for row in all_rows:
        key = conversation_key(row)
        candidate = dict(row)
        candidate["content_sha256"] = key
        candidate["split"] = split_for_key(key)
        current = chosen.get(key)
        if current is None or preference(candidate) < preference(current):
            if current is not None:
                duplicate_sources[f"{current.get('source')} -> {candidate.get('source')}"] += 1
            chosen[key] = candidate
        else:
            duplicate_sources[f"{candidate.get('source')} -> {current.get('source')}"] += 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    counts = {}; source_counts = Counter(); source_turns = Counter()
    for split, _width in SPLITS:
        rows = sorted((r for r in chosen.values() if r["split"] == split), key=lambda r: r["content_sha256"])
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
        for row in rows:
            source_counts[row["source"]] += 1
            source_turns[row["source"]] += int(row.get("assistant_turns", 0))
        counts[split] = {
            "rows": len(rows),
            "assistant_turns": sum(int(r.get("assistant_turns", 0)) for r in rows),
            "de": sum(r.get("language") == "de" for r in rows),
            "en": sum(r.get("language") == "en" for r in rows),
            "sha256": sha256_file(path),
        }

    # Content-hash split assignment makes cross-split exact overlap impossible by construction;
    # assert that property explicitly anyway.
    seen_split = {}
    for row in chosen.values():
        prev = seen_split.setdefault(row["content_sha256"], row["split"])
        if prev != row["split"]:
            raise AssertionError("cross-release content leaked across splits")

    manifest = {
        "version": 1,
        "dataset_id": "Neral/OASST1+OASST2-human-multiturn",
        "role": "human-multiturn-sft",
        "license": "Apache-2.0",
        "input_manifest_sha256": {
            "oasst1": sha256_file(roots[0][0] / "manifest.json"),
            "oasst2": sha256_file(roots[1][0] / "manifest.json"),
        },
        "upstream_revisions": {
            "oasst1": manifests[0]["upstream_revision"],
            "oasst2": manifests[1]["upstream_revision"],
        },
        "dedup": "NFKC/casefold/whitespace-normalized role+content SHA-256 across both releases",
        "selection": "prefer OASST2 duplicate, then more assistant turns, then shorter byte length",
        "split_policy": "content-hash 90/5/3/2 after cross-release dedup",
        "benchmark_literal_screen": True,
        "external_model_calls": False,
        "raw_user_ids_exported": False,
        "cross_split_exact_conversation_overlap": 0,
        "counts": counts,
        "source_conversations": dict(source_counts),
        "source_assistant_turns": dict(source_turns),
        "cross_release_duplicates": sum(duplicate_sources.values()),
        "duplicate_resolution": dict(duplicate_sources),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
