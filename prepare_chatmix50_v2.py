from __future__ import annotations

"""Assemble Neral-50M ChatMix V2 without relaxing the project's data policy.

Inputs:
1. the already contamination-screened I1 SFT split (human OASST2 + deterministic targeted data),
2. the new pinned human-only OASST2 multi-turn split.

No external synthetic assistant corpus is silently added. Multi-turn human data is preferred when
an exact formatted duplicate exists. All split assignments are retained and exact normalized chat
fingerprints are checked globally so a conversation cannot leak across train/val/test/holdout.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from fetch_oasst2_sft50 import normalize, sha256_file
from chat50_v2 import TOKENIZER, expand_conversation, validate_messages

SPLITS = ("train", "val", "test", "holdout")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def single_to_chat(row: dict) -> dict:
    source_id = str(row.get("source_id", "i1-unknown"))
    human = source_id.startswith("oasst")
    return {
        "id": "i1-" + str(row["id"]),
        "source": source_id,
        "source_class": "human" if human else "targeted-deterministic",
        "language": row.get("language", "unknown"),
        "domain": row.get("domain", "chat"),
        "messages": [
            {"role": "user", "content": str(row["prompt"]).strip()},
            {"role": "assistant", "content": str(row["answer"]).strip()},
        ],
    }


def formatted_key(row: dict) -> str:
    validate_messages(row["messages"])
    text = "\n".join(f"{m['role']}:{normalize(m['content'])}" for m in row["messages"])
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def validate_encodable(row: dict, max_seq: int) -> tuple[bool, int, int]:
    examples = expand_conversation(row, max_seq=max_seq, include_all_turns=True)
    if not examples:
        return False, 0, 0
    return True, len(examples), max(ex.length for ex in examples)


def build(i1_root: Path, multiturn_root: Path, out: Path, max_seq: int) -> dict:
    if not 512 <= max_seq <= 4096:
        raise ValueError("max_seq must be 512..4096")
    out.mkdir(parents=True, exist_ok=True)
    reject = collections.Counter()
    global_locations: dict[str, str] = {}
    summaries = {}
    all_written = []

    i1_manifest = json.loads((i1_root / "manifest.json").read_text(encoding="utf8"))
    mt_manifest = json.loads((multiturn_root / "manifest.json").read_text(encoding="utf8"))
    if not mt_manifest.get("benchmark_literal_screen"):
        raise ValueError("multi-turn source lacks benchmark contamination screen")
    if mt_manifest.get("external_model_calls") is not False:
        raise ValueError("multi-turn source provenance contract failed")

    for split in SPLITS:
        candidates: list[dict] = []
        # Human multi-turn rows first so exact duplicates from flat I1 do not replace richer context.
        for row in read_jsonl(multiturn_root / f"{split}.jsonl"):
            row = dict(row)
            row.pop("split", None)
            row["source_class"] = "human"
            candidates.append(row)
        candidates.extend(single_to_chat(r) for r in read_jsonl(i1_root / f"{split}.jsonl"))

        chosen = []
        local = set()
        for row in candidates:
            try:
                key = formatted_key(row)
            except (KeyError, TypeError, ValueError):
                reject["invalid_chat"] += 1
                continue
            if key in local:
                reject["exact_duplicate_same_split"] += 1
                continue
            other = global_locations.get(key)
            if other is not None and other != split:
                raise ValueError(f"exact chat leakage across splits: {other} -> {split}")
            ok, prefix_examples, max_length = validate_encodable(row, max_seq)
            if not ok:
                reject["not_encodable_within_window"] += 1
                continue
            local.add(key)
            global_locations[key] = split
            item = dict(row)
            item["chat_sha256"] = key
            item["prefix_examples"] = prefix_examples
            item["max_encoded_length"] = max_length
            chosen.append(item)

        # Stable order makes corpus hashes and resume signatures reproducible.
        chosen.sort(key=lambda r: (r.get("source_class", ""), r.get("source", ""), r["chat_sha256"]))
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf8")
        summaries[split] = {
            "conversations": len(chosen),
            "prefix_examples": sum(int(r["prefix_examples"]) for r in chosen),
            "assistant_turns": sum(sum(m["role"] == "assistant" for m in r["messages"]) for r in chosen),
            "human_conversations": sum(r.get("source_class") == "human" for r in chosen),
            "targeted_deterministic_conversations": sum(r.get("source_class") == "targeted-deterministic" for r in chosen),
            "de": sum(r.get("language") == "de" for r in chosen),
            "en": sum(r.get("language") == "en" for r in chosen),
            "sha256": sha256_file(path),
        }
        all_written.extend((split, r) for r in chosen)

    if summaries["train"]["conversations"] < 1000:
        raise ValueError("ChatMix V2 has too few training conversations")
    if summaries["train"]["human_conversations"] <= summaries["train"]["targeted_deterministic_conversations"]:
        raise ValueError("ChatMix V2 must remain majority human conversations")

    source_counts = collections.Counter(r.get("source", "unknown") for _, r in all_written)
    manifest = {
        "version": 2,
        "purpose": "Neral-50M post-I1 multi-turn chat specialization",
        "max_seq": max_seq,
        "format": "role-labelled history; each assistant turn expands to a separate assistant-only prefix target with terminal EOS",
        "inputs": {
            "i1_manifest_sha256": sha256_file(i1_root / "manifest.json"),
            "multiturn_manifest_sha256": sha256_file(multiturn_root / "manifest.json"),
        },
        "sources": dict(source_counts),
        "splits": summaries,
        "rejected": dict(reject),
        "cross_split_exact_chat_overlap": 0,
        "human_majority_required": True,
        "external_model_generated_chat_added": False,
        "benchmark_holdouts_added": False,
        "tokenizer": "Neral ByteTokenizer vocab=260",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i1-data", required=True)
    ap.add_argument("--multiturn-data", required=True)
    ap.add_argument("--out", default="chat50_v2_data")
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()
    result = build(Path(args.i1_data), Path(args.multiturn_data), Path(args.out), args.max_seq)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
