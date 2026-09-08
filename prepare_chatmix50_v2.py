from __future__ import annotations

"""Build Neral-50M ChatMix V2 from approved I1 + pinned OASST2 multi-turn data.

Training is deliberately human-majority. All clean human conversations are retained first; the
existing deterministic targeted set is then capped so >=60% of training conversations are human.
Validation/test/holdout retain all clean rows to keep broad capability diagnostics difficult.
No external model-generated chat corpus is added.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from chat50_v2 import expand_conversation, validate_messages
from fetch_oasst2_sft50 import normalize, sha256_file

SPLITS = ("train", "val", "test", "holdout")
HUMAN_TRAIN_FRACTION_MIN = 0.60


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def single_to_chat(row: dict) -> dict:
    source = str(row.get("source_id", "i1-unknown"))
    human = source.startswith("oasst")
    return {
        "id": "i1-" + str(row["id"]),
        "source": source,
        "source_class": "human" if human else "targeted-deterministic",
        "language": row.get("language", "unknown"),
        "domain": row.get("domain", "chat"),
        "messages": [
            {"role": "user", "content": str(row["prompt"]).strip()},
            {"role": "assistant", "content": str(row["answer"]).strip()},
        ],
    }


def chat_key(row: dict) -> str:
    validate_messages(row["messages"])
    text = "\n".join(f"{m['role']}:{normalize(m['content'])}" for m in row["messages"])
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def enrich(row: dict, max_seq: int) -> dict | None:
    examples = expand_conversation(row, max_seq=max_seq, include_all_turns=True)
    if not examples:
        return None
    item = dict(row)
    item["chat_sha256"] = chat_key(row)
    item["prefix_examples"] = len(examples)
    item["max_encoded_length"] = max(ex.length for ex in examples)
    return item


def human_majority_train(rows: list[dict], fraction: float = HUMAN_TRAIN_FRACTION_MIN) -> tuple[list[dict], dict]:
    human = sorted((r for r in rows if r["source_class"] == "human"), key=lambda r: r["chat_sha256"])
    targeted = sorted((r for r in rows if r["source_class"] != "human"), key=lambda r: r["chat_sha256"])
    if not human:
        raise ValueError("ChatMix V2 has no human training conversations")
    # human / (human + targeted) >= fraction  => targeted <= human*(1-f)/f
    max_targeted = math.floor(len(human) * (1.0 - fraction) / fraction)
    kept_targeted = targeted[:max_targeted]
    result = human + kept_targeted
    result.sort(key=lambda r: (r["source_class"], r["source"], r["chat_sha256"]))
    actual = len(human) / max(1, len(result))
    if actual + 1e-12 < fraction:
        raise AssertionError("human-majority quota failed")
    return result, {
        "human_available": len(human),
        "targeted_available": len(targeted),
        "targeted_kept": len(kept_targeted),
        "targeted_deferred": len(targeted) - len(kept_targeted),
        "human_fraction": actual,
        "minimum_human_fraction": fraction,
    }


def build(i1_root: Path, multiturn_root: Path, out: Path, max_seq: int) -> dict:
    if not 512 <= max_seq <= 4096:
        raise ValueError("max_seq must be 512..4096")
    out.mkdir(parents=True, exist_ok=True)
    reject = collections.Counter()
    locations: dict[str, str] = {}
    summaries = {}
    train_selection = None
    source_counts = collections.Counter()

    i1_manifest_path = i1_root / "manifest.json"
    mt_manifest_path = multiturn_root / "manifest.json"
    i1_manifest = json.loads(i1_manifest_path.read_text(encoding="utf8"))
    mt_manifest = json.loads(mt_manifest_path.read_text(encoding="utf8"))
    if not mt_manifest.get("benchmark_literal_screen"):
        raise ValueError("multi-turn source lacks benchmark contamination screen")
    if mt_manifest.get("external_model_calls") is not False:
        raise ValueError("multi-turn source provenance contract failed")

    for split in SPLITS:
        candidates: list[dict] = []
        # Rich human histories win exact-dedup priority over flattened copies.
        for row in read_jsonl(multiturn_root / f"{split}.jsonl"):
            row = dict(row); row.pop("split", None); row["source_class"] = "human"
            candidates.append(row)
        candidates.extend(single_to_chat(r) for r in read_jsonl(i1_root / f"{split}.jsonl"))

        clean = []
        local = set()
        for row in candidates:
            try:
                item = enrich(row, max_seq)
            except (KeyError, TypeError, ValueError):
                reject["invalid_chat"] += 1
                continue
            if item is None:
                reject["not_encodable_within_window"] += 1
                continue
            key = item["chat_sha256"]
            if key in local:
                reject["exact_duplicate_same_split"] += 1
                continue
            previous = locations.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"exact chat leakage across splits: {previous} -> {split}")
            local.add(key); locations[key] = split; clean.append(item)

        if split == "train":
            clean, train_selection = human_majority_train(clean)
        else:
            clean.sort(key=lambda r: (r["source_class"], r["source"], r["chat_sha256"]))

        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf8")
        source_counts.update(r["source"] for r in clean)
        human_n = sum(r["source_class"] == "human" for r in clean)
        summaries[split] = {
            "conversations": len(clean),
            "prefix_examples": sum(int(r["prefix_examples"]) for r in clean),
            "assistant_turns": sum(sum(m["role"] == "assistant" for m in r["messages"]) for r in clean),
            "human_conversations": human_n,
            "human_fraction": human_n / max(1, len(clean)),
            "targeted_deterministic_conversations": len(clean) - human_n,
            "de": sum(r.get("language") == "de" for r in clean),
            "en": sum(r.get("language") == "en" for r in clean),
            "sha256": sha256_file(path),
        }

    if summaries["train"]["conversations"] < 1000:
        raise ValueError("ChatMix V2 has too few training conversations")
    if summaries["train"]["human_fraction"] < HUMAN_TRAIN_FRACTION_MIN:
        raise ValueError("ChatMix V2 training set is not sufficiently human-majority")

    manifest = {
        "version": 2,
        "purpose": "Neral-50M post-I1 multi-turn chat specialization",
        "max_seq": max_seq,
        "format": "role-labelled history; each assistant turn expands to a separate assistant-only prefix target with terminal EOS",
        "inputs": {
            "i1_manifest_sha256": sha256_file(i1_manifest_path),
            "multiturn_manifest_sha256": sha256_file(mt_manifest_path),
            "i1_declared_target_train_examples": i1_manifest.get("target_train_examples"),
        },
        "sources": dict(source_counts),
        "splits": summaries,
        "train_selection": train_selection,
        "rejected": dict(reject),
        "cross_split_exact_chat_overlap": 0,
        "human_majority_required": True,
        "minimum_train_human_fraction": HUMAN_TRAIN_FRACTION_MIN,
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
    print(json.dumps(build(Path(args.i1_data), Path(args.multiturn_data), Path(args.out), args.max_seq), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
