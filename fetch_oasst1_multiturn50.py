from __future__ import annotations

"""Build a pinned, human-only multi-turn OASST1 corpus for Neral-50M Chat V2.

OASST1 is an Apache-2.0 human crowdsourced release already approved by Neral's source policy.
This builder mirrors the hardened OASST2 multi-turn path while keeping OASST1's pinned archive,
review semantics and provenance. Raw user identifiers are never exported.
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from fetch_oasst1_sft50 import (
    UPSTREAM_REVISION,
    UPSTREAM_URL,
    UPSTREAM_SHA256,
    DATASET_URL,
    LICENSE_URL,
    download,
    reviewed_human,
    sha256_file,
)
from fetch_oasst2_sft50 import normalize
from hf_source_policy50 import assert_training_allowed
import prepare_sft50 as sft_base
from prepare_sft50_i1 import targeted_literals
from tokenizer import ByteTokenizer

DATASET_ID = "OpenAssistant/oasst1"
ARCHIVE_NAME = "2023-04-12_oasst_ready.messages.jsonl.gz"
TOKENIZER = ByteTokenizer()
SPLIT_BUCKETS = (("train", 90), ("val", 5), ("test", 3), ("holdout", 2))


def anon(value: str, prefix: str = "oasst1-mt") -> str:
    return f"{prefix}-" + hashlib.sha256(value.encode("utf8")).hexdigest()[:24]


def split_for_root(root_id: str) -> str:
    bucket = int(hashlib.sha256(root_id.encode("utf8")).hexdigest()[:8], 16) % 100
    cursor = 0
    for name, width in SPLIT_BUCKETS:
        cursor += width
        if bucket < cursor:
            return name
    raise AssertionError("split buckets do not sum to 100")


def is_reviewed_human(row: dict) -> bool:
    # OASST1 predates the stricter OASST2 model_name convention. If a model_name is present,
    # fail closed anyway; synthetic=true is already rejected by reviewed_human().
    return reviewed_human(row) and not row.get("model_name")


def trace_chain(messages: dict[str, dict], leaf: dict) -> list[dict] | None:
    chain: list[dict] = []
    row = leaf
    seen: set[str] = set()
    lang = leaf.get("lang")
    while row is not None:
        mid = row.get("message_id")
        if not isinstance(mid, str) or mid in seen:
            return None
        seen.add(mid)
        if row.get("lang") != lang or lang not in {"de", "en"} or not is_reviewed_human(row):
            return None
        role = row.get("role")
        if role not in {"prompter", "assistant"}:
            return None
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        chain.append(row)
        parent = row.get("parent_id")
        row = None if parent is None else messages.get(parent)
        if parent is not None and row is None:
            return None
    chain.reverse()
    if len(chain) < 4 or chain[0].get("role") != "prompter" or chain[-1].get("role") != "assistant":
        return None
    expected = "prompter"
    for item in chain:
        if item.get("role") != expected:
            return None
        expected = "assistant" if expected == "prompter" else "prompter"
    if len(chain) % 2:
        return None
    return chain


def chat_text(chain: list[dict]) -> str:
    return "\n".join(
        f"{'User' if row['role'] == 'prompter' else 'Assistant'}: {row['text'].strip()}"
        for row in chain
    )


def assistant_rank_cost(chain: list[dict]) -> int:
    total = 0
    for row in chain:
        if row.get("role") == "assistant":
            rank = row.get("rank")
            total += 1000 if rank is None else max(0, int(rank))
    return total


def chain_score(chain: list[dict]) -> tuple:
    return (-len(chain), assistant_rank_cost(chain), chain[-1]["message_id"])


def benchmark_phrases(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    phrases, files = sft_base.benchmark_literals(repo_root)
    extra, extra_files = targeted_literals(repo_root)
    return sorted(set(phrases) | set(extra), key=len, reverse=True), files | extra_files


def contaminated(text: str, phrases: list[str]) -> bool:
    norm = normalize(text)
    return any(p in norm for p in phrases if len(p) >= 20)


def build_rows(raw_rows: list[dict], repo_root: Path, max_assistant_turns: int, max_seq: int):
    messages = {r.get("message_id"): r for r in raw_rows if isinstance(r.get("message_id"), str)}
    phrases, benchmark_files = benchmark_phrases(repo_root)
    by_root: dict[str, list[list[dict]]] = collections.defaultdict(list)
    reject = collections.Counter()

    for row in raw_rows:
        if row.get("role") != "assistant" or row.get("lang") not in {"de", "en"}:
            continue
        if not is_reviewed_human(row):
            reject["assistant_not_reviewed_human"] += 1
            continue
        chain = trace_chain(messages, row)
        if chain is None:
            reject["invalid_or_mixed_chain"] += 1
            continue
        by_root[chain[0]["message_id"]].append(chain)

    selected = []
    seen: set[str] = set()
    for root_id, chains in sorted(by_root.items()):
        chain = min(chains, key=chain_score)
        max_messages = 2 * max_assistant_turns
        if len(chain) > max_messages:
            chain = chain[-max_messages:]
        text = chat_text(chain)
        if contaminated(text, phrases):
            reject["benchmark_contamination"] += 1
            continue
        key = hashlib.sha256(normalize(text).encode("utf8")).hexdigest()
        if key in seen:
            reject["exact_conversation_duplicate"] += 1
            continue
        seen.add(key)
        assistant_turns = len(chain) // 2
        estimated = len(TOKENIZER.encode(text, bos=True)) + assistant_turns
        if estimated > max_seq:
            reject["sequence_too_long"] += 1
            continue
        selected.append({
            "id": anon(root_id + "|" + chain[-1]["message_id"]),
            "tree_id": anon(root_id, "tree"),
            "language": chain[0]["lang"],
            "domain": "chat",
            "split": split_for_root(root_id),
            "messages": [
                {"role": "user" if r["role"] == "prompter" else "assistant", "content": r["text"].strip()}
                for r in chain
            ],
            "assistant_turns": assistant_turns,
            "estimated_byte_tokens": estimated,
            "source": "oasst1-human-multiturn",
        })
    return selected, reject, benchmark_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chat50_v2_oasst1")
    ap.add_argument("--max-assistant-turns", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()
    if not 2 <= args.max_assistant_turns <= 8:
        raise ValueError("max-assistant-turns must be 2..8")
    if not 512 <= args.max_seq <= 4096:
        raise ValueError("max-seq must be 512..4096")

    policy = assert_training_allowed(DATASET_ID)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = out / ARCHIVE_NAME
    if args.force_download or not raw.is_file():
        download(UPSTREAM_URL, raw)
    raw_sha = sha256_file(raw)
    if raw_sha != UPSTREAM_SHA256:
        raise ValueError(f"Pinned OASST1 archive SHA-256 mismatch: {raw_sha}")

    rows = []
    with gzip.open(raw, "rt", encoding="utf8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    selected, reject, benchmark_files = build_rows(
        rows, Path(__file__).resolve().parent, args.max_assistant_turns, args.max_seq
    )
    if not selected:
        raise ValueError("No usable multi-turn OASST1 conversations")

    counts = {}
    for split, _ in SPLIT_BUCKETS:
        part = [r for r in selected if r["split"] == split]
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf8")
        counts[split] = {
            "rows": len(part),
            "assistant_turns": sum(r["assistant_turns"] for r in part),
            "de": sum(r["language"] == "de" for r in part),
            "en": sum(r["language"] == "en" for r in part),
            "sha256": sha256_file(path),
        }

    manifest = {
        "version": 1,
        "dataset_id": DATASET_ID,
        "role": "human-multiturn-sft",
        "license": policy.license,
        "license_url": LICENSE_URL,
        "provenance_url": DATASET_URL,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_download_url": UPSTREAM_URL,
        "upstream_sha256": raw_sha,
        "filters": "positively reviewed, non-deleted, non-synthetic, no model_name; same-language de/en; alternating roles",
        "selection": "one deterministic richest/best-ranked all-human chain per root tree",
        "tree_level_split": {name: width for name, width in SPLIT_BUCKETS},
        "max_assistant_turns": args.max_assistant_turns,
        "max_seq": args.max_seq,
        "benchmark_files": benchmark_files,
        "benchmark_literal_screen": True,
        "external_model_calls": False,
        "raw_user_ids_exported": False,
        "counts": counts,
        "rejected": dict(reject),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"accepted": len(selected), "counts": counts, "rejected": dict(reject)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
