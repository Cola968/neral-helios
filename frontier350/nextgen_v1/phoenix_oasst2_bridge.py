"""Offline OASST2 -> Phoenix pilot source adapter; never downloads, trains, or approves.

Requires the exact previously pinned *local* OpenAssistant-2 ready-message archive
and a separately frozen, private holdout-hash registry. Entire message trees are
assigned to one split before producing the existing four-field Phoenix JSONL.
Raw text, IDs, paths, holdout hashes, and checkpoint contents never go to logs.
This adapter cannot establish license, privacy, semantic contamination, or quality.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fetch_oasst2_sft50 import (DATASET_ID, UPSTREAM_REVISION, UPSTREAM_SHA256,
                                choose_pairs, reviewed_human, sha256_file)
from sft_contract import FORBIDDEN, hex_digest, prompt_hash, sha
from phoenix_gate import unique_object

MAX_COMPRESSED_BYTES = 250_000_000
MAX_RAW_MESSAGES = 200_000
MAX_LINE_CHARS = 120_000
SOURCE_ID = "oasst2-pinned-human"


def local_input(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("input must be an existing regular file inside private root; no symlinks")
    return path.resolve(strict=True)


def tree_split(tree_id: str) -> str:
    """Stable split for all DE/EN messages belonging to one conversation tree."""
    if not isinstance(tree_id, str) or not tree_id.strip():
        raise ValueError("missing conversation tree identity; cannot prevent split leakage")
    digest = hashlib.sha256(("PHOENIX:OASST2:TREE:v1:" + tree_id).encode("utf-8")).hexdigest()
    return "validation" if int(digest[:8], 16) % 10 == 0 else "train"


def load_archive(path: Path) -> list[dict]:
    if path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise ValueError("archive exceeds bounded local audit size")
    if sha256_file(path) != UPSTREAM_SHA256:
        raise ValueError("OASST2 archive differs from project-pinned upstream SHA-256")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if number > MAX_RAW_MESSAGES or len(line) > MAX_LINE_CHARS:
                raise ValueError("OASST2 decompression bounds exceeded")
            rows.append(json.loads(line, object_pairs_hook=unique_object))
    if not rows:
        raise ValueError("empty OASST2 archive")
    return rows


def select_tree_disjoint(pairs: dict, messages: list[dict], frozen: set[str]) -> tuple[dict, dict]:
    """Deterministically convert prefiltered human pairs; fail on missing tree lineage."""
    by_id = {}
    for message in messages:
        ident = message.get("message_id") if isinstance(message, dict) else None
        if isinstance(ident, str):
            if ident in by_id:
                raise ValueError("duplicate upstream message ID")
            by_id[ident] = message
    splits: dict[str, list[dict]] = {"train": [], "validation": []}
    groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    seen_prompts = set()
    skipped_duplicate_prompt = skipped_incompatible_text = 0
    for language in ("de", "en"):
        for pair in pairs.get(language, ()):
            if not isinstance(pair, dict) or not isinstance(pair.get("id"), str) or not pair["id"].startswith("oasst2-"):
                raise ValueError("invalid pinned OASST2 pair identity")
            assistant = by_id.get(pair["id"][len("oasst2-"):])
            parent = by_id.get(pair.get("upstream_parent_id"))
            if not assistant or not parent or assistant.get("role") != "assistant" or parent.get("role") != "prompter":
                raise ValueError("unverifiable OASST2 prompt/answer lineage")
            if assistant.get("parent_id") != parent.get("message_id"):
                raise ValueError("OASST2 parent relation inconsistent")
            tree = assistant.get("message_tree_id")
            if not isinstance(tree, str) or not tree.strip() or tree != parent.get("message_tree_id"):
                raise ValueError("missing or inconsistent OASST2 conversation tree ID")
            prompt, answer = pair.get("prompt"), pair.get("answer")
            if not isinstance(prompt, str) or not isinstance(answer, str):
                raise ValueError("OASST2 pair has invalid text types")
            # Never trust the converted pair independently of its pinned raw messages.
            # A byte-consistent archive can still be mismatched by a buggy converter.
            if (parent.get("lang") != language or assistant.get("lang") != language
                    or not reviewed_human(parent) or not reviewed_human(assistant)
                    or not isinstance(parent.get("text"), str)
                    or not isinstance(assistant.get("text"), str)
                    or prompt.strip() != parent["text"].strip()
                    or answer.strip() != assistant["text"].strip()):
                raise ValueError("OASST2 converted pair does not match reviewed human source")
            if not prompt.strip() or not answer.strip() or max(len(prompt), len(answer)) > 8192 or any(
                    marker in text for text in (prompt, answer) for marker in FORBIDDEN):
                skipped_incompatible_text += 1
                continue
            fingerprint = prompt_hash(prompt)
            if fingerprint in frozen:
                raise ValueError("exact frozen benchmark contamination: no Phoenix source export")
            if fingerprint in seen_prompts:
                skipped_duplicate_prompt += 1
                continue
            seen_prompts.add(fingerprint)
            split = tree_split(tree)
            groups[split].add(tree)
            splits[split].append({"id": pair["id"], "source_id": SOURCE_ID,
                                  "user": prompt.strip(), "assistant": answer.strip()})
    if groups["train"] & groups["validation"]:
        raise AssertionError("conversation tree appeared in both splits")
    if min(len(splits["train"]), len(splits["validation"])) < 8:
        raise ValueError("insufficient train/validation rows; no partial export")
    summary = {"rows": {name: len(rows) for name, rows in splits.items()},
               "conversation_trees": {name: len(groups[name]) for name in splits},
               "duplicate_prompts_skipped": skipped_duplicate_prompt,
               "incompatible_text_skipped": skipped_incompatible_text,
               "split_unit": "message_tree_id", "split_disjoint": True}
    return splits, summary


def prepare(root: Path, archive: Path, holdout: Path, out: Path,
            max_en: int = 800, max_de: int = 400) -> dict:
    root = root.resolve(strict=True)
    archive = local_input(archive, root)
    holdout = local_input(holdout, root)
    if not 50 <= max_en <= 2000 or not 50 <= max_de <= 2000:
        raise ValueError("bounded DE/EN sample caps must be 50..2000")
    if out.exists() or out.is_symlink() or not out.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("output must be a new directory inside private asset root")
    registry = json.loads(holdout.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(registry, list) or not registry:
        raise ValueError("nonempty frozen benchmark hash registry required")
    frozen = {hex_digest(value, "frozen prompt") for value in registry}
    if len(frozen) != len(registry):
        raise ValueError("duplicate frozen prompt hashes")
    messages = load_archive(archive)
    pairs = choose_pairs(messages, max_en=max_en, max_de=max_de)
    splits, summary = select_tree_disjoint(pairs, messages, frozen)
    # Create the directory only when all source, tree and frozen-holdout checks pass.
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "classification": "local_source_export_not_training_approval",
              "approved": False, "human_review_required": True,
              "dataset_id": DATASET_ID, "upstream_revision": UPSTREAM_REVISION,
              "upstream_archive_sha256": UPSTREAM_SHA256,
              "frozen_registry_sha256": sha(holdout), "source_id": SOURCE_ID,
              **summary,
              "note": "Only SHA, human-row/lineage/tree-disjointness and exact-hash checks; independent license, PII, near/semantic duplicate and source-row review still required. No training and no approval."}
    try:
        for name, rows in splits.items():
            path = out / f"{name}.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            report[f"{name}_sha256"] = sha(path)
        (out / "source_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        (out / "INVALID_EXPORT").write_text("Incomplete local source export: do not train.\n", encoding="utf-8")
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--holdout-hashes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-en", type=int, default=800)
    parser.add_argument("--max-de", type=int, default=400)
    args = parser.parse_args()
    result = prepare(args.assets_root, args.archive, args.holdout_hashes,
                     args.out, args.max_en, args.max_de)
    # Only counts, hashes and provenance: NEVER print model answers or source text.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
