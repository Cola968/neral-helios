"""Stage, but NEVER approve or train on, local OASST1 dialogue candidates.

Input is the official pinned ready.messages.jsonl.gz. Only existing local files
are read; this program does not download, deserialize Torch, or touch weights.
All outputs are REVIEW REQUIRED, not an approved SFT pack. No text is logged.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SOURCE_NAME = "2023-04-12_oasst_ready.messages.jsonl.gz"
SOURCE_URL = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{REVISION}/{SOURCE_NAME}"
BAD_MARKERS = ("<|bos|>", "<|eos|>", "<|pad|>", "\nUser:", "\nAssistant:")
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[a-zA-Z]{2,}\b")
PHONE_LIKE = re.compile(r"(?:\+\d[\d\s().-]{8,}\d|\b\d{8,}\b)")
BAD_LABELS = ("spam", "pii", "hate_speech", "sexual_content", "lang_mismatch", "not_appropriate", "fails_task")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def frozen_chat_hashes(benchmark: Path) -> set[str]:
    """Read the frozen NIS2 chat-core literals ONLY for leakage checks, not training."""
    syntax = ast.parse(benchmark.read_text(encoding="utf-8"))
    literals = {}
    for node in syntax.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PAIR_CASES", "EXACT_GENERATION_CASES"}:
                    literals[target.id] = ast.literal_eval(node.value)
    if not literals.get("PAIR_CASES") or not literals.get("EXACT_GENERATION_CASES"):
        raise ValueError("frozen NIS2 chat-core extraction failed closed")
    hashes = set()
    for category, prefix, good, bad in literals["PAIR_CASES"]:
        if "User: " not in prefix or not prefix.endswith("\nAssistant:"):
            raise ValueError("unexpected NIS2 dialogue shape")
        latest = prefix.rsplit("User: ", 1)[1].split("\nAssistant:", 1)[0]
        hashes.add(prompt_hash(latest))
        hashes.add(prompt_hash(prefix))
    for case_id, prompt, answers in literals["EXACT_GENERATION_CASES"]:
        hashes.add(prompt_hash(prompt))
    return hashes


def permitted(message: dict, role: str) -> bool:
    if (message.get("role") != role or message.get("lang") not in {"en", "de"}
            or message.get("review_result") is not True or message.get("deleted") is not False
            or message.get("synthetic") is not False or message.get("tree_state") != "ready_for_export"
            or type(message.get("review_count")) is not int or message["review_count"] < 1):
        return False
    text = message.get("text")
    maximum = 260 if role == "prompter" else 520
    if (not isinstance(text, str) or not 5 <= len(text.strip()) <= maximum
            or any(marker in text for marker in BAD_MARKERS)
            or EMAIL.search(text) or PHONE_LIKE.search(text)):
        return False
    labels = message.get("labels", {})
    if isinstance(labels, dict):
        for name in BAD_LABELS:
            entry = labels.get(name)
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) and entry["value"] > 0.2:
                return False
    return True


def select(source: Path, frozen: set[str], *, train_limit: int = 1200, valid_limit: int = 160,
           min_de_train: int = 20, min_de_valid: int = 5) -> tuple[dict, dict]:
    if source.name != SOURCE_NAME or not frozen:
        raise ValueError("pinned official source filename and nonempty frozen registry required")
    roots: dict[str, dict] = {}
    replies = []
    messages = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            messages += 1
            if messages > 100_000:
                raise ValueError("unexpected OASST export length")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("invalid OASST record")
            if record.get("parent_id") is None and permitted(record, "prompter"):
                roots[record["message_id"]] = record
            elif record.get("parent_id") is not None and record.get("rank") == 0 and permitted(record, "assistant"):
                replies.append(record)
    selections = {"train": {"de": [], "en": []}, "validation": {"de": [], "en": []}}
    seen = set()
    for reply in sorted(replies, key=lambda row: row["message_id"]):
        parent = roots.get(reply["parent_id"])
        if not parent or reply["lang"] != parent["lang"] or reply.get("message_tree_id") != parent.get("message_tree_id"):
            continue
        prompt = parent["text"].strip()
        digest = prompt_hash(prompt)
        if digest in frozen or digest in seen:
            continue
        seen.add(digest)
        tree = parent["message_tree_id"]
        split = "validation" if int(hashlib.sha256(tree.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        selections[split][reply["lang"]].append({
            "id": reply["message_id"], "source_id": "oasst1-ready-2023-04-12",
            "user": prompt, "assistant": reply["text"].strip(),
        })
    result = {}
    for split, cap in (("train", train_limit), ("validation", valid_limit)):
        # Reserve up to a quarter for German, rather than allowing English to crowd it out.
        de = selections[split]["de"][: cap // 4]
        en = selections[split]["en"][: cap - len(de)]
        if len(de) < (min_de_train if split == "train" else min_de_valid) or not en:
            raise ValueError(f"not enough reviewed German/English {split} candidates")
        result[split] = sorted([*de, *en], key=lambda r: r["id"])
        if not result[split]:
            raise ValueError("empty split")
    counts = {split: {lang: sum(row["user"] != "" and ("de" if row["id"] in {r["id"] for r in selections[split]["de"]} else "en") == lang for row in result[split]) for lang in ("de", "en")} for split in result}
    if set(row["id"] for row in result["train"]) & set(row["id"] for row in result["validation"]):
        raise ValueError("training/validation identity leakage")
    return result, {"source_messages": messages, "candidate_counts": counts, "frozen_chat_hashes_checked": len(frozen)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--frozen-benchmark", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refuse to reuse any staging directory")
    digest = sha_file(args.source)
    frozen = frozen_chat_hashes(args.frozen_benchmark)
    selected, statistics = select(args.source, frozen)
    args.out.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        with (args.out / f"candidate_{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": 1, "status": "REVIEW_REQUIRED_NOT_APPROVED_NOT_TRAINED",
        "source_id": "OpenAssistant/oasst1", "source_revision": REVISION,
        "source_uri": SOURCE_URL, "source_filename": args.source.name,
        "source_sha256": digest, "license_declared_by_publisher": "Apache-2.0",
        "benchmark_screening": "exact NIS2 chat-core only; not full NIS2 or semantic overlap",
        "screened_benchmark_file_sha256": sha_file(args.frozen_benchmark),
        "independent_privacy_and_license_review": False,
        "approved_for_training": False,
        **statistics,
        "output_sha256": {split: sha_file(args.out / f"candidate_{split}.jsonl") for split in selected},
        "warning": "Automatic PII/near-duplicate filtering is incomplete. Do not fabricate a signed screen.json or train until an independent review and holdout registry are supplied.",
    }
    (args.out / "STAGING_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"source_uri"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
