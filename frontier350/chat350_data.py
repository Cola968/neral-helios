from __future__ import annotations
import argparse
import collections
import hashlib
import json
import unicodedata
from pathlib import Path

REQUIRED_META = ("id", "language", "domain", "source", "license")
SPLITS = ("train", "val", "test")
ROLES = {"user", "assistant"}

def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()

def normalize_row(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    for key in REQUIRED_META:
        if not _norm(row.get(key, "")):
            raise ValueError(f"missing/empty field: {key}")
    if "messages" in row:
        raw = row["messages"]
        if not isinstance(raw, list) or len(raw) < 2 or len(raw) > 32:
            raise ValueError("messages must contain 2..32 turns")
        messages = []
        expected = "user"
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("message must be an object")
            role = item.get("role")
            content = _norm(item.get("content", ""))
            if role not in ROLES or role != expected:
                raise ValueError(f"expected {expected} turn")
            if not content:
                raise ValueError("empty message content")
            if len(content) > 16_384:
                raise ValueError("single turn too long")
            messages.append({"role": role, "content": content})
            expected = "assistant" if role == "user" else "user"
        if messages[-1]["role"] != "assistant":
            raise ValueError("conversation must end with assistant")
    else:
        prompt = _norm(row.get("prompt", ""))
        answer = _norm(row.get("answer", ""))
        if not prompt or not answer:
            raise ValueError("row needs messages or non-empty prompt/answer")
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    total_chars = sum(len(x["content"]) for x in messages)
    if total_chars > 65_536:
        raise ValueError("conversation too long")
    return {
        "id": _norm(row["id"]),
        "language": _norm(row["language"]).lower(),
        "domain": _norm(row["domain"]).lower(),
        "source": _norm(row["source"]),
        "license": _norm(row["license"]),
        "messages": messages,
    }

def conversation_sha256(row: dict) -> str:
    payload = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()

def validate_prompt_holdouts(split_rows: dict[str, list[dict]]) -> dict:
    """Fail closed on ANY normalized user prompt reused across data splits.

    The older first-turn-only check missed exact user messages in later chat
    rounds. This is strict lexical contamination protection, not a semantic
    deduplicator: changing numbers or paraphrasing can still escape this gate.
    No private prompt text appears in error output.
    """
    seen: dict[str, str] = {}
    overlaps: set[tuple[str, str, str]] = set()
    for split in SPLITS:
        if split not in split_rows:
            raise ValueError("missing holdout split: " + split)
        for row in split_rows[split]:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("invalid conversation in prompt holdout check")
            for turn in messages:
                if not isinstance(turn, dict):
                    raise ValueError("invalid turn in prompt holdout check")
                if turn.get("role") != "user":
                    continue
                first = turn.get("content")
                if not isinstance(first, str) or not first.strip():
                    raise ValueError("empty or invalid user turn in holdout check")
                canonical = " ".join(unicodedata.normalize("NFKC", first).casefold().split())
                digest = hashlib.sha256(canonical.encode("utf8")).hexdigest()
                previous = seen.setdefault(digest, split)
                if previous != split:
                    overlaps.add((digest, previous, split))
    if overlaps:
        sample = sorted(overlaps)[0]
        raise ValueError(
            f"all-user-turn prompt leakage across train/val/test: "
            f"{len(overlaps)} cross-split overlap keys (first sha256={sample[0]}, "
            f"{sample[1]} vs {sample[2]})"
        )
    return {
        "first_user_prompt_overlap": 0,  # backward compatible success field
        "all_user_turn_overlap": 0,
        "normalization": "NFKC-casefold-whitespace",
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(normalize_row(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows

def validate_dataset_root(root: Path) -> tuple[dict, dict[str, list[dict]]]:
    root = Path(root)
    split_rows = {}
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    stats = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        split_rows[split] = rows
        languages = collections.Counter()
        domains = collections.Counter()
        sources = collections.Counter()
        turns = collections.Counter()
        for row in rows:
            rid = row["id"]
            if rid in seen_ids:
                raise ValueError(f"duplicate id across splits: {rid} ({seen_ids[rid]} vs {split})")
            seen_ids[rid] = split
            digest = conversation_sha256(row)
            if digest in seen_content:
                raise ValueError(f"duplicate conversation across splits: {seen_content[digest]} vs {split}")
            seen_content[digest] = split
            languages[row["language"]] += 1
            domains[row["domain"]] += 1
            sources[row["source"]] += 1
            turns[len(row["messages"])] += 1
        stats[split] = {
            "rows": len(rows),
            "sha256": file_sha256(path),
            "languages": dict(sorted(languages.items())),
            "domains": dict(sorted(domains.items())),
            "sources": dict(sorted(sources.items())),
            "turn_counts": {str(k): v for k, v in sorted(turns.items())},
        }
    manifest = {
        "schema": 1,
        "format": "neraltalk350-jsonl-v1",
        "splits": stats,
        "total_rows": sum(x["rows"] for x in stats.values()),
        "global_unique_ids": len(seen_ids),
        "global_unique_conversations": len(seen_content),
        "assistant_only_training": True,
        "benchmark_policy": "train/val/test are development data only; frozen NIS2/public benchmark holdouts remain separate and forbidden from training",
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest, split_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest, _ = validate_dataset_root(Path(args.root))
    Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    main()
