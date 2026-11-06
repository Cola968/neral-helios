#!/usr/bin/env python3
"""Read-only audit of historical NeralTalk; export a local, NOT APPROVED QLoRA candidate.

No network, torch, model loading, training, or data uploads. The output is a
fresh local directory; train/val/test are verified but only train is sampled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

SPLITS = ("train", "val", "test")
ROLE_SEQUENCE = ("user", "assistant")
SYSTEM = (
    "Du bist NERAL auf offengelegter Qwen-Basis. Antworte präzise und ehrlich. "
    "Erfinde keine Fakten, Quellen oder ausgeführten Werkzeuge."
)
QUOTAS = {"code": 120, "math": 100, "reasoning": 60,
          "chat": 50, "explanation": 40, "science": 20,
          "format": 20, "summary": 20}
SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"<\s*/?\s*think\s*>)"
)


def _canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_approved(row: dict) -> bool:
    source, license_id = row.get("source"), row.get("license")
    if not isinstance(source, str) or not isinstance(license_id, str):
        return False
    if source.startswith("neral-generated-"):
        return license_id == "Project-generated deterministic data; no external corpus"
    if source.startswith("oasst2-human-"):
        return license_id == "Apache-2.0"
    return False


def _row_messages(row: dict) -> list[dict]:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    elif isinstance(row.get("prompt"), str) and isinstance(row.get("answer"), str):
        messages = [{"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["answer"]}]
    else:
        raise ValueError("missing chat messages")
    if len(messages) < 2 or len(messages) > 16 or len(messages) % 2:
        raise ValueError("unsupported number of chat messages")
    for i, message in enumerate(messages):
        if (not isinstance(message, dict) or
                message.get("role") != ROLE_SEQUENCE[i % 2] or
                not isinstance(message.get("content"), str) or
                not message["content"].strip()):
            raise ValueError("invalid conversational roles or content")
    return messages


def _read_split(root: Path, name: str) -> tuple[list[dict], str]:
    path = root / (name + ".jsonl")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked split: {name}")
    if path.stat().st_size > 40 * 1024 * 1024:
        raise ValueError(f"Oversized split: {name}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank {name} line {line_number}")
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError(f"Malformed JSON in {name} line {line_number}") from exc
            if (not isinstance(row, dict) or
                    not all(isinstance(row.get(k), str) and row[k].strip()
                            for k in ("id", "language", "domain", "source", "license")) or
                    not _source_approved(row)):
                raise ValueError(f"Missing or unapproved source metadata in {name} line {line_number}")
            _row_messages(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset split: {name}")
    return rows, _file_sha(path)


def scan_legacy_dataset(root: Path) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source root missing or symlinked")
    data = {}
    digests = {}
    ids = set()
    for split in SPLITS:
        rows, sha = _read_split(root, split)
        data[split] = rows
        digests[split] = sha
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate row ID across splits")
            ids.add(row["id"])
    # Only first-user prompt hashes from val/test: no target answers are used.
    holdout_prompts = set()
    for split in ("val", "test"):
        for row in data[split]:
            holdout_prompts.add(_digest(_canonical(_row_messages(row)[0]["content"])))
    return {"train": data["train"], "split_counts": {s: len(data[s]) for s in SPLITS},
            "split_sha256": digests, "holdout_prompts": holdout_prompts}


def choose_candidates(scan: dict, *, quotas: dict[str, int] | None = None,
                      additional_blocked: set[str] | None = None) -> tuple[list[dict], dict]:
    """Deterministic sampling from TRAIN ONLY; no semantic-review claims."""
    quotas = dict(QUOTAS if quotas is None else quotas)
    if any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ValueError("Invalid domain quota")
    blocked = scan["holdout_prompts"] | (additional_blocked or set())
    by_domain = {name: [] for name in quotas}
    stats = Counter()
    seen_train_prompts = set()
    ordered = sorted(scan["train"], key=lambda r: (_digest(r["id"]), r["id"]))
    for row in ordered:
        stats["train_rows_seen"] += 1
        domain = row["domain"].lower()
        if domain not in quotas:
            stats["excluded_domain"] += 1
            continue
        messages = _row_messages(row)
        if len(messages) != 2:
            stats["excluded_multiturn"] += 1
            continue
        prompt, answer = (m["content"].strip() for m in messages)
        ph = _digest(_canonical(prompt))
        if ph in blocked:
            stats["excluded_heldout_prompt"] += 1
            continue
        if ph in seen_train_prompts:
            stats["excluded_repeated_train_prompt"] += 1
            continue
        seen_train_prompts.add(ph)
        if (len(prompt) > 850 or len(answer) > 1000 or
                len(prompt) < 8 or len(answer) < 8 or
                SECRET.search(prompt) or SECRET.search(answer) or
                _canonical(prompt) == _canonical(answer) or
                "\x00" in prompt or "\x00" in answer):
            stats["excluded_sensitive_or_invalid"] += 1
            continue
        by_domain[domain].append((row["id"], prompt, answer))
    selected = []
    selected_by_domain = {}
    for domain, quota in quotas.items():
        chosen = sorted(by_domain[domain], key=lambda r: _digest(domain + ":" + r[0]))[:quota]
        if chosen:
            selected_by_domain[domain] = len(chosen)
        for _rid, prompt, answer in chosen:
            selected.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]})
    result = {"train_rows_seen": len(scan["train"]),
              "selected_rows": len(selected), "selected_by_domain": selected_by_domain,
              "excluded_heldout_prompt": stats["excluded_heldout_prompt"],
              "excluded_repeated_train_prompt": stats["excluded_repeated_train_prompt"],
              "excluded_multiturn": stats["excluded_multiturn"],
              "excluded_sensitive_or_invalid": stats["excluded_sensitive_or_invalid"],
              "excluded_domain": stats["excluded_domain"],
              "review_required": True, "training_performed": False}
    return selected, result


def validate_export(path: Path) -> int:
    """Validates the exact schema used by current QLoRA read_examples."""
    size = path.stat().st_size
    if not 0 < size <= 1024 * 1024:
        raise ValueError("Candidate must fit the reviewed QLoRA 1MiB data cap")
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {"messages"}:
                raise ValueError("Bad QLoRA JSONL envelope")
            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError("Expected three chat messages")
            for name, msg in zip(("system", "user", "assistant"), messages):
                if (not isinstance(msg, dict) or set(msg) != {"role", "content"} or
                        msg["role"] != name or not isinstance(msg["content"], str) or
                        not msg["content"].strip()):
                    raise ValueError("Invalid model-training role")
            count += 1
    if not 1 <= count <= 2000:
        raise ValueError("Invalid QLoRA pilot batch length")
    return count


def publish_candidate(destination: Path, selected: list[dict], stats: dict,
                      *, source_root: Path) -> dict:
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite candidate dataset")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("An existing trusted output parent is required")
    src = Path(source_root).resolve()
    dest = destination.resolve()
    if dest == src or src in dest.parents:
        raise ValueError("Do not put output in source data directory")
    if not selected:
        raise ValueError("No reviewed-looking candidate rows found")
    # O_EXCL semantics for destination root; caller selects a unique run ID.
    destination.mkdir(exist_ok=False)
    path = destination / "candidate.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    total = validate_export(path)
    report = {"schema": 1, "status": "LOCAL_CANDIDATE_REVIEW_REQUIRED",
              "approved_for_training": False, "training_performed": False,
              "holdout_examples_exported": 0, "model_weights_modified": False,
              "rows": total, "sha256": _file_sha(path), "stats": stats}
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    scan = scan_legacy_dataset(args.source_root)
    blocked = set()
    try:
        from scripts.neral_r32_compare import TASKS as DEV_CODING_TASKS
        blocked.update(_digest(_canonical(t[1])) for t in DEV_CODING_TASKS)
    except ImportError:
        raise RuntimeError("Current NERAL coding smoke suite must be present before export")
    selected, stats = choose_candidates(scan, additional_blocked=blocked)
    if len(selected) < 100:
        raise ValueError("Fewer than 100 candidate rows; review source data")
    report = publish_candidate(args.output_dir, selected, stats, source_root=args.source_root)
    report["source_split_sha256"] = scan["split_sha256"]
    report["source_split_rows"] = scan["split_counts"]
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # No user prompts or assistant responses in stdout.
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "approved_for_training": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
