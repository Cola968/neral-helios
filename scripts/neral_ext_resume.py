"""Crash-resumable, local-only NERAL-EXT responses; never a public artifact.

The progress journal is an append-only JSONL containing private model answers.
It is tied to the case text, prompt protocol, variant, model hashes and decode
configuration via an explicit 64-hex-character caller-provided run key.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

VARIANTS = frozenset({"base", "base_fewshot", "neral_v5"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
MAX_CASES = 500
MAX_PROGRESS_BYTES = 2_500_000


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate checkpoint JSON key")
        result[key] = value
    return result


def _parse_row(raw: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Malformed NERAL-EXT checkpoint JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Checkpoint JSON must be an object")
    return value


def _normalize_record(case: dict, record: object) -> dict:
    if (not isinstance(record, dict) or set(record) !=
            {"text", "complete", "new_tokens"} or
            not isinstance(record["text"], str) or
            type(record["complete"]) is not bool or
            type(record["new_tokens"]) is not int or
            not 0 <= record["new_tokens"] <= 512):
        raise ValueError("Untrusted local generator result")
    raw = record["text"]
    complete = bool(record["complete"] and 0 < len(raw) <= 4096)
    return {
        "id": case["id"], "text": raw if complete else "",
        "complete": complete, "new_tokens": record["new_tokens"],
    }


def _read_checkpoint(progress: Path, identity: dict, cases: list[dict]) -> list[dict]:
    if progress.is_symlink() or not progress.is_file():
        raise ValueError("Invalid or linked checkpoint")
    try:
        if not 0 < progress.stat().st_size <= MAX_PROGRESS_BYTES:
            raise ValueError("Checkpoint size invalid")
        blob = progress.read_bytes()
        if not blob.endswith(b"\n"):
            raise ValueError("Truncated checkpoint; manual review required")
        raw_lines = blob.decode("utf-8").splitlines()
        if not 1 <= len(raw_lines) <= len(cases) + 1 or any(not s for s in raw_lines):
            raise ValueError("Unexpected checkpoint length")
        meta = _parse_row(raw_lines[0])
        if meta != identity:
            raise ValueError("Checkpoint model, prompt or dataset fingerprint changed")
        rows = []
        for index, raw in enumerate(raw_lines[1:]):
            row = _parse_row(raw)
            if (set(row) != {"id", "text", "complete", "new_tokens"}
                    or row["id"] != cases[index]["id"]
                    or not isinstance(row["text"], str)
                    or type(row["complete"]) is not bool
                    or type(row["new_tokens"]) is not int
                    or not 0 <= row["new_tokens"] <= 512
                    or (row["complete"] and not 0 < len(row["text"]) <= 4096)
                    or (not row["complete"] and row["text"] != "")):
                raise ValueError("Corrupted or reordered NERAL-EXT checkpoint")
            rows.append(row)
        return rows
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError("Checkpoint cannot be read safely") from exc


def _append_sync(progress: Path, row: dict, *, create: bool = False) -> None:
    mode = "x" if create else "a"
    with progress.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(_encode(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_create_only(dest: Path, rows: list[dict]) -> None:
    """Never overwrite the destination, even if another process created it."""
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=dest.parent, prefix=".neral-ext-", suffix=".staged",
                delete=False) as stream:
            staged = Path(stream.name)
            for row in rows:
                stream.write(_encode({
                    "id": row["id"], "text": row["text"],
                    "complete": row["complete"],
                }) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link in the same directory publishes the complete result
        # atomically and fails rather than replacing an existing destination.
        os.link(staged, dest)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def write_variant_checkpointed(
    cases: list[dict],
    variant: str,
    dest: Path,
    generate: Callable[[dict, list[dict]], dict],
    build_messages: Callable[[dict, str], list[dict]],
    *,
    run_key: str,
) -> dict:
    """Resume the *identical* pinned run after interruption; no partial final."""
    dest = Path(dest)
    if (variant not in VARIANTS or not isinstance(run_key, str)
            or SHA256.fullmatch(run_key) is None or
            dest.is_symlink() or dest.exists() or
            not dest.parent.is_dir() or dest.parent.is_symlink()):
        raise ValueError("Unapproved NERAL-EXT output or run fingerprint")
    if (not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES
            or any(not isinstance(c, dict) or
                   not isinstance(c.get("id"), str) or
                   CASE_ID.fullmatch(c["id"]) is None or
                   not isinstance(c.get("prompt"), str) for c in cases)
            or len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("Invalid holdout case IDs or prompts")

    # Hash complete case records and actual rendered messages; the run key
    # additionally pins verified model files and generation configuration.
    protocol = [
        {"case": case, "messages": build_messages(case, variant)}
        for case in cases
    ]
    fingerprint = hashlib.sha256(_encode({
        "run_key": run_key, "variant": variant, "protocol": protocol,
        "schema": 1,
    }).encode("utf-8")).hexdigest()
    identity = {"schema": 1, "variant": variant, "fingerprint": fingerprint}
    progress = dest.with_name(dest.name + ".progress")
    if progress.is_symlink():
        raise ValueError("Linked checkpoint forbidden")
    if progress.exists():
        rows = _read_checkpoint(progress, identity, cases)
    else:
        # Append-only journal lives alongside the private destination.
        _append_sync(progress, identity, create=True)
        rows = []
    resumed = len(rows)
    for case in cases[resumed:]:
        record = generate(case, build_messages(case, variant))
        row = _normalize_record(case, record)
        _append_sync(progress, row)
        rows.append(row)

    # On a failure above, the journal is deliberately retained for recovery.
    _publish_create_only(dest, rows)
    progress.unlink()  # Only after successful atomic creation of final output.
    return {
        "cases": len(rows),
        "generated_tokens": sum(row["new_tokens"] for row in rows),
        "incomplete": sum(not row["complete"] for row in rows),
        "resumed_cases": resumed,
        "status": "LOCAL_RESPONSES_SAVED_NOT_VALIDATED",
    }
