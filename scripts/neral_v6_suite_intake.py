"""Audit private v6 600/600 development and release suites without disclosing rows.

This tool is ONLY an input preflight. It does not create original test cases,
prove independent provenance, certify unit tests or approve a release.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_decontamination import normalize, shingles

DOMAINS = ("math", "code", "reasoning")
KINDS = {"math": "numeric", "code": "python", "reasoning": "boolean"}
FIELDS = {"id", "domain", "family", "source_id", "kind", "prompt", "answer", "tests_sha256"}
IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
LIMIT_BYTES = 8_000_000
MAX_TRAINING_BYTES = 20_000_000


def _no_dupes(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("Duplicate JSON property")
        record[key] = value
    return record


def _read_private_jsonl(path: Path, limit_bytes: int) -> tuple[bytes, list[dict]]:
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= limit_bytes):
        raise ValueError("Missing/linked/oversized private data source")
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
        if not lines or len(lines) > 100000 or any(not x.strip() for x in lines):
            raise ValueError("Empty/invalid JSONL line")
        rows = [json.loads(x, object_pairs_hook=_no_dupes) for x in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid private JSONL") from exc
    if not all(isinstance(x, dict) for x in rows):
        raise ValueError("Expected JSON objects")
    return raw, rows


def _user_message(messages) -> str:
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    users = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"]
    if (len(users) != 1 or not isinstance(users[0], str)
            or not users[0].strip() or len(users[0]) > 10000):
        raise ValueError("Expected one nonempty training user turn")
    return users[0]


def read_training_prompts(rows: list[dict]) -> list[str]:
    if not rows:
        raise ValueError("Training inventory must not be empty")
    prompts = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Training row is not an object")
        if "messages" in row and "prompt" not in row:
            prompt = _user_message(row["messages"])
        elif "prompt" in row and "messages" not in row:
            value = row["prompt"]
            prompt = _user_message(value) if isinstance(value, list) else value
        else:
            raise ValueError("Ambiguous training prompt shape")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
            raise ValueError("Training prompt invalid")
        prompts.append(prompt)
    return prompts


def audit_suite(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or len(rows) != 600:
        raise ValueError("Suite must contain exactly 600 private cases")
    seen_ids, seen_sources, seen_prompts = set(), set(), set()
    by_domain = Counter()
    methods = {domain: set() for domain in DOMAINS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Invalid suite row shape")
        domain, kind = row["domain"], row["kind"]
        if domain not in DOMAINS or kind != KINDS[domain]:
            raise ValueError("Unexpected domain or grading kind")
        for col, seen in (("id", seen_ids), ("source_id", seen_sources)):
            v = row[col]
            if not isinstance(v, str) or not IDENTITY.fullmatch(v) or v in seen:
                raise ValueError("Invalid or duplicated case/source identifier")
            seen.add(v)
        family = row["family"]
        if not isinstance(family, str) or not IDENTITY.fullmatch(family):
            raise ValueError("Missing task method family")
        prompt, answer = row["prompt"], row["answer"]
        if (not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000 or
                not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 5000):
            raise ValueError("Unreviewed or oversized task/answer")
        normalized = normalize(prompt)
        if not normalized or normalized in seen_prompts:
            raise ValueError("Repeated normalized prompt")
        seen_prompts.add(normalized)
        tests_hash = row["tests_sha256"]
        if kind == "python":
            if not isinstance(tests_hash, str) or not HEX.fullmatch(tests_hash):
                raise ValueError("Python benchmark requires hidden unit-test SHA-256")
        elif tests_hash is not None:
            raise ValueError("Unexpected executable test reference")
        by_domain[domain] += 1
        methods[domain].add(family)
    if any(by_domain[d] != 200 or len(methods[d]) < 5 for d in DOMAINS):
        raise ValueError("Need 200 cases and >=5 distinct methods per domain")
    return {domain: {"count": by_domain[domain], "methods": len(methods[domain])}
            for domain in DOMAINS}


def _overlap(a: list[str], b: list[str]) -> bool:
    """Exact normalized matches, plus normalized 10-gram intersections."""
    normalized_a = {normalize(s) for s in a}
    grams_a = set().union(*(shingles(s, n=10) for s in a))
    for item in b:
        if normalize(item) in normalized_a or shingles(item, n=10) & grams_a:
            return True
    return False


def audit_pair(dev: list[dict], release: list[dict], training_prompts: list[str]) -> dict:
    if not training_prompts:
        raise ValueError("Missing training input inventory")
    audited = {"dev": audit_suite(dev), "release": audit_suite(release)}
    if {x["id"] for x in dev} & {x["id"] for x in release}:
        raise ValueError("ID leak across development/release")
    if {x["source_id"] for x in dev} & {x["source_id"] for x in release}:
        raise ValueError("Original task reused across development/release")
    dev_prompts = [x["prompt"] for x in dev]
    release_prompts = [x["prompt"] for x in release]
    if (_overlap(dev_prompts, release_prompts)
            or _overlap(dev_prompts, training_prompts)
            or _overlap(release_prompts, training_prompts)):
        raise ValueError("Normalized 10-gram/exact case overlap detected")
    return {
        "schema": 1,
        "status": "NERAL_V6_SUITE_INPUT_PREFLIGHT_ONLY",
        "counts": {name: {d: stats["count"] for d, stats in groups.items()}
                   for name, groups in audited.items()},
        "method_counts": {name: {d: stats["methods"] for d, stats in groups.items()}
                          for name, groups in audited.items()},
        "normalized_10gram_training_overlap": 0,
        "dev_release_text_overlap": 0,
        "semantically_unseen_proven": False,
        "human_independence_verified": False,
        "grader_correctness_verified": False,
        "suite_frozen": False,
        "release_validated": False,
        "weights_promoted": False,
        "limitations": ("Only exact/10-gram decontamination over explicitly supplied "
                        "training files. Semantic paraphrases and prior Foundation "
                        "pretraining cannot be excluded. Hidden Python test hashes "
                        "are syntax-checked but code correctness is not graded."),
    }


def audit_files(dev_path: Path, release_path: Path,
                train_files: list[Path], output: Path) -> dict:
    """Read private files locally; emit hash/count-only manifest create-once."""
    output = Path(output)
    dev_path, release_path = Path(dev_path), Path(release_path)
    if (output.is_symlink() or output.exists() or
            not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("Audit manifest must be new and in an existing directory")
    if not train_files or not all(isinstance(x, (str, Path)) for x in train_files):
        raise ValueError("All training sources required")
    sources = [dev_path, release_path, *(Path(x) for x in train_files)]
    if len({x.resolve() for x in sources}) != len(sources):
        raise ValueError("Repeated dataset source path")
    raw_dev, dev = _read_private_jsonl(dev_path, LIMIT_BYTES)
    raw_release, release = _read_private_jsonl(release_path, LIMIT_BYTES)
    training = []
    digests = []
    for path in train_files:
        raw, rows = _read_private_jsonl(Path(path), MAX_TRAINING_BYTES)
        training.extend(read_training_prompts(rows))
        if len(training) > 100000:
            raise ValueError("Training inventory too large")
        digests.append(hashlib.sha256(raw).hexdigest())
    manifest = audit_pair(dev, release, training)
    manifest.update({
        "dev_sha256": hashlib.sha256(raw_dev).hexdigest(),
        "release_sha256": hashlib.sha256(raw_release).hexdigest(),
        "training_sha256": digests,
        "training_files_checked": len(digests),
        "training_rows_checked": len(training),
        "intake_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    # Refuse to write raw answers or prompts into the scalar manifest.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--train-jsonl", action="append", required=True, type=Path,
                        help="Repeat for EVERY training corpus: v5, v6, preferences etc.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    outcome = audit_files(args.dev, args.release, args.train_jsonl, args.out)
    print(json.dumps({
        "status": outcome["status"], "counts": outcome["counts"],
        "release_validated": False, "human_independence_verified": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
