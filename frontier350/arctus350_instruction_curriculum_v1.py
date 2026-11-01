"""Original bilingual NERAL instruction curriculum for the native Arctus350.

Creates locally generated, fully reproducible and exactly verifiable examples.
No Qwen/teacher invocation, GPU, training, model weight access, downloads,
private user data or network. The VAL/TEST partitions are *development*
holdouts, not independently frozen external benchmark evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile

from .arctus350_exact_math_curriculum import TOKENS, SOURCE_LICENSE

SOURCE = "neral-original-bilingual-instruction-v1"
SCHEMA = "arctus350-original-instruction-curriculum-v1"
_GROUPS_PER_TASK = 256
_ROW_ID = re.compile(r"ins1-(\d{4})-(de|en)-(select|sort)\Z")


def _task(n: int, language: str, kind: str) -> tuple[str, str]:
    if type(n) is not int or not 0 <= n < _GROUPS_PER_TASK:
        raise ValueError("out-of-range instruction identity")
    if language not in ("de", "en"):
        raise ValueError("language must be de/en")
    if kind == "select":
        # Unique ordered token triples across n; two languages share the same
        # semantic group and therefore always reside in the SAME split.
        x = n % len(TOKENS)
        y = (x + 1 + n // len(TOKENS)) % len(TOKENS)
        z = (x + 2 + (n // len(TOKENS)) * 7) % len(TOKENS)
        while z in (x, y):
            z = (z + 1) % len(TOKENS)
        a, b, c = TOKENS[x], TOKENS[y], TOKENS[z]
        prompt = (
            f"Gib nur das zweite Wort zurück: {a} | {b} | {c}."
            if language == "de" else
            f"Return only the second word: {a} | {b} | {c}."
        )
        return prompt, b
    if kind == "sort":
        # The group-specific number >=1000 prevents semantic collisions
        # across groups; stable per-identity shuffling prevents fixed position.
        nums = [1000 + n, (13 * n + 7) % 1000, (23 * n + 15) % 1000]
        if len(set(nums)) != 3:
            raise ValueError("non-distinct generated number group")
        seed = int.from_bytes(
            hashlib.sha256(f"neral-sort-v1:{n}".encode("ascii")).digest()[:8], "big"
        )
        random.Random(seed).shuffle(nums)
        fields = ", ".join(str(v) for v in nums)
        prompt = (
            f"Sortiere diese drei Zahlen aufsteigend: {fields}."
            if language == "de" else
            f"Sort these three numbers in ascending order: {fields}."
        )
        return prompt, ",".join(str(v) for v in sorted(nums))
    raise ValueError("invalid instruction task family")


def _row(n: int, language: str, kind: str) -> dict:
    prompt, answer = _task(n, language, kind)
    return {
        "id": f"ins1-{n:04d}-{language}-{kind}",
        "language": language,
        "domain": "instruction",
        "task_type": "select_second" if kind == "select" else "sort_three",
        "source": SOURCE,
        "license": SOURCE_LICENSE,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def verify_row(row: object) -> bool:
    """Independently reconstruct the canonical task and label from its ID."""
    if not isinstance(row, dict) or set(row) != {
        "id", "language", "domain", "task_type", "source", "license", "messages"
    }:
        return False
    identity = row.get("id")
    match = _ROW_ID.fullmatch(identity) if isinstance(identity, str) else None
    if match is None:
        return False
    n, language, kind = int(match.group(1)), match.group(2), match.group(3)
    try:
        return row == _row(n, language, kind)
    except (TypeError, ValueError, IndexError):
        return False


def _partition(n: int, kind: str) -> str:
    digest = hashlib.sha256(f"neral-original-instruction-v1:{kind}:{n}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def make_splits() -> dict[str, list[dict]]:
    """1024 native examples; no semantic task group reused between splits."""
    if len(TOKENS) != 48 or len(set(TOKENS)) != 48:
        raise ValueError("locked project-original word vocabulary changed")
    if any(re.fullmatch(r"[A-Z]{3,12}", token) is None for token in TOKENS):
        raise ValueError("invalid word-selection vocabulary")
    output = {name: [] for name in ("train", "val", "test")}
    semantic_groups = {}
    owners = {}
    for kind in ("select", "sort"):
        for n in range(_GROUPS_PER_TASK):
            split = _partition(n, kind)
            # English and German versions are *never* split apart.
            for language in ("de", "en"):
                row = _row(n, language, kind)
                if not verify_row(row):
                    raise ValueError("independently generated instruction invalid")
                task_group = (kind, n)
                if owners.setdefault(task_group, split) != split:
                    raise ValueError("bilingual semantic group split contamination")
                output[split].append(row)
            # Do not accidentally reuse a semantically identical sort task
            # or identical ordered selection triple in a different split.
            prompt, _ = _task(n, "en", kind)
            from .arctus350_teacher_gate import _independent_task
            semantic_type = "select_second" if kind == "select" else "sort_three"
            _, fingerprint = _independent_task(semantic_type, prompt, "en")
            prior = semantic_groups.setdefault(fingerprint, task_group)
            if prior != task_group:
                raise ValueError("repeated semantic instruction group")
    if sum(len(rows) for rows in output.values()) != 1024:
        raise ValueError("original instruction counts changed")
    ids = set()
    for split, rows in output.items():
        if len(rows) < 60:
            raise ValueError("insufficient independent split coverage")
        counts = Counter((row["language"], row["task_type"]) for row in rows)
        if any(counts[language, kind] < 10 for language in ("de", "en")
               for kind in ("select_second", "sort_three")):
            raise ValueError("missing bilingual instruction task coverage")
        for row in rows:
            if row["id"] in ids:
                raise ValueError("duplicate generated native instruction ID")
            ids.add(row["id"])
    return output


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_local_dataset(destination: Path) -> dict:
    """Atomically materialize a bounded local *unapproved* NeralTalk350 candidate."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError("refuse to overwrite existing dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = make_splits()
    from .chat350_data import validate_dataset_root, validate_prompt_holdouts
    with tempfile.TemporaryDirectory(prefix=".arctus-native-instruction-", dir=target.parent) as staging:
        stage = Path(staging)
        for split, rows in parts.items():
            with (stage / f"{split}.jsonl").open("x", encoding="utf8") as output:
                for row in rows:
                    # Native validator normalizes away the supplemental task_type.
                    clean = {k: v for k, v in row.items() if k != "task_type"}
                    output.write(json.dumps(clean, sort_keys=True, ensure_ascii=False) + "\n")
        manifest, parsed = validate_dataset_root(stage)
        validate_prompt_holdouts(parsed)
        report = {
            "schema": SCHEMA, "status": "prepared_local_only",
            "model_family": "neral-arctus350", "source": SOURCE,
            "license": SOURCE_LICENSE,
            "split_counts": {split: len(parts[split]) for split in ("train", "val", "test")},
            "languages": {split: dict(sorted(Counter(r["language"] for r in parts[split]).items()))
                          for split in ("train", "val", "test")},
            "task_families": {split: dict(sorted(Counter(r["task_type"] for r in parts[split]).items()))
                              for split in ("train", "val", "test")},
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "output_sha256": {split: _sha(stage / f"{split}.jsonl")
                              for split in ("train", "val", "test")},
            "source_labels_independently_verified": True,
            "qwen_used": False, "training_started": False,
            "training_approved": False, "native_quality_improved": False,
            "public_benchmark": False, "release_allowed": False,
            "limitation": "Synthetic development holdouts only, not unseen real-world knowledge or proven model improvement.",
        }
        (stage / "metadata_report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf8",
        )
        if target.exists():
            raise FileExistsError("destination created during local preparation")
        os.rename(stage, target)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(write_local_dataset(args.out), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
