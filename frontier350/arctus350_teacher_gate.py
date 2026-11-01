"""Offline, fail-closed Qwen teacher SFT candidate for Arctus350.

No teacher calls, downloads, network, tokenizer, GPU or training.
Output is an unapproved, reviewable NeralTalk dataset candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from .chat350_data import normalize_row, validate_dataset_root, validate_prompt_holdouts

TEACHER = "Qwen/Qwen3.5-9B"
TEACHER_LICENSE = "Apache-2.0"  # Upstream model license, not a grant for input data
TEACHER_SOURCE = "neral-teacher-qwen35-9b-reviewed-v1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_LINE_BYTES = 12 * 1024
MAX_ROWS = 2_000
HASH = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,90}\Z")
ADDITION = re.compile(r"(?:Berechne|Calculate|Add)\s+(\d{1,4})\s*\+\s*(\d{1,4})\.?\Z")
TWO_STEP = re.compile(r"(?:Berechne|Calculate)\s+\((\d{1,4})\s*\+\s*(\d{1,4})\)\s*-\s*(\d{1,4})\.?\Z")
ORDER_EN = re.compile(
    r"([A-Z][a-z]{2,15}) is faster than ([A-Z][a-z]{2,15})\. "
    r"([A-Z][a-z]{2,15}) is faster than ([A-Z][a-z]{2,15})\. "
    r"Who is (fastest|slowest)\?\Z"
)
ORDER_DE = re.compile(
    r"([A-ZÄÖÜ][a-zäöüß]{2,15}) ist schneller als ([A-ZÄÖÜ][a-zäöüß]{2,15})\. "
    r"([A-ZÄÖÜ][a-zäöüß]{2,15}) ist schneller als ([A-ZÄÖÜ][a-zäöüß]{2,15})\. "
    r"Wer ist am (schnellsten|langsamsten)\?\Z"
)
TEACHER_KEYS = frozenset({
    "id", "split", "language", "domain", "task_type", "prompt", "answer",
    "independent_reference", "human_reviewed", "teacher_model",
    "teacher_artifact_sha256", "teacher_license", "generation_mode", "tools_used",
})


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON value forbidden")


def _read_jsonl(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing, symlinked or oversized source file")
    rows = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if line_number > MAX_ROWS or not raw.strip() or len(raw) > MAX_LINE_BYTES:
                raise ValueError("empty, oversized or excessive source rows")
            try:
                row = json.loads(raw.decode("utf8"), object_pairs_hook=_unique_pairs,
                                 parse_constant=_reject_constant)
            except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError("malformed source JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError("source row must be object")
            rows.append(row)
    if not rows:
        raise ValueError("empty source dataset")
    return rows


def _prompt_key(content: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", content).casefold().split())


def _text(value, *, limit=4096):
    return isinstance(value, str) and 1 <= len(value.strip()) <= limit and not any(
        ord(c) < 32 and c not in "\n\t" for c in value
    )


def _independent_task(task_type: str, prompt: str, language: str):
    """Compute reference and semantic fingerprint; no arbitrary expression evaluation."""
    if task_type == "addition":
        match = ADDITION.fullmatch(prompt.strip())
        prefix = "Berechne" if language == "de" else ("Calculate", "Add")
        if match is None or not prompt.strip().startswith(prefix):
            raise ValueError("unsupported exact addition grammar")
        a, b = (int(n) for n in match.groups())
        return str(a + b), ("addition", *sorted((a, b)))
    if task_type == "two_step":
        match = TWO_STEP.fullmatch(prompt.strip())
        if match is None or not prompt.strip().startswith(
            "Berechne" if language == "de" else "Calculate"
        ):
            raise ValueError("unsupported two-step arithmetic grammar")
        a, b, c = (int(n) for n in match.groups())
        return str(a + b - c), ("two_step", *sorted((a, b)), c)
    if task_type == "ordered_logic":
        match = (ORDER_DE if language == "de" else ORDER_EN).fullmatch(prompt.strip())
        if match is None:
            raise ValueError("unsupported strict transitive reasoning grammar")
        first, second, third, fourth, question = match.groups()
        names = {first, second, third, fourth}
        if len(names) != 3 or first == second or third == fourth:
            raise ValueError("need exactly three distinct ordered entities")
        edges = {(first, second), (third, fourth)}
        if len(edges) != 2:
            raise ValueError("repeated or cyclic order fact")
        incoming = {name: 0 for name in names}
        outgoing = {name: 0 for name in names}
        for faster, slower in edges:
            incoming[slower] += 1
            outgoing[faster] += 1
        fastest = [n for n in names if incoming[n] == 0]
        slowest = [n for n in names if outgoing[n] == 0]
        middle = [n for n in names if incoming[n] == 1 and outgoing[n] == 1]
        if len(fastest) != 1 or len(slowest) != 1 or len(middle) != 1:
            raise ValueError("not a complete, unambiguous three-node chain")
        if not (outgoing[fastest[0]] == 1 and incoming[slowest[0]] == 1):
            raise ValueError("incomplete reasoning chain")
        asks_fastest = question in {"fastest", "schnellsten"}
        ordered = (fastest[0], middle[0], slowest[0])
        return ordered[0 if asks_fastest else 2], ("ordered_logic", ordered, asks_fastest)
    if task_type == "exact":
        # Arbitrary text is NOT independently verifiable: explicit human review required.
        return None, None
    raise ValueError("unsupported independently verifiable task type")


def _read_teacher(path: Path):
    output = {"train": [], "val": []}
    seen_ids = set()
    seen_prompts = set()
    seen_semantics = set()
    artifact = None
    for row in _read_jsonl(path):
        if row.keys() != TEACHER_KEYS:
            raise ValueError("unknown/missing teacher fields")
        identity = row["id"]
        if not isinstance(identity, str) or not IDENTIFIER.fullmatch(identity) or identity in seen_ids:
            raise ValueError("invalid/duplicate teacher id")
        seen_ids.add(identity)
        if row["split"] not in output or row["language"] not in {"de", "en"}:
            raise ValueError("teacher examples only permitted in train/val, DE/EN")
        if row["domain"] not in {"math", "instruction", "grounding", "general", "reasoning"}:
            raise ValueError("unsupported teacher task domain")
        if row["task_type"] not in {"addition", "two_step", "ordered_logic", "exact"}:
            raise ValueError("only independently exact-verifiable task types in pilot")
        if row["teacher_model"] != TEACHER or row["teacher_license"] != TEACHER_LICENSE:
            raise ValueError("teacher identity or upstream model license not allowed")
        digest = row["teacher_artifact_sha256"]
        if not isinstance(digest, str) or HASH.fullmatch(digest) is None:
            raise ValueError("immutable local teacher artifact SHA-256 required")
        if artifact is not None and artifact != digest:
            raise ValueError("mixed teacher artifact versions")
        artifact = digest
        if row["generation_mode"] != "model-native" or row["tools_used"] != []:
            raise ValueError("tool-mediated or unattributed model output")
        if row["human_reviewed"] is not True:
            raise ValueError("human-reviewed provenance required")
        prompt, answer, reference = row["prompt"], row["answer"], row["independent_reference"]
        if not _text(prompt) or not _text(answer) or not _text(reference):
            raise ValueError("empty, oversized or unsafe teacher text")
        if answer.strip() != reference.strip():
            raise ValueError("teacher answer disagrees with independent reference")
        calculated, semantic_key = _independent_task(
            row["task_type"], prompt, row["language"]
        )
        if calculated is not None and reference.strip() != calculated:
            raise ValueError("teacher answer contradicts independently computed reference")
        if semantic_key is not None:
            if semantic_key in seen_semantics:
                raise ValueError("duplicate semantic reasoning task across splits")
            seen_semantics.add(semantic_key)
        key = _prompt_key(prompt)
        if key in seen_prompts:
            raise ValueError("duplicate normalized teacher prompt")
        seen_prompts.add(key)
        output[row["split"]].append(normalize_row({
            "id": identity, "language": row["language"], "domain": row["domain"],
            "source": TEACHER_SOURCE,
            "license": "NERAL-private-reviewed-training-only",
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
        }))
    if not output["train"] or not output["val"]:
        raise ValueError("both teacher train and validation required")
    return output, artifact


def _verified_frozen_test(path: Path):
    """Admit only independently re-derived, project-original test labels."""
    from .arctus350_grounding_curriculum_v1 import (
        SOURCE as GROUNDING_SOURCE, verify_row, make_splits as grounding_splits
    )
    from .arctus350_exact_math_curriculum import (
        independently_verify_row, make_splits as exact_math_splits
    )
    # A valid reference does NOT prove membership in the frozen TEST partition:
    # a TRAIN row could otherwise be relabeled as a heldout case.
    grounding_test_ids = {r["id"] for r in grounding_splits()["test"]}
    exact_math_test_ids = {r["id"] for r in exact_math_splits()["test"]}
    rows = []
    for row in _read_jsonl(path):
        if row.get("source") == GROUNDING_SOURCE:
            valid = verify_row(row) and row.get("id") in grounding_test_ids
        elif row.get("source") in {"neral-generated-exact-v1", "neral-generated-small-add-v1"}:
            valid = independently_verify_row(row) and row.get("id") in exact_math_test_ids
        else:
            valid = False
        if not valid:
            raise ValueError("frozen test has nonoriginal or corrupted independent reference")
        rows.append(normalize_row(row))
    return rows


def _sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_teacher_candidate(teacher_jsonl, frozen_test_jsonl, destination):
    """Stage reviewed train/val plus verified project test; never train."""
    teacher_path, test_path = Path(teacher_jsonl), Path(frozen_test_jsonl)
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("refusing to overwrite any candidate directory")
    if dest.parent.is_symlink() or not dest.parent.is_dir():
        raise ValueError("candidate output parent must be an existing real directory")
    if teacher_path.resolve() == test_path.resolve():
        raise ValueError("teacher and frozen test must be distinct sources")
    parts, artifact = _read_teacher(teacher_path)
    parts["test"] = _verified_frozen_test(test_path)
    prompt_keys = set()
    ids = set()
    for split in ("train", "val", "test"):
        for row in parts[split]:
            if row["id"] in ids:
                raise ValueError("duplicate id across splits")
            ids.add(row["id"])
            for message in row["messages"]:
                if message["role"] != "user":
                    continue
                key = _prompt_key(message["content"])
                if key in prompt_keys:
                    raise ValueError("teacher/frozen prompt overlap")
                prompt_keys.add(key)
    validate_prompt_holdouts(parts)
    # Stage atomically: an incomplete run never publishes a candidate folder.
    with tempfile.TemporaryDirectory(prefix=".arctus-teacher-staging-", dir=dest.parent) as temp:
        stage = Path(temp)
        for split in ("train", "val", "test"):
            content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")) + "\n" for row in parts[split])
            (stage / f"{split}.jsonl").write_text(content, encoding="utf8")
        native_manifest, readback = validate_dataset_root(stage)
        validate_prompt_holdouts(readback)
        report = {
            "schema": "arctus350-qwen-teacher-candidate-v1",
            "teacher_model": TEACHER,
            "teacher_model_license": TEACHER_LICENSE,
            "teacher_artifact_sha256": artifact,
            "teacher_outputs_are_native": True,
            "teacher_generation_verified_by_runner": False,
            "human_review_assertion_independently_verified": False,
            "frozen_test_source_verified": True,
            "split_counts": {split: len(parts[split]) for split in ("train", "val", "test")},
            "input_sha256": {"teacher": _sha(teacher_path), "frozen_test": _sha(test_path)},
            "output_sha256": {split: _sha(stage / f"{split}.jsonl") for split in ("train", "val", "test")},
            "dataset_manifest_sha256": native_manifest["manifest_sha256"],
            "training_started": False, "training_approved": False,
            "native_quality_improved": False, "release_allowed": False,
            "public_benchmark": False,
            "contamination_limitations": "Exact normalized prompt checks only; independent historical/near-duplicate and frozen-public-benchmark screening still required.",
        }
        (stage / "teacher_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"
        )
        if dest.exists():
            raise FileExistsError("output was created by another process")
        os.rename(stage, dest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-jsonl", required=True, type=Path)
    parser.add_argument("--frozen-test-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = prepare_teacher_candidate(args.teacher_jsonl, args.frozen_test_jsonl, args.out)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
