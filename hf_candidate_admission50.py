"""Fail-closed admission audit for future Hugging Face training candidates.

This module does not download data and does not change the training allow-list. It validates a
locally produced provenance/contamination manifest before a future corpus can even be considered
for a Neral weight-changing experiment. Frozen public evaluation datasets remain permanently
reserved and cannot pass this gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from hf_source_policy50 import REGISTRY_VERSION, SOURCES

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_SCREEN_KEYS = (
    "exact_prompt_matches",
    "near_duplicate_matches",
    "answer_leak_matches",
)


def _sha256(value: object, field: str) -> str:
    text = str(value).lower()
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _revision(value: object) -> str:
    text = str(value).lower()
    if not REVISION_RE.fullmatch(text):
        raise ValueError("revision must be an immutable 40-hex Hugging Face commit")
    return text


def _nonnegative_int(value: object, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out < 0:
        raise ValueError(f"{field} must be >= 0")
    return out


def assess_manifest(manifest: dict) -> dict:
    dataset_id = str(manifest.get("dataset_id", ""))
    if dataset_id not in SOURCES:
        raise ValueError(f"unregistered Hugging Face dataset: {dataset_id!r}")
    policy = SOURCES[dataset_id]

    failures: list[str] = []
    if policy.role == "public-eval-holdout":
        failures.append("public_eval_holdout")
    if policy.role.startswith("blocked-"):
        failures.append("source_policy_blocked")
    if policy.role not in {"future-pretrain-candidate", "pretrain", "sft-human"}:
        failures.append("unsupported_candidate_role")

    revision = _revision(manifest.get("revision"))
    if str(manifest.get("license", "")) != policy.license:
        failures.append("license_mismatch")
    if str(manifest.get("content_origin", "")) != policy.content_origin:
        failures.append("content_origin_mismatch")

    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict) or not file_hashes:
        failures.append("missing_file_hashes")
        file_hashes = {}
    else:
        for path, digest in sorted(file_hashes.items()):
            if not str(path).strip():
                failures.append("empty_file_path")
                continue
            try:
                _sha256(digest, f"file_hashes[{path!r}]")
            except ValueError:
                failures.append("invalid_file_hash")

    try:
        cleaning_sha = _sha256(manifest.get("cleaning_pipeline_sha256"), "cleaning_pipeline_sha256")
    except ValueError:
        cleaning_sha = None
        failures.append("invalid_cleaning_pipeline_sha256")

    screen = manifest.get("benchmark_screen")
    if not isinstance(screen, dict):
        screen = {}
        failures.append("missing_benchmark_screen")
    try:
        benchmark_manifest_sha = _sha256(screen.get("benchmark_manifest_sha256"), "benchmark_manifest_sha256")
    except ValueError:
        benchmark_manifest_sha = None
        failures.append("invalid_benchmark_manifest_sha256")

    overlap = {}
    for key in REQUIRED_SCREEN_KEYS:
        try:
            overlap[key] = _nonnegative_int(screen.get(key), key)
        except ValueError:
            overlap[key] = None
            failures.append(f"invalid_{key}")
    if any(v not in (0, None) for v in overlap.values()):
        failures.append("benchmark_contamination_detected")

    sampled_rows = _nonnegative_int(manifest.get("sampled_rows", 0), "sampled_rows")
    external_generated = _nonnegative_int(
        manifest.get("external_model_generated_rows", 0), "external_model_generated_rows"
    )
    if sampled_rows <= 0:
        failures.append("no_rows_audited")
    if external_generated != 0:
        failures.append("external_model_generated_training_rows")

    frozen_ids = set(map(str, manifest.get("reserved_eval_dataset_ids", [])))
    required_holdouts = {k for k, v in SOURCES.items() if v.role == "public-eval-holdout"}
    if not required_holdouts.issubset(frozen_ids):
        failures.append("reserved_eval_set_incomplete")

    unique_failures = sorted(set(failures))
    audit_pass = not unique_failures
    return {
        "dataset_id": dataset_id,
        "policy_role": policy.role,
        "registry_version": REGISTRY_VERSION,
        "revision": revision,
        "audit_pass": audit_pass,
        "failures": unique_failures,
        "training_allowed_by_current_policy": bool(policy.allow_training),
        "eligible_for_manual_policy_review": bool(audit_pass and not policy.allow_training),
        "already_training_approved": bool(audit_pass and policy.allow_training),
        "evidence": {
            "files_hashed": len(file_hashes),
            "cleaning_pipeline_sha256": cleaning_sha,
            "benchmark_manifest_sha256": benchmark_manifest_sha,
            "benchmark_overlap": overlap,
            "sampled_rows": sampled_rows,
            "external_model_generated_rows": external_generated,
        },
        "next_action": (
            "manual policy review; do not train until hf_source_policy50.py is explicitly changed"
            if audit_pass and not policy.allow_training
            else "source already approved; pin this audit with the run manifest"
            if audit_pass
            else "quarantine source; fix audit failures before any weight-changing experiment"
        ),
        "policy": (
            "Passing this audit never changes the training allow-list. Public evaluation holdouts "
            "cannot pass. Frozen benchmark prompts/answers are screening targets only, never training data."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    obj = json.loads(Path(args.manifest).read_text(encoding="utf8"))
    result = assess_manifest(obj)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf8")
    print(text)
    if not result["audit_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
