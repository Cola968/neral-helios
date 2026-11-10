"""Re-bind an SFT pack to its approval evidence immediately before training.

This is a fail-closed integrity gate only. It cannot establish that a human review
was competent or that legal/privacy conclusions are correct. It prevents a pilot
from trusting only a copied pack_report.json after the original manifest,
screening report, or frozen holdout changed.
"""
from __future__ import annotations

import json
from pathlib import Path

from sft_contract import hex_digest, read_json, sha


def verify_runtime_evidence(pack_report: dict, manifest_path: Path,
                            screening_path: Path, holdout_path: Path) -> dict:
    if not isinstance(pack_report, dict) or pack_report.get("schema") != 1:
        raise ValueError("valid packed SFT report required")
    for path, label in ((manifest_path, "manifest"), (screening_path, "screening"),
                        (holdout_path, "holdout")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime {label} evidence must be a regular local file")

    manifest_sha = sha(manifest_path)
    screening_sha = sha(screening_path)
    if manifest_sha != hex_digest(pack_report.get("source_manifest_sha256"), "pack manifest"):
        raise ValueError("pack no longer binds to supplied source manifest")
    if screening_sha != hex_digest(pack_report.get("screen_report_sha256"), "pack screening"):
        raise ValueError("pack no longer binds to supplied screening report")

    manifest = read_json(manifest_path)
    screening = read_json(screening_path)
    holdout = read_json(holdout_path)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("runtime source manifest v1 required")
    if not isinstance(screening, dict) or screening.get("version") != 1:
        raise ValueError("runtime independent screening report v1 required")
    if not isinstance(holdout, list) or not holdout:
        raise ValueError("runtime frozen holdout registry required")
    hashes = [hex_digest(item, "holdout prompt") for item in holdout]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate runtime holdout hashes")

    for field in ("train_sha256", "validation_sha256", "tokenizer_sha256",
                  "holdout_sha256", "screen_report_sha256"):
        hex_digest(manifest.get(field), field)
    if sha(holdout_path) != manifest["holdout_sha256"]:
        raise ValueError("frozen holdout changed after SFT packing")
    if screening_sha != manifest["screen_report_sha256"]:
        raise ValueError("manifest screening identity changed")
    for field in ("train_sha256", "validation_sha256", "holdout_sha256"):
        if screening.get(field) != manifest[field]:
            raise ValueError(f"runtime screening {field} mismatch")
    if screening.get("approved") is not True:
        raise ValueError("runtime screening is not explicitly approved")
    if any(screening.get(field) != 0 for field in
           ("exact_overlap_count", "near_overlap_count", "privacy_hits")):
        raise ValueError("runtime screening contains unresolved training-data findings")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("runtime manifest has no sources")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid runtime source row")
        ident = source.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("invalid or duplicate runtime source identity")
        seen.add(ident)
        hex_digest(source.get("revision_sha256"), "source revision")
        if not isinstance(source.get("license"), str) or not source["license"].strip():
            raise ValueError("runtime source license missing")
        if any(source.get(key) is not True for key in
               ("approved_for_training", "privacy_reviewed", "benchmark_screened")):
            raise ValueError("runtime source approval was revoked or incomplete")

    return {
        "status": "runtime_evidence_bound_for_bounded_pilot_only",
        "source_manifest_sha256": manifest_sha,
        "screen_report_sha256": screening_sha,
        "holdout_sha256": manifest["holdout_sha256"],
        "source_count": len(sources),
        "screening_approved": True,
        "training_started": False,
        "production_approved": False,
    }
