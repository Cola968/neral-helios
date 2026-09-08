"""Resolve immutable Hugging Face dataset revisions and compare live metadata with Neral source policy.

Read-only: this tool never downloads training rows and never changes the training allow-list.
It records the Hub commit SHA and license metadata so a later corpus builder can pin exact
revisions before any weight-changing experiment.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from hf_source_policy50 import SOURCES, validate_registry

SHA40 = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_DATASETS = (
    "HuggingFaceFW/fineweb-2",
    "HuggingFaceFW/fineweb-edu",
    "OpenAssistant/oasst2",
    "HuggingFaceTB/finemath",
    "HuggingFaceTB/stack-edu",
)


def _norm_license(value: object) -> set[str]:
    if value is None:
        return set()
    values = value if isinstance(value, list) else [value]
    out = set()
    for item in values:
        text = str(item).strip().lower().replace("_", "-")
        aliases = {
            "odc-by": "odc-by-1.0",
            "odc-by-1-0": "odc-by-1.0",
            "apache-2": "apache-2.0",
            "apache2": "apache-2.0",
        }
        out.add(aliases.get(text, text))
    return out


def fetch_dataset_info(dataset_id: str, timeout: int = 20) -> dict:
    url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(dataset_id, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": "neral-helios-hf-audit/1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def audit_one(dataset_id: str, info: dict) -> dict:
    if dataset_id not in SOURCES:
        raise ValueError(f"unregistered dataset: {dataset_id}")
    policy = SOURCES[dataset_id]
    revision = str(info.get("sha", "")).lower()
    if not SHA40.fullmatch(revision):
        raise ValueError(f"{dataset_id}: Hub did not return an immutable 40-hex revision")
    card = info.get("cardData") or {}
    live_license = _norm_license(card.get("license"))
    policy_license = _norm_license(policy.license)
    license_match = bool(live_license & policy_license) if live_license else False
    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "last_modified": info.get("lastModified"),
        "policy_role": policy.role,
        "training_allowed_by_policy": policy.allow_training,
        "policy_license": policy.license,
        "hub_license": sorted(live_license),
        "license_match": license_match,
        "content_origin": policy.content_origin,
        "status": (
            "approved_revision_pinned" if policy.allow_training and license_match
            else "candidate_revision_pinned_manual_review_required" if not policy.allow_training and license_match
            else "quarantine_metadata_mismatch_or_missing_license"
        ),
    }


def resolve(dataset_ids: list[str], fetcher=fetch_dataset_info) -> dict:
    validate_registry()
    rows = [audit_one(dataset_id, fetcher(dataset_id)) for dataset_id in dataset_ids]
    return {
        "version": 1,
        "purpose": "Immutable Hugging Face revision and license preflight for Neral",
        "read_only": True,
        "changes_training_allow_list": False,
        "datasets": rows,
        "all_currently_approved_metadata_match": all(
            row["license_match"] for row in rows if row["training_allowed_by_policy"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", dest="datasets")
    ap.add_argument("--out", default="HF_REVISION_AUDIT.json")
    args = ap.parse_args()
    dataset_ids = args.datasets or list(DEFAULT_DATASETS)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("duplicate dataset id")
    result = resolve(dataset_ids)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["all_currently_approved_metadata_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
