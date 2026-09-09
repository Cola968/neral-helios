"""Resolve immutable Hugging Face dataset revisions and compare live metadata with Neral policy.

Read-only: this tool never downloads training rows and never changes the training allow-list.
It records Hub commit SHAs and validates license/language metadata for approved training sources
and permanently reserved public evaluation datasets.
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

# Sources that must never silently disappear from the default audit. This is derived from the
# executable policy instead of being copied by hand, so adding an approved source automatically
# makes it part of the live provenance check.
REQUIRED_DATASETS = tuple(
    sorted(
        dataset_id
        for dataset_id, policy in SOURCES.items()
        if policy.allow_training or policy.role == "public-eval-holdout"
    )
)
DEFAULT_DATASETS = REQUIRED_DATASETS + tuple(
    dataset_id
    for dataset_id in ("HuggingFaceTB/finemath", "HuggingFaceTB/stack-edu")
    if dataset_id not in REQUIRED_DATASETS
)

# Only stable requirements that are meaningful for Neral are encoded here. FineWeb2 publishes
# ISO-639-3 tags (deu), while OpenAssistant uses the two-letter language tags (de/en).
REQUIRED_LANGUAGE_TAGS = {
    "HuggingFaceFW/fineweb": {"language:en"},
    "HuggingFaceFW/fineweb-2": {"language:deu"},
    "HuggingFaceFW/fineweb-edu": {"language:en"},
    "OpenAssistant/oasst1": {"language:de", "language:en"},
    "OpenAssistant/oasst2": {"language:de", "language:en"},
    "openai/gsm8k": {"language:en"},
    "google-research-datasets/mbpp": {"language:en"},
    "deepmind/code_contests": {"language:en"},
    "google/IFEval": {"language:en"},
    "truthfulqa/truthful_qa": {"language:en"},
    "google/boolq": {"language:en"},
    "allenai/ai2_arc": {"language:en"},
}


def _norm_license(value):
    if value is None:
        return set()
    values = value if isinstance(value, list) else [value]
    out = set()
    aliases = {
        "odc-by": "odc-by-1.0",
        "odc-by-1-0": "odc-by-1.0",
        "apache-2": "apache-2.0",
        "apache2": "apache-2.0",
        "cc-by-4": "cc-by-4.0",
        "cc-by-4-0": "cc-by-4.0",
        "cc-by-sa-3": "cc-by-sa-3.0",
        "cc-by-sa-3-0": "cc-by-sa-3.0",
        "cc-by-sa-4": "cc-by-sa-4.0",
        "cc-by-sa-4-0": "cc-by-sa-4.0",
    }
    for item in values:
        text = str(item).strip().lower().replace("_", "-")
        out.add(aliases.get(text, text))
    return out


def _language_tags(info):
    tags = {str(tag).strip().lower() for tag in (info.get("tags") or [])}
    card = info.get("cardData") or {}
    languages = card.get("language") or []
    if isinstance(languages, str):
        languages = [languages]
    tags.update(f"language:{str(lang).strip().lower()}" for lang in languages if str(lang).strip())
    return {tag for tag in tags if tag.startswith("language:")}


def fetch_dataset_info(dataset_id, timeout=20):
    url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(dataset_id, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": "neral-helios-hf-audit/3"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def audit_one(dataset_id, info):
    if dataset_id not in SOURCES:
        raise ValueError(f"unregistered dataset: {dataset_id}")

    policy = SOURCES[dataset_id]
    revision = str(info.get("sha", "")).lower()
    if not SHA40.fullmatch(revision):
        raise ValueError(f"{dataset_id}: Hub did not return immutable revision")

    card = info.get("cardData") or {}
    live_license = _norm_license(card.get("license"))
    expected_license = _norm_license(policy.license)
    license_match = bool(live_license & expected_license) if live_license else False

    live_languages = _language_tags(info)
    required_languages = REQUIRED_LANGUAGE_TAGS.get(dataset_id, set())
    language_match = required_languages.issubset(live_languages) if required_languages else True
    metadata_match = license_match and language_match

    if policy.allow_training and metadata_match:
        status = "approved_revision_pinned"
    elif policy.role == "public-eval-holdout" and metadata_match:
        status = "public_eval_revision_pinned_never_train"
    elif not policy.allow_training and metadata_match:
        status = "candidate_revision_pinned_manual_review_required"
    else:
        status = "quarantine_metadata_mismatch_or_missing_required_tags"

    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "last_modified": info.get("lastModified"),
        "policy_role": policy.role,
        "training_allowed_by_policy": policy.allow_training,
        "policy_license": policy.license,
        "hub_license": sorted(live_license),
        "license_match": license_match,
        "required_language_tags": sorted(required_languages),
        "hub_language_tags": sorted(live_languages),
        "language_match": language_match,
        "metadata_match": metadata_match,
        "content_origin": policy.content_origin,
        "status": status,
    }


def resolve(dataset_ids, fetcher=fetch_dataset_info):
    validate_registry()
    rows = [audit_one(dataset_id, fetcher(dataset_id)) for dataset_id in dataset_ids]
    audited = {row["dataset_id"] for row in rows}
    required_trainable = {dataset_id for dataset_id, p in SOURCES.items() if p.allow_training}
    audited_trainable = {row["dataset_id"] for row in rows if row["training_allowed_by_policy"]}

    return {
        "version": 3,
        "purpose": "Immutable Hugging Face revision, license and language preflight for Neral",
        "read_only": True,
        "changes_training_allow_list": False,
        "datasets": rows,
        "required_trainable_sources": sorted(required_trainable),
        "audited_trainable_sources": sorted(audited_trainable),
        "all_trainable_sources_covered": required_trainable.issubset(audited),
        "all_currently_approved_metadata_match": all(
            row["metadata_match"] for row in rows if row["training_allowed_by_policy"]
        ),
        "all_public_eval_holdouts_nontrainable": all(
            not row["training_allowed_by_policy"]
            for row in rows
            if row["policy_role"] == "public-eval-holdout"
        ),
        "all_public_eval_metadata_match": all(
            row["metadata_match"] for row in rows if row["policy_role"] == "public-eval-holdout"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", dest="datasets")
    ap.add_argument("--out", default="HF_REVISION_AUDIT.json")
    args = ap.parse_args()
    ids = args.datasets or list(DEFAULT_DATASETS)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset id")

    out = resolve(ids)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Custom targeted audits may intentionally inspect only a subset. The default live audit,
    # however, must cover every trainable source so an allow-listed dataset cannot escape drift checks.
    default_coverage_ok = bool(args.datasets) or out["all_trainable_sources_covered"]
    if not (
        default_coverage_ok
        and out["all_currently_approved_metadata_match"]
        and out["all_public_eval_holdouts_nontrainable"]
        and out["all_public_eval_metadata_match"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
