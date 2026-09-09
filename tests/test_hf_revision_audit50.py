from hf_revision_audit50 import (
    DEFAULT_DATASETS,
    REQUIRED_DATASETS,
    _language_tags,
    _norm_license,
    audit_one,
    resolve,
)
from hf_source_policy50 import SOURCES

SHA = "a" * 40


def info(license_name="odc-by", languages=("en",)):
    return {
        "sha": SHA,
        "tags": [f"language:{lang}" for lang in languages],
        "cardData": {"license": license_name, "language": list(languages)},
    }


def test_license_aliases():
    assert _norm_license("odc-by") == {"odc-by-1.0"}
    assert _norm_license("Apache-2.0") == {"apache-2.0"}
    assert _norm_license("cc-by-sa-3") == {"cc-by-sa-3.0"}


def test_language_tags_merge_api_tags_and_card_data():
    tags = _language_tags({"tags": ["language:deu"], "cardData": {"language": ["en"]}})
    assert tags == {"language:deu", "language:en"}


def test_default_audit_covers_every_trainable_source():
    trainable = {dataset_id for dataset_id, policy in SOURCES.items() if policy.allow_training}
    assert trainable.issubset(set(REQUIRED_DATASETS))
    assert trainable.issubset(set(DEFAULT_DATASETS))


def test_approved_source_pins_revision_license_and_language():
    row = audit_one("HuggingFaceFW/fineweb-edu", info())
    assert row["revision"] == SHA
    assert row["license_match"] is True
    assert row["language_match"] is True
    assert row["metadata_match"] is True
    assert row["status"] == "approved_revision_pinned"


def test_fineweb2_requires_german_language_tag():
    row = audit_one("HuggingFaceFW/fineweb-2", info(languages=("en",)))
    assert row["license_match"] is True
    assert row["language_match"] is False
    assert row["metadata_match"] is False
    assert row["status"] == "quarantine_metadata_mismatch_or_missing_required_tags"


def test_oasst_requires_both_german_and_english():
    row = audit_one("OpenAssistant/oasst2", info("apache-2.0", languages=("de", "en")))
    assert row["language_match"] is True

    row_missing_de = audit_one("OpenAssistant/oasst2", info("apache-2.0", languages=("en",)))
    assert row_missing_de["language_match"] is False


def test_future_candidate_stays_manual_review():
    row = audit_one("HuggingFaceTB/finemath", info())
    assert row["license_match"] is True
    assert row["training_allowed_by_policy"] is False
    assert row["status"] == "candidate_revision_pinned_manual_review_required"


def test_missing_license_stays_quarantined():
    row = audit_one(
        "HuggingFaceTB/stack-edu",
        {"sha": SHA, "tags": [], "cardData": {}},
    )
    assert row["license_match"] is False
    assert row["status"] == "quarantine_metadata_mismatch_or_missing_required_tags"


def test_resolve_does_not_change_policy_and_reports_incomplete_subset():
    rows = {
        "HuggingFaceFW/fineweb-2": info(languages=("deu",)),
        "HuggingFaceTB/finemath": {**info(), "sha": "b" * 40},
    }
    out = resolve(list(rows), fetcher=lambda dataset_id: rows[dataset_id])
    assert out["read_only"] is True
    assert out["changes_training_allow_list"] is False
    assert out["all_currently_approved_metadata_match"] is True
    assert out["all_trainable_sources_covered"] is False


def test_resolve_can_prove_complete_trainable_coverage():
    trainable = [dataset_id for dataset_id, p in SOURCES.items() if p.allow_training]
    rows = {}
    for dataset_id in trainable:
        if dataset_id == "HuggingFaceFW/fineweb-2":
            rows[dataset_id] = info(languages=("deu",))
        elif dataset_id in {"OpenAssistant/oasst1", "OpenAssistant/oasst2"}:
            rows[dataset_id] = info("apache-2.0", languages=("de", "en"))
        else:
            rows[dataset_id] = info(languages=("en",))
    out = resolve(trainable, fetcher=lambda dataset_id: rows[dataset_id])
    assert out["all_trainable_sources_covered"] is True
    assert out["all_currently_approved_metadata_match"] is True
