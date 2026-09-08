from hf_revision_audit50 import _norm_license, audit_one, resolve

SHA = "a" * 40


def test_license_aliases():
    assert _norm_license("odc-by") == {"odc-by-1.0"}
    assert _norm_license("Apache-2.0") == {"apache-2.0"}


def test_approved_source_pins_revision_and_license():
    row = audit_one("HuggingFaceFW/fineweb-edu", {"sha": SHA, "cardData": {"license": "odc-by"}})
    assert row["revision"] == SHA
    assert row["license_match"] is True
    assert row["status"] == "approved_revision_pinned"


def test_future_candidate_stays_manual_review():
    row = audit_one("HuggingFaceTB/finemath", {"sha": SHA, "cardData": {"license": "odc-by"}})
    assert row["license_match"] is True
    assert row["training_allowed_by_policy"] is False
    assert row["status"] == "candidate_revision_pinned_manual_review_required"


def test_missing_license_stays_quarantined():
    row = audit_one("HuggingFaceTB/stack-edu", {"sha": SHA, "cardData": {}})
    assert row["license_match"] is False
    assert row["status"] == "quarantine_metadata_mismatch_or_missing_license"


def test_resolve_does_not_change_policy():
    info = {
        "HuggingFaceFW/fineweb-2": {"sha": SHA, "cardData": {"license": "odc-by"}},
        "HuggingFaceTB/finemath": {"sha": "b" * 40, "cardData": {"license": "odc-by"}},
    }
    out = resolve(list(info), fetcher=lambda dataset_id: info[dataset_id])
    assert out["read_only"] is True
    assert out["changes_training_allow_list"] is False
    assert out["all_currently_approved_metadata_match"] is True
