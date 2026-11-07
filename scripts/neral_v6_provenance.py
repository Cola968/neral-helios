"""CPU audit of private 600+600 NERAL test provenance and human attestations.

Source-index and reviews are separate private sidecars; the existing eight-field
suite format remains unchanged. This cannot prove genuine human independence.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re

from scripts.neral_v6_suite_intake import audit_suite, _read_private_jsonl

FIELDS = {
    "id", "source_id", "domain", "family", "language", "source_class",
    "source", "source_revision", "source_index", "license",
    "reference_sha256", "tests_sha256",
}
REVIEW_FIELDS = {
    "id", "reviewer", "checked_task", "checked_reference",
    "checked_overlap", "reviewed_at_utc",
}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_.\-/]{1,120}\Z")
DATE = re.compile(r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


def _audit_one(cases: list[dict], provenance: list[dict],
               reviews: list[dict], label: str) -> tuple[dict, set, int]:
    audit_suite(cases)
    if len(provenance) != 600 or not isinstance(reviews, list):
        raise ValueError("Every task needs provenance and reviewed samples")
    indexed = {row["id"]: row for row in cases}
    provenance_ids = set()
    roots = set()
    public = 0
    for row in provenance:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected provenance metadata")
        identity = row["id"]
        if identity not in indexed or identity in provenance_ids:
            raise ValueError("Foreign or duplicate case provenance")
        provenance_ids.add(identity)
        source = indexed[identity]
        if any(row[field] != source[field] for field in
               ("source_id","domain","family","tests_sha256")):
            raise ValueError("Provenance does not match case contents")
        if row["language"] not in ("de", "en"):
            raise ValueError("Unknown task language")
        if row["source_class"] not in ("public", "original"):
            raise ValueError("Missing original/public source classification")
        for field in ("source", "license"):
            if not isinstance(row[field],str) or not NAME.fullmatch(row[field]):
                raise ValueError("Missing source or license evidence")
        if row["source_class"] == "public":
            public += 1
            if (not isinstance(row["source_revision"],str)
                    or not HEX40.fullmatch(row["source_revision"])):
                raise ValueError("Public dataset needs immutable 40-char revision")
        elif row["source_revision"] != "original-human":
            raise ValueError("Original human source needs explicit origin marker")
        if type(row["source_index"]) is not int or row["source_index"] < 0:
            raise ValueError("Original index must be a nonnegative integer")
        ref = row["reference_sha256"]
        if (not isinstance(ref, str) or not HEX64.fullmatch(ref)
                or ref != sha256(source["answer"].encode("utf8")).hexdigest()):
            raise ValueError("Reference answer fingerprint mismatch")
        key = (row["source_class"], row["source"], row["source_revision"],
               row["source_index"])
        if key in roots:
            raise ValueError("Duplicated underlying task origin within suite")
        roots.add(key)
    if provenance_ids != set(indexed):
        raise ValueError("Provenance coverage incomplete")

    review_ids = set()
    by_domain = Counter()
    for row in reviews:
        if not isinstance(row,dict) or set(row) != REVIEW_FIELDS:
            raise ValueError("Unexpected review attestation")
        ident = row["id"]
        if ident not in indexed or ident in review_ids:
            raise ValueError("Reviewer sample repeated or unknown")
        review_ids.add(ident)
        if (not isinstance(row["reviewer"], str) or not NAME.fullmatch(row["reviewer"])
                or not isinstance(row["reviewed_at_utc"],str)
                or not DATE.fullmatch(row["reviewed_at_utc"])):
            raise ValueError("Human reviewer identity/time required")
        if any(type(row[f]) is not bool or row[f] is not True
               for f in ("checked_task","checked_reference","checked_overlap")):
            raise ValueError("Failed/unreviewed case cannot count toward required sample")
        by_domain[indexed[ident]["domain"]] += 1
    if any(by_domain[domain] < 20 for domain in ("math","code","reasoning")):
        raise ValueError("Need at least 20 reviewed tasks per domain")
    return {"cases":600, "review_rows":len(reviews),
            "method_families":len({r["family"] for r in cases}),
            "languages":dict(Counter(r["language"] for r in provenance))}, roots, public


def audit_provenance(dev: list[dict], dev_origin: list[dict], dev_reviews: list[dict],
                     release: list[dict], release_origin: list[dict],
                     release_reviews: list[dict]) -> dict:
    da, droots, pubd = _audit_one(dev, dev_origin, dev_reviews, "dev")
    ra, rroots, pubr = _audit_one(release, release_origin, release_reviews, "release")
    if droots & rroots:
        raise ValueError("Underlying task origin reused across development/release")
    if {r["source_id"] for r in dev} & {r["source_id"] for r in release}:
        raise ValueError("Source identity reused across development/release")
    return {
        "schema":1, "status":"NERAL_V6_ORIGIN_AND_REVIEW_ATTESTATION_ONLY",
        "rows":{"dev":da["cases"],"release":ra["cases"]},
        "review_attestations":{"dev":da["review_rows"],
                               "release":ra["review_rows"]},
        "languages":{"dev":da["languages"],"release":ra["languages"]},
        "public_sources_count":pubd+pubr,
        "human_review_independently_verified":False,
        "pretraining_unseen_proven":False,
        "suite_frozen":False,
        "release_validated":False, "weights_promoted":False,
        "limitations": (
            "Attestation text and SHA checks only: can be fabricated, and source "
            "uniqueness alone does not show Qwen pretraining novelty. Public "
            "benchmarks such as GSM8K/HumanEval may be contaminated. Independent "
            "expert review and immutable pre-registration are separate mandatory gates."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("dev", "dev-provenance", "dev-review",
                "release", "release-provenance", "release-review", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args=parser.parse_args(argv)
    source_files = [args.dev,args.dev_provenance,args.dev_review,
                    args.release,args.release_provenance,args.release_review]
    output=args.out
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.parent.is_symlink()):
        raise ValueError("New-only report path required")
    files = []
    hashes = []
    for p in source_files:
        raw, rows = _read_private_jsonl(p, 8_000_000)
        hashes.append(sha256(raw).hexdigest())
        files.append(rows)
    report = audit_provenance(*files)
    report["source_file_sha256"] = hashes
    with output.open("x",encoding="utf8") as fp:
        json.dump(report,fp,sort_keys=True)
        fp.write("\n")
    print("NERAL_V6_PROVENANCE_PREFLIGHT: no release approval")


if __name__=="__main__":
    main()
