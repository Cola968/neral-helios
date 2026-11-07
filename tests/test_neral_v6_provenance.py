"""Source-index provenance and human sample-attestation contracts (CPU)."""
import hashlib
import unittest

from scripts.neral_v6_provenance import audit_provenance


def suite(label):
    cases, provenance, reviews = [], [], []
    for domain in ("math","code","reasoning"):
        for index in range(200):
            kind={"math":"numeric","code":"python","reasoning":"boolean"}[domain]
            ident=f"{label}-{domain}-{index:03}"
            answer=f"reference answer {index}"
            case={
                "id": ident, "domain": domain, "family": f"{domain}-family-{index%10}",
                "source_id": f"{label}-source-{domain}-{index:03}",
                "kind": kind, "prompt": f"{label} independent {domain} test {index} with fresh vocabulary",
                "answer": answer,
                "tests_sha256": "a"*64 if kind == "python" else None,
            }
            cases.append(case)
            provenance.append({
                "id": ident, "source_id": case["source_id"],
                "domain": domain, "family": case["family"], "language": "de" if index%2 else "en",
                "source_class": "original", "source": f"independent-review-{label}",
                "source_revision": "original-human",
                "source_index": index + {"math":0,"code":200,"reasoning":400}[domain] + (0 if label == "dev" else 600),
                "license": "author-original",
                "reference_sha256": hashlib.sha256(answer.encode("utf8")).hexdigest(),
                "tests_sha256": case["tests_sha256"],
            })
            if index < 20:
                reviews.append({
                    "id": ident, "reviewer": f"reviewer-{label}",
                    "checked_task": True, "checked_reference": True,
                    "checked_overlap": True, "reviewed_at_utc": "2026-11-07T18:00:00Z",
                })
    return cases, provenance, reviews


class ProvenanceTests(unittest.TestCase):
    def test_two_suites_all_cases_sourced_and_twenty_per_domain_reviewed(self):
        a=suite("dev"); b=suite("release")
        report=audit_provenance(*a, *b)
        self.assertEqual(report["rows"], {"dev":600,"release":600})
        self.assertEqual(report["review_attestations"], {"dev":60,"release":60})
        self.assertEqual(report["public_sources_count"], 0)
        self.assertFalse(report["human_review_independently_verified"])
        self.assertFalse(report["pretraining_unseen_proven"])
        self.assertFalse(report["suite_frozen"])
        self.assertFalse(report["release_validated"])

    def test_source_index_duplicate_across_suites_rejected(self):
        a=suite("dev"); b=suite("release")
        b[1][0]=dict(b[1][0], source=b[1][0]["source"].replace("release","dev"),
                     source_index=0)
        with self.assertRaises(ValueError):
            audit_provenance(*a,*b)

    def test_public_dataset_not_unseen_by_claim(self):
        a=suite("dev"); b=suite("release")
        a[1][0]=dict(a[1][0],source_class="public",source="gsm8k",
                     source_revision="f"*40)
        report=audit_provenance(*a,*b)
        self.assertEqual(report["public_sources_count"],1)
        self.assertFalse(report["pretraining_unseen_proven"])

    def test_review_missing_or_invalid_or_fabricated_label_is_rejected(self):
        a=suite("dev"); b=suite("release")
        for idx in (0,):
            with self.assertRaises(ValueError):
                audit_provenance(a[0],a[1],a[2][1:],*b)
            badreviews=list(a[2])
            badreviews[idx]=dict(badreviews[idx],checked_reference=False)
            with self.assertRaises(ValueError):
                audit_provenance(a[0],a[1],badreviews,*b)
        wrong=list(a[1])
        wrong[0]=dict(wrong[0],reference_sha256="c"*64)
        with self.assertRaises(ValueError):
            audit_provenance(a[0],wrong,a[2],*b)

    def test_mismatched_source_id_weak_origin_and_test_hash_rejected(self):
        a=suite("dev"); b=suite("release")
        for modified in (
            dict(a[1][0],source_id="not-matching"),
            dict(a[1][0],source_revision="main"),
            dict(a[1][0],tests_sha256="b"*64),
            dict(a[1][0],language="unknown"),
        ):
            changed=list(a[1]); changed[0]=modified
            with self.assertRaises(ValueError):
                audit_provenance(a[0],changed,a[2],*b)


if __name__=="__main__":
    unittest.main()
