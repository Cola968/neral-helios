"""Synthetic-only tests. Never access local OASST2 source, model or GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phoenix_review_decisions import FIELDS, check, validate


def fixture():
    rows = [{"id": f"synthetic-{i}", "split": "train" if i < 12 else "validation",
             "language": "de" if i % 2 else "en", "user": f"fake prompt {i}",
             "assistant": f"fake reply {i}", **{field: "pending" for field in FIELDS}}
            for i in range(24)]
    queue = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    digest = hashlib.sha256(queue).hexdigest()
    receipt = json.dumps({"review_file_sha256": digest, "rows": 24,
                          "all_human_reviews_pending": True, "training_approved": False,
                          "training_started": False}).encode()
    payload = {"version": 1, "queue_sha256": digest, "training_approved": False,
               "rows": [{"id": item["id"], "decisions": {field: "pass" for field in FIELDS},
                         "note": ""} for item in rows]}
    return queue, receipt, payload, digest


class PrivateReviewDecisionTests(unittest.TestCase):
    def test_complete_review_never_approves_training(self):
        queue, receipt, payload, digest = fixture()
        payload["rows"][0]["decisions"]["privacy_review"] = "reject"
        report = validate(queue, receipt, json.dumps(payload).encode(), digest)
        self.assertTrue(report["all_sample_decisions_present"])
        self.assertEqual(report["field_counts"]["privacy_review"], {"pass": 23, "reject": 1})
        self.assertIs(report["training_approved"], False)
        self.assertIs(report["entire_training_corpus_reviewed"], False)
        self.assertIs(report["independent_holdout_checked"], False)

    def test_fail_closed_on_tampering_and_incomplete_review(self):
        queue, receipt, payload, digest = fixture()
        for mutate in (
            lambda p: p.update(training_approved=True),
            lambda p: p.update(queue_sha256="0" * 64),
            lambda p: p["rows"][0].update(id="unlisted"),
            lambda p: p["rows"][1].update(id=p["rows"][0]["id"]),
            lambda p: p["rows"][0]["decisions"].update(privacy_review=None),
            lambda p: p["rows"][0]["decisions"].update(privacy_review="pending"),
            lambda p: p["rows"][0]["decisions"].update(unknown="pass"),
            lambda p: p["rows"][0].update(note="s" * 1001),
            lambda p: p.update(rows=p["rows"][:-1]),
            lambda p: p.update(version=True),
        ):
            _, _, candidate, _ = fixture()
            mutate(candidate)
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(ValueError):
                    validate(queue, receipt, json.dumps(candidate).encode(), digest)
        with self.assertRaises(ValueError):
            validate(queue + b" ", receipt, json.dumps(payload).encode(), digest)
        with self.assertRaises(ValueError):
            validate(queue, receipt.replace(b'"rows": 24', b'"rows": 23'),
                     json.dumps(payload).encode(), digest)
        duplicate_keys = json.dumps(payload).replace('"training_approved": false',
                                                    '"training_approved": true, "training_approved": false')
        with self.assertRaises(ValueError):
            validate(queue, receipt, duplicate_keys.encode(), digest)

    def test_private_local_output_is_new_and_not_approval(self):
        queue, receipt, payload, digest = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q, r, d, out = [root / name for name in ("queue.jsonl", "receipt.json", "decisions.json", "report.json")]
            q.write_bytes(queue)
            r.write_bytes(receipt)
            d.write_text(json.dumps(payload), encoding="utf-8")
            summary = check(root, q, r, d, out, digest)
            self.assertFalse(summary["training_approved"])
            self.assertTrue(out.exists())
            self.assertFalse(json.loads(out.read_text(encoding="utf-8"))["training_approved"])
            with self.assertRaises(ValueError):
                check(root, q, r, d, out, digest)
            with self.assertRaises(ValueError):
                check(root, q, r, d, root.parent / "review_escape.json", digest)
            symlink = root / "linked.json"
            symlink.symlink_to(d)
            with self.assertRaises(ValueError):
                check(root, q, r, symlink, root / "other.json", digest)
            self.assertFalse((root / "other.json").exists())


if __name__ == "__main__":
    unittest.main()
