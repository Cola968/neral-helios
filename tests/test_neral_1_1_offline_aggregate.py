"""Synthetic ZIP fixtures only; no original blind-test cases."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from scripts.neral_1_1_offline_aggregate import aggregate


class OfflineAggregateCPU(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for n in range(100):
            category = "logic" if n < 25 else "mathematics" if n < 50 else "mixed"
            fail = n < 17 or 25 <= n < 31
            for arm in ("qwen", "neral"):
                rows.append({
                    "arm": arm, "id": f"synthetic-{n:03d}",
                    "category": category, "family": "synthetic_family",
                    "correct": not fail,
                    "error_code": "token_limit" if fail else None,
                    "decision": "explicit_safe_refusal" if fail else "raw_qwen",
                    "extra": {"extraction_attempts": [], "route": "logic_z3"},
                    "proof": {"z3": False},
                    "reply": "SYNTHETIC_CONTENT_MUST_NOT_LEAK",
                })
        raw = ("\n".join(json.dumps(x) for x in rows) + "\n").encode("utf8")
        digest, seal = hashlib.sha256(raw).hexdigest(), "synthetic-seal"
        path = root / "synthetic.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("raw-200.jsonl", raw)
            z.writestr("summary.json", json.dumps({
                "raw_sha256": digest, "source_run_id": "43722550890",
                "sealed_sha256": seal, "source_commit": "synthetic-commit"}))
            z.writestr("new-seal.json", json.dumps({"dataset_sha256": seal}))
            z.writestr("inference-complete.json", "{}")
        return path, digest, seal

    def test_aggregate_only_metadata_never_model_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path, digest, seal = self.fixture(Path(temp))
            with patch("scripts.neral_1_1_offline_aggregate.EXPECTED_RAW_SHA", digest), \
                 patch("scripts.neral_1_1_offline_aggregate.EXPECTED_SEAL", seal):
                report = aggregate(path)
            self.assertEqual(report["categories"]["logic"]["failed"], 17)
            self.assertEqual(report["categories"]["mathematics"]["failed"], 6)
            self.assertEqual(report["failure_count"], 23)
            self.assertNotIn("SYNTHETIC_CONTENT_MUST_NOT_LEAK", json.dumps(report))

    def test_wrong_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path, digest, seal = self.fixture(Path(temp))
            with patch("scripts.neral_1_1_offline_aggregate.EXPECTED_RAW_SHA", "invalid"), \
                 patch("scripts.neral_1_1_offline_aggregate.EXPECTED_SEAL", seal):
                with self.assertRaisesRegex(ValueError, "raw_evidence_mismatch"):
                    aggregate(path)

    def test_missing_raw_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("summary.json", "{}")
            with self.assertRaisesRegex(ValueError, "raw_evidence_missing"):
                aggregate(path)


if __name__ == "__main__":
    unittest.main()
