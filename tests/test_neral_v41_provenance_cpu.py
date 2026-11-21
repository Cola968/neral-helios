"""CPU-only fail-closed provenance checks for historical V3.9 and V3.10."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import neral_v41_blind200_build as build


class PinnedProvenanceCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.v39 = Path(self.tmp.name) / "v39.jsonl"
        self.v310 = Path(self.tmp.name) / "v310.jsonl"
        self.write(self.v39, [
            {"instruction": f"V39 instruction {i}", "meta": {"source": "v39_benchmark"}}
            for i in range(97)
        ])
        self.write(self.v310, [
            {"question": f"V310 question {cat} {i}", "category": cat}
            for cat in sorted(build.V310_CATEGORIES) for i in range(20)
        ])
        self.p39 = patch.object(build, "V39_SHA256", self.digest(self.v39))
        self.p310 = patch.object(build, "V310_SHA256", self.digest(self.v310))
        self.p39.start()
        self.p310.start()
        self.addCleanup(self.p39.stop)
        self.addCleanup(self.p310.stop)

    @staticmethod
    def write(path, rows):
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def audit(self):
        return build._prior_rows([], required=(self.v39, self.v310))

    def test_accepts_exact_verified_archives_and_excludes_all_197(self):
        seen, provenance = self.audit()
        self.assertEqual(len(seen), 197)
        self.assertEqual(provenance, {"v39_benchmark", "v310_private_holdout"})

    def test_missing_archive_fails_closed(self):
        self.v310.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.audit()

    def test_modified_bytes_fail_sha256(self):
        self.v310.write_bytes(self.v310.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.audit()

    def test_wrong_category_distribution_fails_even_with_matching_hash(self):
        rows = [json.loads(line) for line in self.v310.read_text(encoding="utf-8").splitlines()]
        rows[0]["category"] = "web"
        self.write(self.v310, rows)
        with patch.object(build, "V310_SHA256", self.digest(self.v310)):
            with self.assertRaisesRegex(ValueError, "category provenance mismatch"):
                self.audit()


if __name__ == "__main__":
    unittest.main()
