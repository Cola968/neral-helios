"""CPU-only byte fixtures. These are NOT authentic model checkpoints or training."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from checkpoint_artifacts import verify_artifacts
from test_admission import fixture
from test_stage_ledger import encode, synthetic_events


class CheckpointArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.names = [f"boundary_{i}.pt" for i in range(4)]
        self.digests = []
        for index, filename in enumerate(self.names):
            data = ("toy-checkpoint-not-real-" + str(index)).encode("ascii")
            (self.directory / filename).write_bytes(data)
            self.digests.append(hashlib.sha256(data).hexdigest())
        self.events = synthetic_events()
        for index, event in enumerate(self.events):
            event["checkpoint_start_sha256"] = self.digests[index]
            event["checkpoint_end_sha256"] = self.digests[index + 1]
        self.record = fixture()
        self.record["baseline"]["checkpoint_sha256"] = self.digests[0]
        self.record["candidate"]["checkpoint_sha256"] = self.digests[-1]
        self.manifest = {"schema": 1, "checkpoints": [
            {"sha256": digest, "filename": name}
            for digest, name in zip(self.digests, self.names)
        ]}

    def run_verify(self):
        raw, digest = encode(self.events)
        return verify_artifacts(self.record, raw, digest, self.directory, self.manifest)

    def test_matches_all_real_bytes_without_claiming_training(self):
        result = self.run_verify()
        self.assertEqual(result["checkpoint_count"], 4)
        self.assertTrue(result["checkpoint_bytes_verified"])
        self.assertTrue(result["contract_passed"])
        self.assertFalse(result["independent_training_verified"])
        self.assertFalse(result["exact_resume_verified"])
        self.assertFalse(result["eligible_for_production"])

    def test_one_changed_byte_rejected(self):
        (self.directory / self.names[2]).write_bytes(b"altered checkpoint")
        with self.assertRaisesRegex(ValueError, "bytes do not match"):
            self.run_verify()

    def test_missing_or_empty_checkpoint_rejected(self):
        (self.directory / self.names[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_verify()
        (self.directory / self.names[1]).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_verify()

    def test_manifest_requires_exact_boundary_set(self):
        self.manifest["checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "every boundary"):
            self.run_verify()
        self.manifest["checkpoints"].append({"sha256": "a" * 64, "filename": self.names[-1]})
        with self.assertRaisesRegex(ValueError, "bytes do not match"):
            self.run_verify()

    def test_path_traversal_and_duplicate_filename_rejected(self):
        self.manifest["checkpoints"][0]["filename"] = "../steal.pt"
        with self.assertRaisesRegex(ValueError, "safe basename"):
            self.run_verify()
        self.manifest["checkpoints"][0]["filename"] = self.names[0]
        self.manifest["checkpoints"][1]["filename"] = self.names[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_verify()

    def test_symlink_checkpoint_rejected(self):
        link = self.directory / self.names[1]
        link.unlink()
        try:
            link.symlink_to(self.directory / self.names[0])
        except (NotImplementedError, OSError):
            self.skipTest("OS does not permit symlinks")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_verify()

    def test_pinned_ledger_digest_required(self):
        raw, _ = encode(self.events)
        with self.assertRaisesRegex(ValueError, "pinned SHA"):
            verify_artifacts(self.record, raw, "0" * 64, self.directory, self.manifest)

    def test_quality_regression_never_becomes_release_approval(self):
        self.record["candidate"]["dev_categories"]["math"] = 0.0
        result = self.run_verify()
        self.assertTrue(result["checkpoint_bytes_verified"])
        self.assertFalse(result["contract_passed"])
        self.assertFalse(result["eligible_for_production"])
        self.assertIn("regression_dev_math", result["contract_failures"])


if __name__ == "__main__":
    unittest.main()
