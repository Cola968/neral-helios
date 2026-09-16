"""Offline fixtures only: no real Neral checkpoint, corpus, CUDA or GPU work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import real_gpu_preflight as preflight


class RealGpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.input = root / "input"
        self.input.mkdir()
        self.anchor = self.input / "safe.pt"
        self.anchor.write_bytes(b"fictional anchor bytes")
        self.anchor_sha = self.sha(self.anchor)
        self.tokenizer_sha = "a" * 64
        self.corpus = self.input / "corpus"
        self.main_data = self.corpus / "train"
        self.main_data.mkdir(parents=True)
        self.main_shard = self.main_data / "train.bin"
        self.main_shard.write_bytes(b"fictional training shard")
        self.main_manifest = self.corpus / "manifest.json"
        self.main_manifest.write_text(json.dumps({
            "tokenizer": {"tokenizer_sha256": self.tokenizer_sha},
            "benchmark_content_included": False,
            "train_files": [{"path": "train.bin", "sha256": self.sha(self.main_shard)}],
        }), encoding="utf-8")
        self.replay = self.input / "replay"
        self.replay.mkdir()
        domains = {}
        for domain in ("education", "general", "math_reasoning"):
            directory = self.replay / domain
            directory.mkdir()
            shard = directory / "replay.bin"
            shard.write_bytes(("fictional " + domain).encode())
            domains[domain] = {"files": [{"path": "replay.bin", "sha256": self.sha(shard)}]}
        self.replay_manifest = self.replay / "manifest.json"
        self.replay_manifest.write_text(json.dumps({"tokenizer_sha256": self.tokenizer_sha,
                                                   "domains": domains}), encoding="utf-8")
        self.out_root = root / "outputs"
        self.out_root.mkdir()
        self.arm_a = self.out_root / "arm_a"
        self.arm_b = self.out_root / "arm_b"
        self.gpu = {"gpu": "RTX 4070", "free_mib": 10240, "temperature_c": 51,
                    "compute_processes": 0}
        minimum = mock.patch.object(preflight, "MIN_FREE_DISK_BYTES", 0)
        minimum.start()
        self.addCleanup(minimum.stop)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, **kw):
        params = dict(anchor=self.anchor, anchor_sha256=self.anchor_sha,
                      main_data=self.main_data, replay=self.replay,
                      tokenizer_sha256=self.tokenizer_sha, arm_a=self.arm_a,
                      arm_b=self.arm_b, gpu_probe=lambda: self.gpu)
        params.update(kw)
        return preflight.verify(**params)

    def test_complete_fixture_is_only_readiness_not_training(self):
        report = self.check()
        self.assertEqual(report["status"], "preflight_passed_not_training_evidence")
        self.assertEqual((report["main_shards_verified"], report["replay_shards_verified"]), (1, 3))
        self.assertFalse(report["training_started"])
        self.assertFalse(report["real_tokens_per_second_verified"])
        self.assertFalse(report["eligible_for_production"])
        self.assertFalse(self.arm_a.exists())
        self.assertFalse(self.arm_b.exists())

    def test_modified_anchor_and_wrong_pin_are_rejected(self):
        self.anchor.write_bytes(b"changed anchor")
        with self.assertRaisesRegex(ValueError, "safe anchor bytes differ"):
            self.check()
        with self.assertRaisesRegex(ValueError, "independently pinned"):
            self.check(anchor_sha256="not-a-sha")

    def test_modified_or_unlisted_data_is_rejected(self):
        self.main_shard.write_bytes(b"contaminated")
        with self.assertRaisesRegex(ValueError, "training shard SHA-256 mismatch"):
            self.check()
        self.main_shard.write_bytes(b"fictional training shard")
        self.replay_manifest.write_text(self.replay_manifest.read_text().replace(self.tokenizer_sha, "b" * 64))
        with self.assertRaisesRegex(ValueError, "replay tokenizer identity"):
            self.check()

    def test_extra_main_or_replay_bin_rejected_before_training(self):
        (self.main_data / "unlisted.bin").write_bytes(b"unlisted corpus")
        with self.assertRaisesRegex(ValueError, "unlisted or missing"):
            self.check()
        (self.main_data / "unlisted.bin").unlink()
        (self.replay / "math_reasoning" / "extra.bin").write_bytes(b"unlisted replay")
        with self.assertRaisesRegex(ValueError, "unlisted or missing"):
            self.check()

    def test_non_bin_manifest_entry_is_rejected(self):
        (self.main_data / "surprise.txt").write_bytes(b"not a trainer shard")
        data = json.loads(self.main_manifest.read_text())
        data["train_files"][0] = {"path": "surprise.txt", "sha256": self.sha(self.main_data / "surprise.txt")}
        self.main_manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be a "):
            self.check()

    def test_benchmark_contamination_and_duplicate_json_fields_fail(self):
        self.main_manifest.write_text(self.main_manifest.read_text().replace("false", "true"))
        with self.assertRaisesRegex(ValueError, "benchmark separation"):
            self.check()
        self.main_manifest.write_text('{"benchmark_content_included":false,"benchmark_content_included":false}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON manifest field"):
            self.check()

    def test_symlink_shard_and_traversal_fail(self):
        target = self.main_shard.with_name("target.bin")
        self.main_shard.rename(target)
        try:
            self.main_shard.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink unsupported")
        with self.assertRaisesRegex(ValueError, "symlink input"):
            self.check()
        self.main_shard.unlink()
        self.main_shard.write_bytes(b"fictional training shard")
        data = json.loads(self.main_manifest.read_text())
        data["train_files"][0]["path"] = "../escape.bin"
        self.main_manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "basename"):
            self.check()

    def test_output_overlap_reuse_and_duplicate_arms_fail(self):
        with self.assertRaisesRegex(ValueError, "distinct output"):
            self.check(arm_b=self.arm_a)
        self.arm_a.mkdir()
        with self.assertRaisesRegex(ValueError, "must not already exist"):
            self.check()
        self.arm_a.rmdir()
        with self.assertRaisesRegex(ValueError, "overlaps input"):
            self.check(arm_a=self.main_data / "unsafe_experiment")

    def test_missing_disk_or_busy_gpu_fail(self):
        with mock.patch.object(preflight, "MIN_FREE_DISK_BYTES", 10 ** 30):
            with self.assertRaisesRegex(ValueError, "insufficient disk"):
                self.check()
        self.gpu["compute_processes"] = 1
        with self.assertRaisesRegex(ValueError, "GPU readiness"):
            self.check()
        self.gpu["compute_processes"] = 0
        self.gpu["free_mib"] = 2000
        with self.assertRaisesRegex(ValueError, "GPU readiness"):
            self.check()

    def test_busy_gpu_does_not_hash_checkpoint_or_data(self):
        self.gpu["compute_processes"] = 2
        with mock.patch.object(preflight, "hash_file", side_effect=AssertionError("should not hash")):
            with self.assertRaisesRegex(ValueError, "GPU readiness"):
                self.check()

    def test_real_gpu_command_requires_idle_single_4070(self):
        output = lambda text: mock.Mock(stdout=text)
        with mock.patch.object(preflight.subprocess, "run", side_effect=[
            output("NVIDIA GeForce RTX 4070, 10240, 55\n"), output("\n")]):
            self.assertEqual(preflight.query_gpu()["compute_processes"], 0)
        with mock.patch.object(preflight.subprocess, "run", side_effect=[
            output("RTX 4070, 10240, 55\n"), output("1234\n")]):
            with self.assertRaisesRegex(ValueError, "compute_processes"):
                preflight.query_gpu()

    def test_gpu_error_explains_memory_temperature_and_process_count(self):
        output = lambda text: mock.Mock(stdout=text)
        with mock.patch.object(preflight.subprocess, "run", side_effect=[
            output("RTX 4070, 8843, 60\n"), output("12\n13\n")]):
            with self.assertRaisesRegex(ValueError, "free_mib=8843.*temperature_c=60.*compute_processes=2"):
                preflight.query_gpu()


if __name__ == "__main__":
    unittest.main()
