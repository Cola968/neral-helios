from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from phoenix_oasst1_on_refresh import (
    FOUNDATION_CLASS,
    FOUNDATION_SCHEMA,
    load_refresh,
)
from helios350_preview import TOKENIZER_SHA


class FakeTorch:
    def __init__(self, state):
        self.state = state

    def load(self, path, map_location=None, weights_only=False):
        return self.state


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_state(config: dict) -> dict:
    return {
        "checkpoint_schema": FOUNDATION_SCHEMA,
        "classification": FOUNDATION_CLASS,
        "config": config,
        "tokenizer_sha256": TOKENIZER_SHA,
        "production_eligible": False,
        "lineage": {
            "kind": "phoenix_foundation_refresh_fresh_adamw_v1",
        },
        "model": {"w": object()},
        "optimizer": {"state": {0: {"exp_avg": object()}}},
        "processed_tokens": 1_048_576,
        "refresh_manifest_sha256": "a" * 64,
    }


class RefreshContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "resume.pt"
        self.path.write_bytes(b"owned-refresh-checkpoint-placeholder")
        self.digest = sha(self.path)
        self.config = {"n_layers": 24, "vocab_size": 16384}

    def tearDown(self):
        self.tmp.cleanup()

    def test_accepts_structurally_complete_development_refresh(self):
        state = valid_state(self.config)
        loaded = load_refresh(FakeTorch(state), self.path, self.digest, self.config)
        self.assertEqual(loaded["processed_tokens"], 1_048_576)

    def test_rejects_wrong_sha_before_load(self):
        state = valid_state(self.config)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            load_refresh(FakeTorch(state), self.path, "0" * 64, self.config)

    def test_rejects_missing_optimizer_state(self):
        state = valid_state(self.config)
        state["optimizer"] = {"state": {}}
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_refresh(FakeTorch(state), self.path, self.digest, self.config)

    def test_rejects_wrong_lineage(self):
        state = valid_state(self.config)
        state["lineage"]["kind"] = "historical_exact_resume"
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_refresh(FakeTorch(state), self.path, self.digest, self.config)

    def test_rejects_production_claim(self):
        state = valid_state(self.config)
        state["production_eligible"] = True
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_refresh(FakeTorch(state), self.path, self.digest, self.config)

    def test_rejects_zero_processed_tokens(self):
        state = valid_state(self.config)
        state["processed_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_refresh(FakeTorch(state), self.path, self.digest, self.config)


if __name__ == "__main__":
    unittest.main()
