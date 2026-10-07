from __future__ import annotations
import unittest

from frontier350.chat350_contract import (
    EXPECTED_CONFIG,
    EXPECTED_SPECIALS,
    EXPECTED_TOKENIZER_SHA256,
    release_identity,
    validate_checkpoint_metadata,
    validate_tokenizer_lock,
)
from frontier350.chat350_sft_format import encode_prompt_answer, render_multiturn


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    specials = {"<|bos|>": 0, "<|eos|>": 1, "<|sep|>": 2, "<|pad|>": 3}

    def token_to_id(self, token):
        return self.specials.get(token)

    def encode(self, text):
        # Deterministic stand-in; production uses the locally locked 16K BPE.
        return _Encoding([10 + (ord(ch) % 200) for ch in text])


class Chat350CloudPrepTests(unittest.TestCase):
    def test_lock_contract(self):
        validate_tokenizer_lock({
            "architecture": "frontier350-v1",
            "vocab_size": 16384,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "special_token_ids": EXPECTED_SPECIALS,
            "foreign_vocab_or_merges_loaded": False,
        })

    def test_checkpoint_metadata_accepts_fresh_optimizer_handoff(self):
        validate_checkpoint_metadata({
            "model": {"placeholder": True},
            "config": EXPECTED_CONFIG,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "processed_tokens": 2_000_000_000,
        })

    def test_checkpoint_rejects_wrong_tokenizer(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata({
                "model": {},
                "config": EXPECTED_CONFIG,
                "tokenizer_sha256": "0" * 64,
                "processed_tokens": 2_000_000_000,
            })

    def test_assistant_only_alignment(self):
        row = encode_prompt_answer(FakeTokenizer(), "Hallo", "Hi!", max_seq_len=256)
        self.assertEqual(len(row.input_ids), len(row.targets))
        self.assertGreater(row.assistant_tokens, 0)
        first_supervised = next(i for i, value in enumerate(row.targets) if value != -100)
        self.assertGreater(first_supervised, 0)
        self.assertEqual(row.targets[-1], 1)

    def test_multiturn_contract(self):
        text = render_multiturn([
            {"role": "user", "content": "Mein Name ist Max."},
            {"role": "assistant", "content": "Verstanden."},
            {"role": "user", "content": "Wie heiße ich?"},
            {"role": "assistant", "content": "Max."},
        ])
        self.assertIn("Assistant: Max.", text)

    def test_release_identity(self):
        self.assertEqual(release_identity()["parameters"], 349_825_152)


if __name__ == "__main__":
    unittest.main()
