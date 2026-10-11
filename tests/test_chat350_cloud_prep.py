from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.chat350_contract import (
    EXPECTED_CONFIG, EXPECTED_SPECIALS, EXPECTED_TOKENIZER_SHA256,
    release_identity, validate_checkpoint_metadata, validate_tokenizer_lock,
)
from frontier350.chat350_data import validate_dataset_root, validate_prompt_holdouts
from frontier350.chat350_generation_metrics import compare_health, summarize_generations
from frontier350.chat350_sft_format import encode_messages, encode_prompt_answer, render_multiturn

class _Encoding:
    def __init__(self, ids): self.ids = ids

class FakeTokenizer:
    specials = {"<|bos|>": 0, "<|eos|>": 1, "<|sep|>": 2, "<|pad|>": 3}
    def token_to_id(self, token): return self.specials.get(token)
    def encode(self, text): return _Encoding([10 + (ord(ch) % 200) for ch in text])

class Chat350CloudPrepTests(unittest.TestCase):
    def test_lock_contract(self):
        validate_tokenizer_lock({
            "architecture":"frontier350-v1","vocab_size":16384,
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "special_token_ids":EXPECTED_SPECIALS,"foreign_vocab_or_merges_loaded":False,
        })

    def test_checkpoint_metadata_accepts_fresh_optimizer_handoff(self):
        validate_checkpoint_metadata({
            "model":{"placeholder":True},"config":EXPECTED_CONFIG,
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"processed_tokens":2_000_000_000,
        })

    def test_checkpoint_rejects_wrong_tokenizer(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata({
                "model":{},"config":EXPECTED_CONFIG,"tokenizer_sha256":"0"*64,
                "processed_tokens":2_000_000_000,
            })

    def test_assistant_only_alignment_single_and_multi(self):
        tok = FakeTokenizer()
        one = encode_prompt_answer(tok, "Hallo", "Hi!", max_seq_len=256)
        self.assertEqual(len(one.input_ids), len(one.targets))
        self.assertEqual(one.targets[-1], 1)
        multi = encode_messages(tok, [
            {"role":"user","content":"Mein Name ist Max."},
            {"role":"assistant","content":"Verstanden."},
            {"role":"user","content":"Wie heiße ich?"},
            {"role":"assistant","content":"Max."},
        ], max_seq_len=512)
        self.assertGreater(multi.assistant_tokens, one.assistant_tokens)
        self.assertEqual(multi.targets[-1], 1)

    def test_multiturn_render(self):
        text = render_multiturn([
            {"role":"user","content":"A"},{"role":"assistant","content":"B"},
            {"role":"user","content":"C"},{"role":"assistant","content":"D"},
        ])
        self.assertIn("Assistant: D", text)

    def test_dataset_manifest_and_cross_split_dedupe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def write(name, idx, prompt):
                row={"id":idx,"language":"de","domain":"general","source":"unit","license":"CC0",
                     "prompt":prompt,"answer":"Antwort"}
                (root/f"{name}.jsonl").write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf8")
            write("train","tr1","Frage eins"); write("val","va1","Frage zwei"); write("test","te1","Frage drei")
            manifest,_ = validate_dataset_root(root)
            self.assertEqual(manifest["total_rows"],3)
            self.assertEqual(manifest["global_unique_conversations"],3)

    def test_dataset_rejects_duplicate_conversation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            base={"language":"de","domain":"general","source":"unit","license":"CC0","prompt":"gleich","answer":"gleich"}
            for split, rid in [("train","a"),("val","b"),("test","c")]:
                row=dict(base); row["id"]=rid
                if split=="test": row["prompt"]="anders"
                (root/f"{split}.jsonl").write_text(json.dumps(row)+"\n",encoding="utf8")
            with self.assertRaises(ValueError): validate_dataset_root(root)

    def test_generation_health(self):
        base=summarize_generations([
            {"generated_ids":[5,6,7,5,6,7],"text":"x","expected_exact":"OK"}
        ], eos_id=1)
        cand=summarize_generations([
            {"generated_ids":[8,9,1],"text":"OK","expected_exact":"OK"}
        ], eos_id=1)
        gate=compare_health(base,cand)
        self.assertTrue(gate["health_no_regression"])

    def test_release_identity(self):
        self.assertEqual(release_identity()["parameters"],349_825_152)

if __name__=="__main__": unittest.main()
