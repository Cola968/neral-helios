"""Regression for shipping independent localhost NL extraction without dev fixtures.

CPU fake-transport only; no network, models, GPUs, historic/hidden corpus.
"""
import json
import unittest
from unittest.mock import patch
from neral_nl_local_ollama import make_local_extractor


class LocalNLExtractorRuntimeTests(unittest.TestCase):
    def test_exact_pinned_models_use_v2_source_bound_schema(self):
        seen=[]
        def fake(payload):
            seen.append(payload)
            return {"done":True, "done_reason":"stop",
                    "message":{"content":"{\"version\":2}"}}
        messages=[
            {"role":"system","content":"Generate strict v2"},
            {"role":"user","content":"Count binary strings of length 4 with exactly 2 ones, first two bits opposite."},
        ]
        for model in ("neral-qwen35-qlora-v1:latest","qwen3.5:9b"):
            result=make_local_extractor(model,transport=fake)(messages)
            self.assertEqual(result,'{"version":2}')
        for payload in seen:
            self.assertEqual(payload["messages"],messages)
            self.assertEqual(payload["format"]["properties"]["variables"]["required"],
                             ["B0","B1","B2","B3"])
            self.assertEqual(payload["options"]["num_predict"],1024)
            self.assertEqual(payload["options"]["repeat_penalty"],1.05)
            self.assertEqual(payload["options"]["temperature"],0)
            self.assertFalse(payload["stream"])
            self.assertFalse(payload["think"])

    def test_bad_finish_and_bad_json_envelope_rejected(self):
        turns=[{"role":"system","content":"system"},{"role":"user","content":"question"}]
        for reply in (
            {"done":False,"done_reason":"stop","message":{"content":"{}"}},
            {"done":True,"done_reason":"length","message":{"content":"{}"}},
            {"done":True,"done_reason":"stop","message":{"content":""}},
            {"done":True,"done_reason":"stop","message":{}},
            {"done":True,"done_reason":"stop","message":{"content":"A"*33000}},
            {"message":{"content":"{}"}}, {}):
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    make_local_extractor("qwen3.5:9b",transport=lambda _:reply)(turns)

    def test_invalid_model_tags_and_messages_cannot_make_calls(self):
        called=[]
        def send(payload):
            called.append(payload)
            return {"done":True,"done_reason":"stop","message":{"content":"{}"}}
        for name in ("","../hack","qwen name","qwen\nsecret","http://evil"):
            with self.subTest(name=name),self.assertRaises(ValueError):
                make_local_extractor(name,transport=send)
        model=make_local_extractor("qwen3.5:9b",transport=send)
        for messages in ([],[{"role":"user","content":"x"}],
                         [{"role":"system","content":"x"},{"role":"tool","content":"y"}]):
            with self.assertRaises(ValueError):
                model(messages)
        self.assertEqual(called,[])

    def test_not_in_dev_test_or_sealed_dataset_package_imports(self):
        from pathlib import Path
        code=Path(__file__).resolve().parents[1].joinpath(
            "neral_nl_local_ollama.py").read_text(encoding="utf8")
        self.assertNotIn("neral_blind_100",code)
        self.assertNotIn("neral_generalization_100",code)
        self.assertNotIn("neral_nl_schema_probe",code)
        self.assertNotIn("3feca67d",code)


if __name__=="__main__":
    unittest.main()
