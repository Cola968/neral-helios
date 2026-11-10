"""CPU-only safety contracts for an opt-in 16K interactive Ollama profile."""
import json
import os
import unittest
from unittest.mock import patch

from neral_agent_autonomy import OllamaLocalBackend

MESSAGES = [{"role":"system","content":"You are NERAL."},
            {"role":"user","content":"Summarize this code."}]

class _FakeResponse:
    def __init__(self, packet):
        self.packet = json.dumps(packet).encode("utf-8")
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, *args): return self.packet

class _FakeOllama:
    def __init__(self):
        self.requests = []
    def open(self, req, timeout):
        self.requests.append(json.loads(req.data))
        return _FakeResponse({"done":True,"done_reason":"stop",
                              "message":{"content":"OK"}})

class LongContextProfileContract(unittest.TestCase):
    def test_no_16k_without_explicit_host_flag(self):
        with patch.dict(os.environ, {"OLLAMA_FLASH_ATTENTION": "0"}):
            with self.assertRaises(ValueError):
                OllamaLocalBackend("qwen3.5:9b",
                    inference_profile="interactive_long16k")

    def test_16k_only_for_separate_interactive_profile(self):
        fake=_FakeOllama()
        with patch.dict(os.environ, {"OLLAMA_FLASH_ATTENTION": "1"}), patch(
                "neral_agent_autonomy.local_only_opener", return_value=fake):
            long = OllamaLocalBackend("qwen3.5:9b",
                                     inference_profile="interactive_long16k")
            measured = OllamaLocalBackend("qwen3.5:9b",
                                     inference_profile="measured_nonthinking_2048")
            self.assertEqual(long.respond_chat(MESSAGES), "OK")
            self.assertEqual(measured.respond_chat(MESSAGES), "OK")
        self.assertEqual(fake.requests[0]["options"]["num_ctx"], 16384)
        self.assertEqual(fake.requests[0]["options"]["num_predict"], 2048)
        self.assertEqual(fake.requests[1]["options"]["num_ctx"], 4096)
        self.assertEqual(fake.requests[1]["options"]["num_predict"], 2048)
        self.assertFalse(fake.requests[0]["think"])
        self.assertFalse(fake.requests[1]["think"])

    def test_default_profile_retains_4096_even_when_flash_is_set(self):
        fake=_FakeOllama()
        with patch.dict(os.environ, {"OLLAMA_FLASH_ATTENTION": "1"}), patch(
                "neral_agent_autonomy.local_only_opener", return_value=fake):
            old = OllamaLocalBackend("qwen3:8b")
            self.assertEqual(old.respond_chat(MESSAGES),"OK")
        self.assertEqual(fake.requests[0]["options"]["num_ctx"],4096)
        self.assertEqual(fake.requests[0]["options"]["num_predict"],1024)

if __name__ == "__main__":
    unittest.main()
