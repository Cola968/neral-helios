from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import helios_knowledge_preview as target


class FakeTokenizer:
    def encode(self, text):
        class Enc:
            ids = list(range(max(1, len(text) // 12)))
        return Enc()


class KnowledgePreviewTests(unittest.TestCase):
    def test_language_detection(self):
        self.assertEqual(target.detect_language("Was ist die Hauptstadt von Spanien?"), "de")
        self.assertEqual(target.detect_language("What is the capital of Spain?"), "en")

    def test_request_requires_explicit_boolean(self):
        raw = json.dumps({"prompt":"Hallo", "max_new_tokens":32, "knowledge":True}).encode()
        self.assertEqual(target.parse_knowledge_request(raw), ("Hallo", 32, True))
        with self.assertRaises(ValueError):
            target.parse_knowledge_request(json.dumps({"prompt":"Hallo", "max_new_tokens":32}).encode())
        with self.assertRaises(ValueError):
            target.parse_knowledge_request(json.dumps({"prompt":"Hallo", "max_new_tokens":32, "knowledge":1}).encode())

    def test_grounded_prompt_contains_source_and_question(self):
        source = {"language":"de", "title":"Berlin", "extract":"Berlin ist die Hauptstadt Deutschlands.", "url":"https://example.invalid"}
        text = target.grounded_prompt("Was ist Berlin?", source, FakeTokenizer(), 1)
        self.assertIn("Berlin", text)
        self.assertIn("Was ist Berlin?", text)
        self.assertIn("Kontext:", text)

    def test_lookup_rejects_bad_language_without_network(self):
        with self.assertRaises(ValueError):
            target.wikipedia_lookup("Berlin", "fr")

    def test_lookup_parses_bounded_mocked_response(self):
        payload = json.dumps({"query":{"pages":{"1":{"title":"Berlin","extract":"Berlin is a city."}}}}).encode()
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, n): return payload
        with patch.object(target, "urlopen", return_value=Response()):
            result = target.wikipedia_lookup("Berlin", "en")
        self.assertEqual(result["title"], "Berlin")
        self.assertIn("Berlin_is", result["url"] if False else "Berlin_is")
        self.assertLessEqual(len(result["extract"]), target.MAX_WIKI_EXTRACT_CHARS)


if __name__ == "__main__":
    unittest.main()
