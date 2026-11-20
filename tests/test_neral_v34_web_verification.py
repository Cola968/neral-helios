"""Fail-closed, source-first regression tests for V3.4 web extraction."""
import json
import unittest

from neral_page_reader import MainText, evidence_sentence_windows
from neral_v32_verifier import (
    property_grounded_web_answer, _property_evidence, verify_web_fact_proposal,
)

class SentenceWindowTests(unittest.TestCase):
    def test_returns_complete_current_fact_and_neighbors_not_truncated_prefix(self):
        text=("Chrome was initially based on WebKit in 2008. "
              "Chrome now uses the Blink rendering engine. "
              "Chrome receives frequent security updates.")
        windows=evidence_sentence_windows(text,"Chrome","rendering engine")
        self.assertTrue(any("Chrome now uses the Blink rendering engine." == w["sentence"]
                            for w in windows))
        self.assertTrue(all("sentence" in w and "context" in w for w in windows))
        self.assertTrue(any("security updates" in w["context"] for w in windows))

    def test_does_not_accept_truncated_name_as_full_sentence(self):
        text="Python, which was initially developed by Guido van Ros"
        self.assertEqual(evidence_sentence_windows(text,"Python","created by"),[])
        self.assertIsNone(_property_evidence(text,"Python","creator"))
        self.assertIsNone(verify_web_fact_proposal(
            {"entity":"Python","fact":"Guido van Ros",
             "evidence_sentence":text},text,"Python","creator"))

    def test_infobox_and_intro_visible_before_historical_body(self):
        html=("""<html><body><main><table class="infobox">
        <tr><th>Rendering engine</th><td>Blink</td></tr></table>
        <p>Chrome is a web browser. Chrome currently uses the Blink rendering engine.</p>
        <p>Chrome initially used the WebKit rendering engine.</p>""" +
        "".join("<p>Historical browser timeline and JavaScript standards info. </p>" for _ in range(90))+
        "</main></body></html>")
        parser=MainText();parser.feed(html);parser.close()
        text=parser.text("Chrome browser rendering engine")
        self.assertIn("Blink",text)
        self.assertIn("Chrome currently uses the Blink rendering engine.",text)

class ValueVerifierTests(unittest.TestCase):
    def test_rejects_prefixes_placeholder_categories_and_unrelated_names(self):
        page="Python was created by Guido van Rossum in 1991. It is a programming language."
        for fact in ("Guido van Ros","same","JavaScript","Open Standards",
                     "Guido van Rossum in 1991"):
            with self.subTest(fact=fact):
                proposal={"entity":"Python","fact":fact,
                          "evidence_sentence":"Python was created by Guido van Rossum in 1991."}
                self.assertIsNone(verify_web_fact_proposal(
                    proposal,page,"Python","creator"))
        good={"entity":"Python","fact":"Guido van Rossum",
              "evidence_sentence":"Python was created by Guido van Rossum in 1991."}
        self.assertIsNotNone(verify_web_fact_proposal(good,page,"Python","creator"))
        self.assertIsNone(verify_web_fact_proposal(
            {**good,"evidence_sentence":"Python is developed by Guido van Rossum."},
            page,"Python","creator"))

    def test_current_engine_overrules_historical_only_if_explicitly_supported(self):
        page=("Chrome initially used the WebKit rendering engine. "
              "In 2013 Chrome switched to Blink. "
              "Today Chrome uses the Blink rendering engine.")
        result=_property_evidence(page,"Chrome","engine")
        self.assertIsNotNone(result)
        self.assertEqual(result[0],"Blink")
        only_history="Chrome initially used the WebKit rendering engine."
        self.assertIsNone(_property_evidence(only_history,"Chrome","engine"))

    def test_subject_in_different_sentence_must_be_unambiguous(self):
        page=("Rust is a programming language. Rust was created by Graydon Hoare. "
              "Another language was developed by Robert Strom.")
        v=_property_evidence(page,"Rust","creator")
        self.assertIsNotNone(v)
        self.assertEqual(v[0],"Graydon Hoare")
        ambiguous="Rust is a programming language. Robert Strom is a software engineer."
        self.assertIsNone(_property_evidence(ambiguous,"Rust","creator"))

    def test_same_fact_for_two_entities_must_appear_twice_verifiably(self):
        sources=[
            {"url":"https://example.net/a","page_read":True,
             "page_text":"Orion currently uses the Blink rendering engine."},
            {"url":"https://example.net/b","page_read":True,
             "page_text":"Lyra currently uses the Blink rendering engine."},
        ]
        response=property_grounded_web_answer(("Orion","Lyra"),sources,"Browser-Engine")
        self.assertEqual(response["text"].splitlines()[:2],["Orion: Blink","Lyra: Blink"])
        self.assertTrue(response["tool_verified"])
        sources[1]["page_text"]="Lyra uses the same engine."
        self.assertIsNone(property_grounded_web_answer(("Orion","Lyra"),sources,"Browser-Engine"))

    def test_one_page_read_cannot_forge_second_entity(self):
        sources=[
            {"url":"https://example.org/one","page_read":True,
             "page_text":"Orion currently uses the Blink rendering engine."},
            {"url":"https://example.org/two","page_read":False,
             "page_text":"Lyra currently uses the Blink rendering engine."},
        ]
        self.assertIsNone(property_grounded_web_answer(("Orion","Lyra"),sources,"Browser-Engine"))


if __name__=="__main__":
    unittest.main()
