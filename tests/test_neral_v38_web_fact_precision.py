"""Regression cases based on erroneous web fact extraction in real GPU receipts."""
import unittest
from neral_v32_verifier import _property_evidence

class WebPropertyPrecisionTests(unittest.TestCase):
    def check(self, entity, prop, text, wanted):
        result = _property_evidence(text, entity, prop)
        self.assertIsNotNone(result, (entity, text))
        self.assertEqual(result[0], wanted)

    def test_engine_infobox_and_browser_clauses(self):
        self.check("Chrome", "engine",
                   "| Engines | Blink (WebKit on iOS), V8 JavaScript engine\nGoogle Chrome is a web browser.", "Blink")
        self.check("Edge", "engine",
                   "Microsoft Edge is a Chromium-based browser with Blink and V8 engines.", "Blink")
        self.check("Safari", "engine",
                   "Safari was based on WebKit, Apple's fork of the KHTML browser engine.", "WebKit")
        self.check("Firefox", "engine",
                   "Firefox uses the Gecko rendering engine to display pages.", "Gecko")

    def test_creators_are_attributed_to_named_language(self):
        self.check("Python", "creator",
                   "Python was conceived by Guido van Rossum in the Netherlands.", "Guido van Rossum")
        self.check("Rust", "creator",
                   "Software developer Graydon Hoare created Rust in 2006. "
                   "NIL is cited as an influence for Rust; NIL was developed by Robert Strom.", "Graydon Hoare")
        self.check("Ruby", "creator",
                   "Ruby is a language. Development of the language began in 1995 by Yukihiro \"Matz\" Matsumoto in Japan. "
                   "Ruby's specifications were developed by Open Standards Promotion Center.", "Yukihiro Matsumoto")
        self.check("JavaScript", "creator",
                   "Created by Brendan Eich in 1995, JavaScript is maintained by Ecma International.", "Brendan Eich")

    def test_reject_unrelated_creator_and_engine_pseudofacts(self):
        self.assertIsNone(_property_evidence(
            "NIL is an influence for Rust; NIL was developed by Robert Strom.", "Rust", "creator"))
        self.assertIsNone(_property_evidence(
            "Microsoft uses proprietary URL handlers to redirect to Edge.", "Edge", "engine"))
        self.assertIsNone(_property_evidence(
            "The Chrome rendering engine passed a test in 2011.", "Chrome", "engine"))
        self.assertIsNone(_property_evidence(
            "FreeBSD forks use microkernel features. DragonFlyBSD is a fork of FreeBSD.",
            "FreeBSD", "kernel"))

if __name__ == "__main__":
    unittest.main()
