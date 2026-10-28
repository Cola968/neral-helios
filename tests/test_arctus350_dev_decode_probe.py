"""CPU-only guarantees for the read-only development decoding A/B."""
from __future__ import annotations
import unittest
from frontier350.arctus350_dev_decode_probe import (
    forbidden_source_ngrams, disallows_source_trigram, normalize_text, echo_flags,
    validate_case_selector,
)


class DecodeProbeTests(unittest.TestCase):
    def test_repetition_guard_only_blocks_completed_source_trigram(self):
        grams=forbidden_source_ngrams([10,11,12,13])
        self.assertTrue(disallows_source_trigram([10,11],12,grams))
        self.assertTrue(disallows_source_trigram([11,12],13,grams))
        self.assertFalse(disallows_source_trigram([10,11],14,grams))
        self.assertFalse(disallows_source_trigram([11],12,grams))
        self.assertFalse(disallows_source_trigram([],12,grams))
        self.assertEqual(len(grams),2)

    def test_source_repetition_ignores_short_contexts(self):
        self.assertEqual(forbidden_source_ngrams([1,2]),set())

    def test_unicode_text_echo_and_prefix(self):
        p="  GRÜßE   aus BERLIN  "
        self.assertEqual(normalize_text(p),"grüsse aus berlin")
        self.assertEqual(echo_flags(p,"grüße aus berlin"),{"exact":True,"prefix":True})
        self.assertEqual(echo_flags("Hallo Welt","Hallo Welten"),{"exact":False,"prefix":False})
        self.assertEqual(echo_flags(p,"Hallo!"),{"exact":False,"prefix":False})

    def test_select_only_approved_development_ids(self):
        wanted=("de_chat_01","en_chat_01")
        self.assertEqual(validate_case_selector(wanted,("de_chat_01","en_chat_01","frozen")),
                         wanted)
        with self.assertRaises(ValueError):
            validate_case_selector(("missing",),("de_chat_01",))
        with self.assertRaises(ValueError):
            validate_case_selector(("de_chat_01","de_chat_01"),("de_chat_01",))
        with self.assertRaises(ValueError):
            validate_case_selector(tuple(range(7)),tuple(range(7)))


if __name__=="__main__":
    unittest.main()
