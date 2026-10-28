"""Independent reviewer regressions: exact Arctus350 test-time prompt and metric gates."""
from __future__ import annotations

import unittest

from frontier350.chat_prompt350 import prompt_ids
from frontier350.chat350_eval_contract import (
    parse_chat_prefix, case_prefix_ids, generation_forbidden_ids,
    trim_hallucinated_user_turn,
)
from frontier350.compare_chat350_slices import compare


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class BoundaryTokenizer:
    special = {"<|bos|>": 0, "<|eos|>": 1, "<|sep|>": 2, "<|pad|>": 3}

    def token_to_id(self, name):
        return self.special.get(name)

    def encode(self, text):
        result = []
        i = 0
        while i < len(text):
            if text.startswith(": ", i):
                result.append(999)
                i += 2
            else:
                result.append(100 + ord(text[i]))
                i += 1
        return FakeEncoding(result)


class EvaluatorParityTests(unittest.TestCase):
    def test_single_turn_uses_same_segments_as_chat_runtime(self):
        tokenizer = BoundaryTokenizer()
        prefix = "User: Wie geht es dir?\nAssistant:"
        expected = prompt_ids(tokenizer, [("user", "Wie geht es dir?")])
        self.assertEqual(parse_chat_prefix(prefix), [("user", "Wie geht es dir?")])
        self.assertEqual(case_prefix_ids(tokenizer, prefix), expected)

    def test_multi_turn_bpe_boundary_is_preserved(self):
        tokenizer = BoundaryTokenizer()
        prefix = "User: Hi\nAssistant: Gut.\nUser: Und jetzt?\nAssistant:"
        expected = prompt_ids(tokenizer, [
            ("user", "Hi"), ("assistant", "Gut."), ("user", "Und jetzt?")
        ])
        self.assertEqual(case_prefix_ids(tokenizer, prefix), expected)
        self.assertNotEqual(expected, [0] + tokenizer.encode(prefix).ids)

    def test_non_chat_or_nonterminal_prefix_is_rejected(self):
        tokenizer = BoundaryTokenizer()
        for prefix in (
            "Plain prompt", "Assistant: hallo", "User: Hi",
            "User: Hi\nAssistant: Gut", "User: Hi\nAssistant:\nUser: okay",
            "User: \nAssistant:",
        ):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                case_prefix_ids(tokenizer, prefix)

    def test_bos_pad_sep_are_forbidden_but_eos_is_allowed(self):
        tokenizer = BoundaryTokenizer()
        self.assertEqual(generation_forbidden_ids(tokenizer), (0, 2, 3))
        self.assertNotIn(tokenizer.token_to_id("<|eos|>"), generation_forbidden_ids(tokenizer))

    def test_hallucinated_new_user_turn_is_not_an_assistant_answer(self):
        self.assertEqual(
            trim_hallucinated_user_turn("42\nUser: Was ist 2+2?\nAssistant: 4"),
            ("42", True),
        )
        self.assertEqual(trim_hallucinated_user_turn("Antwort mit zwei Zeilen\nZweite Zeile"),
                         ("Antwort mit zwei Zeilen\nZweite Zeile", False))


def slice_report(overall=3.0, include_detail=False, skipped=0):
    rows = {"overall": {
        "answer_nll": overall, "rows": 25,
        "answer_tokens": 1000, "skipped_too_long": skipped,
    }}
    if include_detail:
        rows["language:de"] = {"answer_nll": overall, "rows": 15,
                               "answer_tokens": 600, "skipped_too_long": 0}
    return {"tokenizer_sha256": "tokenizer", "dataset_manifest_sha256": "data",
            "split": "val", "max_seq": 512, "slices": rows}


class SliceGateIntegrityTests(unittest.TestCase):
    def test_only_overall_is_not_evidence_for_slice_no_regression(self):
        value = compare(slice_report(3.0), slice_report(2.0))
        self.assertFalse(value["slice_no_regression"])

    def test_skipped_long_sequences_are_not_silently_approved(self):
        value = compare(slice_report(3.0, True, skipped=0),
                        slice_report(2.0, True, skipped=3))
        self.assertFalse(value["slice_no_regression"])

    def test_valid_representative_slices_can_pass(self):
        value = compare(slice_report(3.0, True), slice_report(2.0, True))
        self.assertTrue(value["slice_no_regression"])


if __name__ == "__main__":
    unittest.main()
