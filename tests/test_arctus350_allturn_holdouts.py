"""Every user turn, not only the first, must be disjoint across splits."""
from __future__ import annotations
import unittest
from frontier350.chat350_data import validate_prompt_holdouts


def example(key,*texts):
    if len(texts)%2 or not texts:raise ValueError("bad synthetic fixture")
    return {"id":key,"messages":[
        {"role":"user" if i%2==0 else "assistant","content":t}
        for i,t in enumerate(texts)
    ]}


class AllTurnHoldouts(unittest.TestCase):
    def test_later_user_turn_cannot_leak_into_validation(self):
        data={
          "train":[example("a","Hallo Ada","Hallo","Was ist 7+8?","15")],
          "val":[example("b","Hallo Bea","Hallo","Was ist 7+8?","15")],
          "test":[example("c","Unabhängig","Ja")],
        }
        with self.assertRaisesRegex(ValueError,"prompt leakage"):
            validate_prompt_holdouts(data)

    def test_validation_to_test_overlap_also_rejected(self):
        data={
          "train":[example("a","Train","okay")],
          "val":[example("b","Val","okay","Wie alt bist du?","Unklar")],
          "test":[example("c","Test","okay","WIE ALT BIST DU?","Keine Angabe")],
        }
        with self.assertRaisesRegex(ValueError,"prompt leakage"):
            validate_prompt_holdouts(data)

    def test_unicode_nfkc_whitespace_all_turns(self):
        data={
          "train":[example("a","Start","okay","WER  ist  ２ ?","Zwei")],
          "val":[example("b","Anderer Start","okay","wer ist 2 ?","2")],
          "test":[example("c","Unabhängig","okay")],
        }
        with self.assertRaisesRegex(ValueError,"prompt leakage"):
            validate_prompt_holdouts(data)

    def test_same_split_repetition_not_cross_split_contamination(self):
        data={
          "train":[example("a","Hallo","okay","Hallo","Nochmal")],
          "val":[example("b","Val unabh.","okay")],
          "test":[example("c","Test unabhängig.","okay")],
        }
        result=validate_prompt_holdouts(data)
        self.assertEqual(result["first_user_prompt_overlap"],0)
        self.assertEqual(result["all_user_turn_overlap"],0)

    def test_number_variants_not_mislabelled_as_exact(self):
        data={
          "train":[example("a","Berechne 7+8","15")],
          "val":[example("b","Berechne 9+8","17")],
          "test":[example("c","Was bedeutet 7?","eine Zahl")],
        }
        self.assertEqual(validate_prompt_holdouts(data)["all_user_turn_overlap"],0)


if __name__=="__main__":
    unittest.main()
