from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import chatbot_v1_data as data
from phoenix_chatbot_v1_pilot import gate,verify_training_eval_disjoint


class Enc:
    def __init__(self,ids):self.ids=ids


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        return Enc([3+(b%200) for b in text.encode("utf-8")])


class ChatbotV1DataTests(unittest.TestCase):
    def test_train_validation_are_bilingual_large_and_prompt_disjoint(self):
        train=data.build_split("train");valid=data.build_split("validation")
        self.assertGreater(len(train),500)
        self.assertGreater(len(valid),100)
        self.assertEqual({row["language"] for row in train},{"de","en"})
        self.assertEqual({row["language"] for row in valid},{"de","en"})
        train_prompts={data.norm(turn["user"]) for row in train for turn in row["turns"]}
        valid_prompts={data.norm(turn["user"]) for row in valid for turn in row["turns"]}
        self.assertFalse(train_prompts&valid_prompts)
        expected={"exact_instruction","arithmetic","multi_turn_state","multi_turn_correction",
                  "uncertainty","format_control","clarification","short_explanation",
                  "bilingual","identity"}
        self.assertEqual({row["category"] for row in train},expected)
        self.assertEqual({row["category"] for row in valid},expected)

    def test_real_multiturn_examples_exist_in_both_languages(self):
        train=data.build_split("train")
        for language in ("de","en"):
            turns=[len(row["turns"]) for row in train if row["language"]==language]
            self.assertGreaterEqual(max(turns),3)
            self.assertTrue(any(n==2 for n in turns))

    def test_encoder_masks_context_and_supervises_final_eos(self):
        row=data.conversation("x","multi_turn_state","de",[
            ("Merke dir X7.","Verstanden."),
            ("Was sollst du dir merken?","X7."),
        ])
        packed=data.encode_conversation(row,FakeTokenizer(),1,2,512)
        self.assertEqual(packed["turn_count"],2)
        self.assertEqual(len(packed["input_ids"]),len(packed["labels"]))
        self.assertEqual(packed["labels"].count(2),1)
        self.assertEqual(packed["labels"][-1],2)
        self.assertIn(-100,packed["labels"])
        self.assertGreater(packed["supervised_tokens"],2)

    def test_encoded_report_has_no_external_or_private_data_claim(self):
        encoded,report=data.build_encoded(FakeTokenizer(),1,2,512)
        self.assertEqual(report["schema"],data.DATASET_SCHEMA)
        self.assertFalse(report["external_text_sources"])
        self.assertFalse(report["human_private_data"])
        self.assertFalse(report["quality_promotion"])
        self.assertEqual(report["train_rows"],len(encoded["train"]))
        self.assertEqual(report["validation_rows"],len(encoded["validation"]))
        self.assertLessEqual(report["max_encoded_tokens"],512)
        self.assertEqual(len(report["dataset_sha256"]),64)

    def test_public_development_prompts_are_exact_hash_disjoint(self):
        report=verify_training_eval_disjoint()
        self.assertEqual(report["exact_hash_overlap"],0)
        self.assertGreater(report["public_development_hashes_checked"],0)


class ChatbotV1GateTests(unittest.TestCase):
    @staticmethod
    def baseline():
        return {
            "nis2_instruction_following_pct":25.0,
            "nis2_dialogue_state_pct":25.0,
            "nis2_exact_generation_pct":12.5,
            "nis2_truthfulness_pct":50.0,
            "nis2_robustness_pct":50.0,
            "nis2_compositional_reasoning_pct":50.0,
            "free_answer_nonempty":6,
            "free_answer_digit_prefix":1,
        }

    def test_gate_requires_real_primary_improvement(self):
        base=self.baseline();same=dict(base)
        result=gate(base,same,2.0,1.8)
        self.assertFalse(result["passed"])
        self.assertFalse(result["primary_improvement"])

    def test_gate_accepts_improvement_with_protected_dimensions(self):
        base=self.baseline();candidate=dict(base)
        candidate["nis2_instruction_following_pct"]=37.5
        candidate["nis2_dialogue_state_pct"]=37.5
        result=gate(base,candidate,2.0,1.7)
        self.assertTrue(result["passed"])

    def test_gate_rejects_knowledge_style_regression(self):
        base=self.baseline();candidate=dict(base)
        candidate["nis2_instruction_following_pct"]=37.5
        candidate["nis2_truthfulness_pct"]=25.0
        result=gate(base,candidate,2.0,1.7)
        self.assertFalse(result["passed"])
        self.assertFalse(result["protected_pair_regression_within_one_case"])


if __name__=="__main__":
    unittest.main()
