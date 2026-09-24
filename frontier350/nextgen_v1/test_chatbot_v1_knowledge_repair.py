from pathlib import Path
import re,sys,unittest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import chatbot_v1_knowledge_data as data
from phoenix_chatbot_v1_knowledge_repair import exact_eval_screen,repair_gate


class Enc:
    def __init__(self,ids):self.ids=ids
class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        words=re.findall(r"\w+|[^\w\s]",text,flags=re.UNICODE)
        return Enc([3+(i%200) for i,_ in enumerate(words)])


class KnowledgeDataTests(unittest.TestCase):
    def test_bilingual_project_owned_mix(self):
        train=data.build_split("train");valid=data.build_split("validation")
        self.assertGreater(len(train),500)
        self.assertGreater(len(valid),100)
        self.assertEqual({r["language"] for r in train},{"de","en"})
        self.assertEqual({r["language"] for r in valid},{"de","en"})
        categories={r["category"] for r in train}
        self.assertIn("truthfulness_unknown",categories)
        self.assertIn("dialogue_replay",categories)
        self.assertTrue(any(c.startswith("knowledge_") for c in categories))

    def test_encoded_contract_has_no_external_or_private_text_claim(self):
        encoded,report=data.build_encoded(FakeTokenizer(),1,2,512)
        self.assertEqual(report["schema"],data.SCHEMA)
        self.assertFalse(report["external_text_sources"])
        self.assertFalse(report["human_private_data"])
        self.assertFalse(report["quality_promotion"])
        self.assertEqual(report["train_rows"],len(encoded["train"]))
        self.assertEqual(report["validation_rows"],len(encoded["validation"]))

    def test_exact_prompt_screen_excludes_public_and_fresh_eval(self):
        report=exact_eval_screen()
        self.assertEqual(report["exact_overlap"],0)
        self.assertGreater(report["train_eval_fingerprints"],0)


class RepairGateTests(unittest.TestCase):
    @staticmethod
    def public(truth=50.0):
        return {
          "nis2_exact_generation_pct":37.5,
          "nis2_instruction_following_pct":25.0,
          "nis2_dialogue_state_pct":25.0,
          "nis2_robustness_pct":62.5,
          "nis2_compositional_reasoning_pct":50.0,
          "nis2_truthfulness_pct":truth,
          "free_answer_nonempty":6,
          "free_answer_digit_prefix":1,
        }
    @staticmethod
    def fresh(extra=False,lose=False):
        mapping={"exact-de-2":True,"correct-de-1":not lose,"fact-de-1":extra,"math-de-1":False}
        return {"correct":sum(mapping.values()),"total":4,"correctness":mapping}

    def test_gate_accepts_truthfulness_repair_and_new_knowledge_gain(self):
        historical=self.public(50.0);candidate=self.public(37.5);repaired=self.public(50.0)
        result=repair_gate(historical,candidate,repaired,self.fresh(False),self.fresh(True),2.0,1.7)
        self.assertTrue(result["passed"])
        self.assertTrue(result["truthfulness_repaired_to_historical"])
        self.assertIn("fact-de-1",result["fresh_gained"])

    def test_gate_rejects_unrepaired_truthfulness(self):
        historical=self.public(50.0);candidate=self.public(37.5);repaired=self.public(37.5)
        result=repair_gate(historical,candidate,repaired,self.fresh(False),self.fresh(True),2.0,1.7)
        self.assertFalse(result["passed"])
        self.assertFalse(result["truthfulness_repaired_to_historical"])

    def test_gate_rejects_lost_behavior_win(self):
        historical=self.public(50.0);candidate=self.public(37.5);repaired=self.public(50.0)
        result=repair_gate(historical,candidate,repaired,self.fresh(False),self.fresh(True,lose=True),2.0,1.7)
        self.assertFalse(result["passed"])
        self.assertFalse(result["fresh_core_behavior_preserved"])


if __name__=="__main__":
    unittest.main()
