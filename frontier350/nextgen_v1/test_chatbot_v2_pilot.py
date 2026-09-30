from pathlib import Path
import sys,unittest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

from phoenix_chatbot_v2_pilot import gate

class ChatbotV2GateTests(unittest.TestCase):
    @staticmethod
    def public():
        return {
            "nis2_pair_overall_pct":40.0,
            "nis2_exact_generation_pct":10.0,
            "nis2_dialogue_state_pct":50.0,
            "nis2_truthfulness_pct":37.5,
            "nis2_robustness_pct":62.5,
            "nis2_compositional_reasoning_pct":50.0,
            "free_answer_nonempty":8,
            "free_answer_digit_prefix":0,
        }
    @staticmethod
    def fresh():
        return {"correct":4,"total":24,"accuracy_pct":100/6,
                "categories":{"exact":25.0,"format":0.0,"dialogue":50.0,"concise":50.0,
                              "truthfulness":25.0,"knowledge":0.0,"math":0.0,"bilingual":0.0}}

    def test_rejects_fresh_dialogue_regression_even_with_public_gain(self):
        b=self.public();c=dict(b);c["nis2_exact_generation_pct"]=35.0
        fb=self.fresh();fc={"correct":4,"total":24,"accuracy_pct":fb["accuracy_pct"],
                            "categories":dict(fb["categories"])}
        fc["categories"]["dialogue"]=25.0
        g=gate(b,c,fb,fc,2.0,1.8)
        self.assertFalse(g["passed"])
        self.assertFalse(g["fresh_categories_protected"])

    def test_rejects_concise_regression(self):
        b=self.public();c=dict(b);c["nis2_pair_overall_pct"]=45.0
        fb=self.fresh();fc={"correct":4,"total":24,"accuracy_pct":fb["accuracy_pct"],
                            "categories":dict(fb["categories"])}
        fc["categories"]["concise"]=0.0
        g=gate(b,c,fb,fc,2.0,1.8)
        self.assertFalse(g["passed"])

    def test_accepts_fresh_generalization_gain_with_protection(self):
        b=self.public();c=dict(b);c["nis2_exact_generation_pct"]=22.5
        fb=self.fresh();fc={"correct":5,"total":24,"accuracy_pct":100*5/24,
                            "categories":dict(fb["categories"])}
        fc["categories"]["dialogue"]=75.0
        g=gate(b,c,fb,fc,2.0,1.7)
        self.assertTrue(g["passed"])
        self.assertTrue(g["fresh_behavior_gain"])

if __name__=="__main__":
    unittest.main()
