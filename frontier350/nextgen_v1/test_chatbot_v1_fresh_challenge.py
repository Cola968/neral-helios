from pathlib import Path
import sys,unittest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import chatbot_v1_fresh_challenge as fresh

class FreshChallengeTests(unittest.TestCase):
    def test_case_contract_and_balance(self):
        self.assertEqual(len(fresh.CASES),24)
        ids=[row[0] for row in fresh.CASES]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual({row[2] for row in fresh.CASES},{"de","en"})
        counts={}
        for _ident,category,_lang,_prompt,_scorer in fresh.CASES:
            counts[category]=counts.get(category,0)+1
        self.assertEqual(counts["truthfulness"],4)
        self.assertGreaterEqual(counts["dialogue"],4)
        self.assertEqual(counts["knowledge"],4)
        self.assertEqual(counts["math"],2)

    def test_challenge_is_curriculum_disjoint(self):
        report=fresh.verify_disjoint()
        self.assertEqual(report["curriculum_overlap"],0)
        self.assertEqual(report["cases"],len(fresh.CASES))
        self.assertEqual(len(report["challenge_sha256"]),64)

    def test_objective_scorers_have_expected_positive_and_negative_examples(self):
        by_id={ident:scorer for ident,_c,_l,_p,scorer in fresh.CASES}
        self.assertTrue(by_id["exact_de_1"]("NEBELFELD"))
        self.assertFalse(by_id["exact_de_1"]("NEBELFELD ist die Antwort."))
        self.assertTrue(by_id["memory_en"]("LANTERN-58"))
        self.assertTrue(by_id["unknown_en_1"]("I cannot know that because it is fictional."))
        self.assertFalse(by_id["unknown_en_1"]("The population was 12345."))
        self.assertTrue(by_id["math_en"]("56"))
        self.assertFalse(by_id["math_en"]("42"))

if __name__=="__main__":
    unittest.main()
