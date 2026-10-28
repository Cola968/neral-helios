"""Contract-only synthetic validation readout. No private text leaves the GPU PC."""
import unittest
from frontier350.arctus350_unseen_task_eval import (
    validate_eval_partition, record_aggregate, improvement_gate
)

class UnseenTaskEvalContracts(unittest.TestCase):
    def test_only_validated_synthetic_val_not_train_or_test(self):
        x={"train":[{"id":"TRAIN"}],"val":[{"id":"VALID"}],"test":[{"id":"TEST"}]}
        self.assertEqual(validate_eval_partition(x),[{"id":"VALID"}])
        with self.assertRaises(ValueError):
            validate_eval_partition({"val":[{"id":"VALID"}]})
        with self.assertRaises(ValueError):
            validate_eval_partition({"train":[],"val":[],"test":[]})

    def test_families_language_and_eos_aggregate_no_text_leak(self):
        sensitive=[
          {"id":"SECRET1","domain":"format","language":"de","expected":"STERN","generated":"STERN","eos":True,"tokens":[1,2,1]},
          {"id":"SECRET2","domain":"format","language":"en","expected":"GALAXY","generated":"GALAXY and more","eos":False,"tokens":[2,3,2]},
          {"id":"SECRET3","domain":"math","language":"de","expected":"15","generated":"15","eos":True,"tokens":[4,5,6]},
          {"id":"SECRET4","domain":"math","language":"en","expected":"10","generated":"11","eos":True,"tokens":[7,8,9]},
        ]
        out=record_aggregate(sensitive)
        self.assertEqual(out["total"],4)
        self.assertEqual(out["exact_correct"],2)
        self.assertEqual(out["eos_count"],3)
        self.assertEqual(out["by_family_language"]["format/de"]["exact_correct"],1)
        self.assertEqual(out["by_family_language"]["math/en"]["exact_correct"],0)
        import json
        rendered=json.dumps(out)
        for value in ("SECRET","GALAXY","STERN","and more"):
            self.assertNotIn(value,rendered)

    def test_parsing_exactness_no_normalized_cheating(self):
        rows=[{"id":"ID","domain":"math","language":"en","expected":"15","generated":" 15 ","eos":True,"tokens":[1]},
              {"id":"ID2","domain":"math","language":"de","expected":"15","generated":"15. ","eos":True,"tokens":[1]}]
        r=record_aggregate(rows)
        self.assertEqual(r["exact_correct"],1)

    def test_improvement_requires_real_exactness_and_no_health_regression(self):
        baseline={"exact_correct":0,"total":76,"eos_count":70,"repeated_trigram_fraction":.02}
        good={"exact_correct":8,"total":76,"eos_count":71,"repeated_trigram_fraction":.02}
        bad={"exact_correct":10,"total":76,"eos_count":68,"repeated_trigram_fraction":.06}
        self.assertTrue(improvement_gate(baseline,good))
        self.assertFalse(improvement_gate(baseline,bad))
        self.assertFalse(improvement_gate(baseline,{**good,"exact_correct":0}))
        with self.assertRaises(ValueError):
            improvement_gate(baseline,{**good,"total":77})

if __name__=="__main__":
    unittest.main()
