"""CPU-only readiness aggregation over synthetic metadata, no original texts."""
import unittest
from frontier350.arctus350_training_readiness import aggregate_eligibility

class ReadinessTests(unittest.TestCase):
    def test_counts_domains_language_license_and_length(self):
        rows=[{"id":"a","language":"de","domain":"math","source":"synthetic","license":"CC0","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]},
              {"id":"b","language":"en","domain":"code","source":"human","license":"Apache-2.0","messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}]
        def encoder(row,seq):
            if row["id"]=="b" and seq==128:raise ValueError("long")
            return [42]*12
        result=aggregate_eligibility({"train":rows,"val":[rows[0]],"test":[rows[1]]},encoder)
        self.assertEqual(result["split"]["train"]["rows"],2)
        self.assertEqual(result["split"]["train"]["eligible_128"],1)
        self.assertEqual(result["split"]["train"]["eligible_256"],2)
        self.assertEqual(result["split"]["train"]["language"]["de"],1)
        self.assertEqual(result["split"]["train"]["domain"]["math"],1)
        self.assertEqual(result["split"]["train"]["license"]["CC0"],1)

    def test_fail_closed_without_train_val_test(self):
        with self.assertRaises(ValueError):
            aggregate_eligibility({"train":[],"val":[]},lambda row,seq:[1])

    def test_no_raw_dialogue_or_record_ids_exported(self):
        row={"id":"PRIVATE_ID","language":"de","domain":"general","source":"local","license":"CC0",
             "messages":[{"role":"user","content":"PRIVATE_QUESTION"},{"role":"assistant","content":"PRIVATE_REPLY"}]}
        result=aggregate_eligibility({s:[row] for s in ("train","val","test")},lambda row,seq:[1,2,3])
        import json
        txt=json.dumps(result)
        for token in ("PRIVATE_ID","PRIVATE_QUESTION","PRIVATE_REPLY"):
            self.assertNotIn(token,txt)

    def test_unexpected_encoder_exception_not_silently_swallowed(self):
        row={"id":"x","language":"de","domain":"general","source":"human","license":"CC0","messages":[]}
        def crash(row,seq):raise RuntimeError("broken encoder")
        with self.assertRaises(RuntimeError):
            aggregate_eligibility({s:[row] for s in ("train","val","test")},crash)

if __name__=="__main__":
    unittest.main()
