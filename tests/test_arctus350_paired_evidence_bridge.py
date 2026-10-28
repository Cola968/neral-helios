"""RED: bridge real v3 carry run observations to privacy-minimal paired evidence.

Fixtures use synthetic *labels*, never actual Arctus GPU outputs.
"""
import json
import unittest
from frontier350.arctus350_carry_curriculum_v3 import _row
from frontier350.arctus350_paired_evidence import (
    make_v3_carry_run, compare_native_runs, EvidenceError,
)


class CarryV3NativeObservationsBridgeTests(unittest.TestCase):
    def observations(self,miss=False):
        cases=[
            _row(57,68,"de","direct"), _row(59,65,"de","direct"),
            _row(51,62,"en","direct"), _row(52,63,"en","direct"),
            _row(58,67,"de","trace"), _row(56,78,"de","trace"),
            _row(53,64,"en","trace"), _row(54,65,"en","trace"),
        ]
        items=[]
        for i,row in enumerate(cases):
            text=row["messages"][1]["content"]
            if miss and i==7:text="incomplete"
            items.append({"row":row,"text":text,"eos":i!=6,"tokens":[1,2,3]})
        return items

    def test_bridge_builds_strict_paired_native_schema_without_text_leak(self):
        b=make_v3_carry_run(self.observations(miss=True),
                            checkpoint_sha256="a"*64,
                            sampling={"temperature":0,"top_k":0,"max_new_tokens":96})
        c=make_v3_carry_run(self.observations(miss=False),
                            checkpoint_sha256="b"*64,
                            sampling={"temperature":0,"top_k":0,"max_new_tokens":96})
        self.assertEqual(len(b["cases"]),8)
        self.assertEqual(b["benchmark_sha256"],c["benchmark_sha256"])
        self.assertEqual(b["prompt_template_sha256"],c["prompt_template_sha256"])
        self.assertEqual(b["benchmark_id"],"arctus350-v3-val-development-only")
        self.assertFalse(b["cases"][6]["correct"],"exact text without EOS is incomplete")
        self.assertFalse(b["cases"][7]["correct"],"wrong text must fail")
        self.assertTrue(c["cases"][7]["correct"])
        self.assertFalse(c["cases"][6]["correct"])
        report=compare_native_runs(b,c)
        self.assertEqual(report["decision"],"REVIEW_CANDIDATE")
        self.assertEqual(report["overall"]["gained"],1)
        self.assertEqual(report["release_authorization"],"none")
        self.assertNotIn("Einer:",json.dumps(b,ensure_ascii=False))
        self.assertNotIn("incomplete",json.dumps(b,ensure_ascii=False))

    def test_bridge_rejects_duplicate_ids_and_forged_source_answers(self):
        items=self.observations()
        with self.assertRaises(EvidenceError):
            make_v3_carry_run(items+[items[0]],
                              checkpoint_sha256="a"*64,
                              sampling={"temperature":0,"top_k":0,"max_new_tokens":96})
        tampered=self.observations()
        tampered[0]["row"]["messages"][1]["content"]="999"
        with self.assertRaises(EvidenceError):
            make_v3_carry_run(tampered,checkpoint_sha256="a"*64,
                              sampling={"temperature":0,"top_k":0,"max_new_tokens":96})

    def test_bridge_cannot_claim_other_model_or_invent_native_weights(self):
        items=self.observations()
        for digest in ("0x01","",None):
            with self.subTest(digest=digest),self.assertRaises(EvidenceError):
                make_v3_carry_run(items,checkpoint_sha256=digest,
                                  sampling={"temperature":0,"top_k":0,"max_new_tokens":96})


if __name__=="__main__":
    unittest.main()
