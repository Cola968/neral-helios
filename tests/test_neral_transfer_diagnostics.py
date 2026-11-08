"""CPU contracts for six-arm error audit and retention/transfer pilot gate."""
import unittest
from fractions import Fraction
from scripts.neral_transfer_diagnostics import diagnose_sixarm,pilot_gate

ARMS=("qwen_zero","v5_zero","v61_zero","qwen_three","v5_three","v61_three")

def cases():
    return [dict(id=f"{d}-{i:02}",domain=d,family=f"{d}_family_{i//6}",
                 kind="integer",answer=str(10+i%6),prompt=f"private_{d}_{i}")
            for d in ("math","reasoning","code_trace") for i in range(24)]

def outputs(cases):
    return {a:[dict(id=c["id"],text=c["answer"],complete=True,new_tokens=2)
               for c in cases] for a in ARMS}

def parse(case,text):
    try:
        return Fraction(int(text))
    except (TypeError,ValueError):
        return None

def score_rows(name,n):
    return [dict(id=f"{name}_{i:03}",family=f"{name}_family_{i%8}",
                 baseline_correct=True,candidate_correct=True) for i in range(n)]

class FailureAuditTests(unittest.TestCase):
    def test_separates_wrong_unreadable_incomplete_and_paired_flips(self):
        cs=cases()
        rows=outputs(cs)
        rows["v61_zero"][0]["text"]="999"
        rows["v61_zero"][1]["text"]="not a number"
        rows["v61_zero"][2]["complete"]=False
        rows["v5_zero"][0]["text"]="999"
        report=diagnose_sixarm(cs,rows,parse)
        self.assertEqual(report["arms"]["v61_zero"]["correct"],69)
        self.assertEqual(report["arms"]["v61_zero"]["wrong_readable"],1)
        self.assertEqual(report["arms"]["v61_zero"]["unreadable"],1)
        self.assertEqual(report["arms"]["v61_zero"]["incomplete"],1)
        self.assertEqual(report["paired"]["zero"]["v61_vs_v5"]["improved"],0)
        self.assertEqual(report["paired"]["zero"]["v61_vs_v5"]["regressed"],2)
        self.assertEqual(report["families"]["math"]["math_family_0"]["v61_zero"]["wrong_readable"],1)
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["release_validated"])
        self.assertNotIn("private_math_0",str(report))
        self.assertNotIn("not a number",str(report))

    def test_rejects_missing_arm_and_reordered_responses(self):
        cs=cases()
        rows=outputs(cs)
        with self.assertRaises(ValueError):
            diagnose_sixarm(cs,{k:v for k,v in rows.items() if k!="v61_three"},parse)
        rows["qwen_zero"][0],rows["qwen_zero"][1]=rows["qwen_zero"][1],rows["qwen_zero"][0]
        with self.assertRaises(ValueError):
            diagnose_sixarm(cs,rows,parse)

    def test_rejects_wrong_domain_or_noncanonical_gold_and_invalid_complete(self):
        cs=cases()
        rows=outputs(cs)
        cs[0]["domain"]="unrecognized"
        with self.assertRaises(ValueError):
            diagnose_sixarm(cs,rows,parse)
        cs=cases()
        rows=outputs(cs)
        cs[0]["answer"]=True
        with self.assertRaises(ValueError):
            diagnose_sixarm(cs,rows,parse)
        cs=cases()
        rows=outputs(cs)
        rows["v5_zero"][0]["complete"]="yes"
        with self.assertRaises(ValueError):
            diagnose_sixarm(cs,rows,parse)

class PilotGateTests(unittest.TestCase):
    def test_identical_models_cannot_pass(self):
        report=pilot_gate(score_rows("ret",164),score_rows("trans",64))
        self.assertFalse(report["eligible_for_review"])
        self.assertFalse(report["adapter_promoted"])
        self.assertFalse(report["release_validated"])

    def test_strong_transfer_without_retention_loss_only_allows_review(self):
        ret=score_rows("ret",164)
        trans=score_rows("trans",64)
        for r in trans[:12]:
            r["baseline_correct"]=False
        result=pilot_gate(ret,trans)
        self.assertTrue(result["eligible_for_review"])
        self.assertEqual(result["transfer"]["newly_solved"],12)
        self.assertEqual(result["transfer"]["regressed"],0)
        self.assertFalse(result["release_validated"])

    def test_blocks_forgetting_or_ambiguous_transfer(self):
        ret=score_rows("ret",164)
        trans=score_rows("trans",64)
        for r in trans[:12]:
            r["baseline_correct"]=False
        for r in ret[:5]:
            r["candidate_correct"]=False
        result=pilot_gate(ret,trans)
        self.assertFalse(result["eligible_for_review"])
        self.assertIn("retention_net_loss",result["failed_checks"])
        ret=score_rows("ret",164)
        trans=score_rows("trans",64)
        for r in trans[:12]:
            r["baseline_correct"]=False
        for r in trans[12:18]:
            r["candidate_correct"]=False
        self.assertFalse(pilot_gate(ret,trans)["eligible_for_review"])

    def test_fails_closed_on_overlap_missing_rows_or_non_boolean_scores(self):
        ret=score_rows("ret",164)
        trans=score_rows("trans",64)
        trans[0]["id"]=ret[0]["id"]
        with self.assertRaises(ValueError):
            pilot_gate(ret,trans)
        with self.assertRaises(ValueError):
            pilot_gate(ret,score_rows("trans",63))
        trans=score_rows("trans",64)
        trans[0]["candidate_correct"]=1
        with self.assertRaises(ValueError):
            pilot_gate(ret,trans)

if __name__=="__main__":
    unittest.main()
