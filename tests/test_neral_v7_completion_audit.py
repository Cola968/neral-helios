"""CPU tests: distinguish token-budget truncation from strict EOS-only scoring."""
import unittest
import tempfile
import json
from pathlib import Path
from scripts import neral_v7_logic_compare as source
from scripts.neral_v7_completion_audit import main
from scripts.neral_v7_completion_audit import diagnostic_report
from scripts.neral_v7_logic_pilot import evaluation_cases

def rows(cases):
    return [{
        "id":c["id"],"text":"FINAL_ANSWER: "+c["answer"],
        "complete":True,"new_tokens":10,
    } for c in cases]

class OutputAuditTests(unittest.TestCase):
    def test_reports_capped_incomplete_and_visible_answers_as_diagnostics_not_scores(self):
        cases=evaluation_cases()
        arms={a:rows(cases) for a in ("qwen","v5","candidate")}
        arms["qwen"][0].update(text="I reasoned a lot",complete=False,new_tokens=160)
        arms["qwen"][1].update(text="FINAL_ANSWER: "+cases[1]["answer"],
                                complete=False,new_tokens=160)
        arms["qwen"][2].update(text="FINAL_ANSWER: 999999",
                                complete=False,new_tokens=160)
        arms["v5"][0].update(text="x",complete=False,new_tokens=14)
        r=diagnostic_report(cases,arms)
        self.assertEqual(r["arms"]["qwen"]["token_cap_incomplete"],3)
        self.assertEqual(r["arms"]["qwen"]["parseable_incomplete"],2)
        self.assertEqual(r["arms"]["qwen"]["possibly_correct_but_incomplete"],1)
        self.assertEqual(r["arms"]["qwen"]["strict_correct"],69)
        self.assertEqual(r["arms"]["v5"]["noncap_incomplete"],1)
        self.assertFalse(r["quality_improvement_proven"])
        self.assertFalse(r["adapter_promoted"])
        self.assertNotIn(cases[0]["prompt"],str(r))
        self.assertNotIn(cases[0]["id"],str(r))
        self.assertNotIn("I reasoned a lot",str(r))

    def test_rejects_malformed_ids_missing_arms_and_unsafe_values(self):
        cases=evaluation_cases()
        original={a:rows(cases) for a in ("qwen","v5","candidate")}
        with self.assertRaises(ValueError):
            diagnostic_report(cases,{"qwen":rows(cases)})
        altered={a:[dict(x) for x in v] for a,v in original.items()}
        altered["qwen"][0]["id"]="fake"
        with self.assertRaises(ValueError):
            diagnostic_report(cases,altered)
        altered={a:[dict(x) for x in v] for a,v in original.items()}
        altered["qwen"][0]["new_tokens"]=True
        with self.assertRaises(ValueError):
            diagnostic_report(cases,altered)
        altered={a:[dict(x) for x in v] for a,v in original.items()}
        altered["qwen"][0]["complete"]=1
        with self.assertRaises(ValueError):
            diagnostic_report(cases,altered)

    def test_all_not_completed_with_short_token_count_flags_possible_eos_mismatch(self):
        cases=evaluation_cases()
        arms={a:rows(cases) for a in ("qwen","v5","candidate")}
        for r in arms["qwen"]:
            r.update(complete=False,new_tokens=1)
        audit=diagnostic_report(cases,arms)
        self.assertEqual(audit["arms"]["qwen"]["noncap_incomplete"],72)
        self.assertEqual(audit["arms"]["qwen"]["token_cap_incomplete"],0)
        self.assertTrue(audit["arms"]["qwen"]["possible_stop_token_mismatch"])
        self.assertEqual(audit["arms"]["qwen"]["strict_correct"],0)
        self.assertEqual(audit["arms"]["qwen"]["possibly_correct_but_incomplete"],72)

    def test_scalar_only_artifact_can_be_written_to_ephemeral_repo_checkout(self):
        # Windows GitHub Actions uses a temporary, fresh aggregate-only folder
        # under the checkout for artifact upload. This should be allowed.
        with tempfile.TemporaryDirectory() as private, tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1]
        ) as checkout_output:
            run=Path(private)/"old"
            source.prepare_run(run)
            cases=source._cases_folder(run)
            for arm in ("qwen","v5","candidate"):
                with (run/(arm+".jsonl")).open("x",encoding="utf8") as f:
                    for row in rows(cases):
                        f.write(json.dumps(row,ensure_ascii=False)+"\n")
            path=Path(checkout_output)/"summary.json"
            main(["--run-dir",str(run),"--out",str(path)])
            report=json.loads(path.read_text("utf8"))
            self.assertEqual(report["status"],"NERAL_V7_INCOMPLETE_OUTPUT_DIAGNOSTIC_ONLY")
            self.assertFalse(report["raw_responses_exported"])
            self.assertNotIn(cases[0]["id"],path.read_text("utf8"))

    def test_aggregate_by_role_retention_and_transfer(self):
        cases=evaluation_cases()
        arms={a:rows(cases) for a in ("qwen","v5","candidate")}
        for row in arms["candidate"][24:]:
            row.update(text="no final",complete=False,new_tokens=160)
        report=diagnostic_report(cases,arms)
        self.assertEqual(report["slices"]["transfer"]["candidate"]["token_cap_incomplete"],48)
        self.assertEqual(report["slices"]["retention"]["candidate"]["strict_correct"],24)
        self.assertEqual(report["slices"]["transfer"]["candidate"]["strict_correct"],0)

if __name__=="__main__":
    unittest.main()
