"""CPU-only, private-log truncation audit regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.neral_foundation_output_audit import audit_rows, main, AuditError

def rows():
    cases=[]
    for domain in ("math","reasoning","code_trace"):
        for fam in range(4):
            for i in range(6):
                cid=f"{domain}-family{fam}-{i:02d}"
                cases.append((cid,domain,f"family{fam}",i+50))
    result=[]
    for model in ("qwen3:8b","qwen3.5:9b"):
        for idx,(cid,domain,family,answer) in enumerate(cases):
            cut=(idx<31 if model=="qwen3:8b" else idx<62)
            raw="Partial math. Still working on" if cut else f"Done.\nFINAL_ANSWER: {answer}"
            result.append({
                "id":cid,"domain":domain,"family":family,
                "prompt_private":f"private problem {cid}","answer_private":answer,
                "model":model,"raw_private":raw,
                "grade":({"status":"incomplete","value":None,"format_valid":False}
                        if cut else {"status":"correct","value":answer,"format_valid":True}),
                "done_reason":"length" if cut else "stop"
            })
    return result

def realistic_rows():
    source=rows()
    for model in ("qwen3:8b","qwen3.5:9b"):
        finished=[r for r in source if r["model"]==model and r["done_reason"]=="stop"]
        wrong=11 if model=="qwen3:8b" else 2
        for r in finished[:wrong]:
            answer=r["answer_private"]+1
            r["raw_private"]=f"Wrong.\nFINAL_ANSWER: {answer}"
            r["grade"]={"status":"wrong_final","value":answer,"format_valid":True}
        if model=="qwen3:8b":
            r=finished[wrong]
            r["raw_private"]="An inconclusive answer."
            r["grade"]={"status":"ungraded","value":None,"format_valid":False}
    return source

class FoundationAuditTests(unittest.TestCase):
    def test_counts_truncation_and_does_not_claim_correct_reasoning(self):
        result,private=audit_rows(rows(),expected_counts=None)
        self.assertEqual(result["models"]["qwen3:8b"]["token_cap_incomplete"],31)
        self.assertEqual(result["models"]["qwen3.5:9b"]["token_cap_incomplete"],62)
        self.assertEqual(result["pairs"]["both_capped"],31)
        self.assertEqual(result["pairs"]["only_qwen35_capped"],31)
        self.assertTrue(result["cannot_infer_correct_reasoning_from_truncation"])
        self.assertFalse(result["model_superiority_proven"])
        self.assertEqual(len(private),93)

    def test_matching_terminal_number_while_length_capped_is_not_correct(self):
        source=rows()
        source[0]["raw_private"]="Long reasoning.\nFINAL_ANSWER: 50"
        result,_=audit_rows(source,expected_counts=None)
        self.assertEqual(result["models"]["qwen3:8b"]["visible_matching_final_but_incomplete"],1)
        self.assertEqual(result["models"]["qwen3:8b"]["strict_correct"],41)

    def test_no_guessing_from_random_number_and_duplicate_final_marker(self):
        source=rows()
        source[0]["raw_private"]="The answer might be 50 but maybe not"
        source[1]["raw_private"]="FINAL_ANSWER: 51\nFINAL_ANSWER: 51"
        result,_=audit_rows(source,expected_counts=None)
        self.assertEqual(result["models"]["qwen3:8b"]["visible_matching_final_but_incomplete"],0)

    def test_rejects_malformed_paired_private_source(self):
        with self.assertRaises(AuditError):audit_rows(rows()[:-1],expected_counts=None)
        source=rows()
        source[0]["grade"]["status"]="correct"
        with self.assertRaises(AuditError):audit_rows(source,expected_counts=None)
        source=rows()
        source[0]["id"]=source[1]["id"]
        with self.assertRaises(AuditError):audit_rows(source,expected_counts=None)

    def test_public_aggregate_never_contains_private_questions_or_answers(self):
        source=rows()
        source[0]["raw_private"]="secret unique payload "*20
        result,private=audit_rows(source,expected_counts=None)
        self.assertNotIn("secret unique payload",json.dumps(result))
        self.assertNotIn("private problem",json.dumps(result))
        self.assertNotIn("family0",json.dumps(result))
        self.assertIn("secret unique payload",json.dumps(private))

    def test_original_published_counts_must_match(self):
        with self.assertRaises(AuditError):audit_rows(rows())
        result,_=audit_rows(realistic_rows())
        self.assertEqual(result["models"]["qwen3:8b"]["strict_correct"],29)
        self.assertEqual(result["models"]["qwen3.5:9b"]["strict_correct"],8)

    def test_create_only_scalar_and_private_review_outside_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"v7-foundation-local-42499079467-1"
            folder.mkdir()
            (folder/"private-responses.jsonl").write_text(
                "\n".join(json.dumps(x) for x in realistic_rows())+"\n",encoding="utf8")
            pub=Path(temp)/"scalar.json"
            private=Path(temp)/"private.jsonl"
            main(["--run-dir",str(folder),"--aggregate-out",str(pub),
                  "--private-out",str(private)])
            self.assertTrue(pub.is_file())
            self.assertTrue(private.is_file())
            self.assertNotIn("private problem",pub.read_text(encoding="utf8"))
            with self.assertRaises(AuditError):
                main(["--run-dir",str(folder),"--aggregate-out",str(pub),
                      "--private-out",str(private)])

if __name__=="__main__":
    unittest.main()
