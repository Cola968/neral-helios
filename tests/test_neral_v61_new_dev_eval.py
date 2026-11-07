"""New 72-case *development* comparison v61 vs v5 and fixed few-shot Qwen."""
from collections import Counter
import json
import unittest
from fractions import Fraction

from scripts.neral_v61_new_dev_eval import (
    cases72, decontaminate_cases, score_result, validate_candidate_receipt,
)


class NewV61DevTests(unittest.TestCase):
    def test_72_diverse_problems_and_structurally_new_domains(self):
        cases=cases72()
        self.assertEqual(len(cases),72)
        self.assertEqual(Counter(c["domain"] for c in cases),
                         {"math":24,"reasoning":24,"code_trace":24})
        self.assertEqual(len({c["id"] for c in cases}),72)
        self.assertEqual(len({c["prompt"] for c in cases}),72)
        self.assertEqual(len({(c["domain"],c["family"]) for c in cases}),12)
        self.assertEqual(len({(c["family"],c["language"]) for c in cases}),24)
        self.assertTrue(all(c["kind"]=="integer" for c in cases))
        self.assertTrue(all(c["answer"].lstrip("-").isdigit() for c in cases))
        self.assertEqual(cases,cases72())

    def test_python_trace_oracles_replayed_by_independent_cpu_formulas(self):
        # Hand-executable CPU reference checks, not model-generated code.
        cases=cases72()
        by_family={}
        for row in cases:
            by_family.setdefault(row["family"],[]).append(row)
        for i,case in enumerate(by_family["python_mutable_aliasing"]):
            p,q=4+i,7+i
            expected=p+q+4*i+6  # appended item 3 AND shared list length 3
            self.assertEqual(int(case["answer"]),expected)
        self.assertEqual([int(x["answer"]) for x in by_family["python_recursion_fib"]],
                         [5,8,13,21,34,55])
        for i,case in enumerate(by_family["python_set_comprehension"]):
            numbers=[3+i,5+i,6+2*i,12+i,15+3*i,20+i]
            divisor=3+i%3
            unique=set()
            for number in numbers:
                if number%2==0: unique.add(number%divisor)
            self.assertEqual(int(case["answer"]),sum(unique))
        for i,case in enumerate(by_family["python_enumerate_weight"]):
            numbers=[1+i,4+i,6+2*i,9+i,12+i]
            total=0
            for index in range(0,len(numbers),2):
                total+=(index+1)*numbers[index]
            self.assertEqual(int(case["answer"]),total)

    def test_decon_against_v61_train_and_old_72_holdout(self):
        self.assertTrue(decontaminate_cases(cases72()))

    def test_grades_unambiguous_and_no_promotable_release_claim(self):
        cases=cases72()
        results={v:[{"id":r["id"],"text":"FINAL_ANSWER: "+r["answer"],
                     "complete":True,"new_tokens":8} for r in cases]
                 for v in ("qwen_fewshot","neral_v5","neral_v6")}
        results["neral_v6"][0]["text"]="I cannot tell"
        results["neral_v5"][1]["text"]="FINAL_ANSWER: 999999"
        result=score_result(cases,results,candidate_sha="e"*64)
        self.assertEqual(result["scores"]["neral_v61"]["correct"],71)
        self.assertEqual(result["scores"]["neral_v5"]["correct"],71)
        self.assertEqual(result["scores"]["qwen_fewshot"]["correct"],72)
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["independent_generalization_proven"])
        self.assertFalse(result["adapter_promoted"])
        self.assertEqual(result["paired"]["v61_vs_v5"]["regressions"],1)
        self.assertEqual(result["paired"]["v61_vs_v5"]["improvements"],1)
        self.assertNotIn(cases[0]["prompt"],json.dumps(result))

    def test_withdrawn_alias_gold_fails_execution_and_current_gold_matches(self):
        # d7c33cc used (p+3*i)+(q+i)+3, which counts the appended 3 once
        # and drops len(b). CPython prints three more.
        from scripts.neral_dev_oracle import (
            OracleMismatch, assert_independent_oracles, replay_source,
        )
        cases=cases72()
        alias=[row for row in cases if row["family"]=="python_mutable_aliasing"]
        self.assertEqual(len(alias),6)
        for i,case in enumerate(alias):
            p,q=4+i,7+i
            withdrawn=(p+3*i)+(q+i)+3
            self.assertEqual(withdrawn,14+6*i)
            source=(f"a = [{p}, {q}]\nb = a\na.append(3)\nb[0] += {3*i}\n"
                    f"a[1] += {i}\nprint(sum(a) + len(b))")
            replayed=replay_source(source)
            self.assertEqual(replayed,withdrawn+3)
            self.assertEqual(int(case["answer"]),replayed)
            forged=dict(case,answer=str(withdrawn))
            with self.assertRaises(OracleMismatch):
                assert_independent_oracles([forged])

    def test_unsafe_source_is_rejected_before_execution(self):
        from scripts.neral_dev_oracle import replay_source
        for source in (
            "import os\nprint(1)",
            "print(open('x'))",
            "print(__import__('os'))",
            "exec('print(1)')\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError) as caught:
                    replay_source(source)
                self.assertIn("before execution",str(caught.exception))

    def test_every_family_wrong_integer_fails_and_prose_stays_ungraded(self):
        cases=cases72()
        families=sorted({row["family"] for row in cases})
        self.assertEqual(len(families),12)
        variants=("qwen_fewshot","neral_v5","neral_v6")
        for family in families:
            wrong={name:[{"id":row["id"],"text":"FINAL_ANSWER: "+row["answer"],
                          "complete":True,"new_tokens":4} for row in cases]
                   for name in variants}
            prose={name:[dict(item) for item in wrong[name]] for name in variants}
            for row,item in zip(cases,wrong["neral_v6"]):
                if row["family"]==family:
                    item["text"]="FINAL_ANSWER: "+str(int(row["answer"])+1)
            for row,item in zip(cases,prose["qwen_fewshot"]):
                if row["family"]==family:
                    item["text"]="Zwischenwert "+row["answer"]+" bleibt unklar"
            scored=score_result(cases,wrong,candidate_sha="e"*64)
            blurred=score_result(cases,prose,candidate_sha="e"*64)
            with self.subTest(family=family):
                self.assertEqual(scored["scores"]["neral_v61"]["correct"],66)
                self.assertEqual(scored["scores"]["neral_v61"]["ungraded_completed"],0)
                self.assertEqual(scored["scores"]["neral_v5"]["correct"],72)
                self.assertEqual(blurred["scores"]["qwen_fewshot"]["correct"],66)
                self.assertEqual(blurred["scores"]["qwen_fewshot"]["ungraded_completed"],6)
                self.assertFalse(scored["release_validated"])

    def test_local_weights_and_receipt_sha_validated_without_powershell_coercion(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from scripts.neral_v61_new_dev_eval import validate_local_candidate
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            v5=root/"v5";candidate=root/"candidate"
            v5.mkdir();candidate.mkdir()
            b=b"model-candidate"
            (candidate/"adapter_model.safetensors").write_bytes(b)
            (v5/"adapter_model.safetensors").write_bytes(b"old-v5")
            sha=hashlib.sha256(b).hexdigest()
            receipt={
                "status":"NERAL_V61_R16_REPLAY_120_STEPS_EXPERIMENTAL",
                "steps_completed":120,"train_rows":800,"validation_rows":140,
                "source_counts":{"v5_train_replay":560,"v6_train_multistep":240},
                "predecessor_adapter_sha256":"2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5",
                "candidate_adapter_sha256":sha,
                "release_validated":False,"adapter_promoted":False,
            }
            p=root/"receipt.json";p.write_text(json.dumps(receipt),encoding="utf8")
            with self.assertRaises(ValueError):
                validate_local_candidate(p,candidate,v5)
            expected_v5=hashlib.sha256(b"old-v5").hexdigest()
            self.assertEqual(validate_local_candidate(
                p,candidate,v5,expected_v5_sha=expected_v5),sha)
            # Candidate bytes changed -> never trust metadata alone.
            (candidate/"adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate_local_candidate(p,candidate,v5,expected_v5_sha=expected_v5)

    def test_evaluation_workflow_uses_python_receipt_validator(self):
        from pathlib import Path
        workflow=(Path(__file__).resolve().parents[1]/
                  ".github"/"workflows"/"neral-v61-dev72-gpu.yml").read_text(encoding="utf8")
        self.assertIn("--mode verify",workflow)
        self.assertIn("Python verification rejected local v6.1 receipt or adapter SHA",workflow)
        self.assertNotIn("$report=Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json",workflow)

    def test_reject_candidate_receipt_from_unrelated_run_or_model(self):
        good={
            "status":"NERAL_V61_R16_REPLAY_120_STEPS_EXPERIMENTAL",
            "steps_completed":120,"train_rows":800,"validation_rows":140,
            "source_counts":{"v5_train_replay":560,"v6_train_multistep":240},
            "predecessor_adapter_sha256":"2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5",
            "candidate_adapter_sha256":"e"*64,
            "release_validated":False,"adapter_promoted":False,
        }
        self.assertEqual(validate_candidate_receipt(good),"e"*64)
        for bad in (
            dict(good,steps_completed=119),
            dict(good,release_validated=True),
            dict(good,source_counts={}),
            dict(good,predecessor_adapter_sha256="a"*64),
        ):
            with self.assertRaises(ValueError):
                validate_candidate_receipt(bad)


if __name__=="__main__":
    unittest.main()
