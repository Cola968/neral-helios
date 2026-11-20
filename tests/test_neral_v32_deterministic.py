"""V3.2 deterministic boundaries: red tests precede implementations."""
import unittest
from neral_v32_verifier import (
    validate_python_script, verified_python_question,
    normalize_z3_schema, verified_shift_count,
    grounded_entity_facts, deterministic_web_answer,
)
from scripts.neral_v31_prospective100 import evaluate_case, summarize, build_suite

class V32DeterministicContracts(unittest.TestCase):
    def test_python_multiline_safe_whitelist(self):
        code = "import math\nfrom itertools import product\ns=0\nfor x in range(3):\n for y in range(2):\n  s+=x+y\nprint(s)"
        self.assertIn("for x in range(3)", validate_python_script(code))
        for bad in (
            "import os\nprint(os.getcwd())", "open('secret')",
            "while True:\n print('x')",
            "print((1).__class__)", "__import__('os')",
            "import subprocess", "eval('3')", "exec('print(1)')",
            "print(globals())", "print(getattr(1,'real'))",
            "print([x for x in range(9999999)])",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_python_script(bad)

    def test_complete_python_question_uses_real_sandbox_and_stdout(self):
        calls=[]
        def docker(code):
            calls.append(code)
            return {"status":"PASSED","output":"24\n","exit_code":0}
        q="Welchen ganzzahligen Wert gibt dieses Python-Programm aus?\n'''python\nx=8\nfor i in range(2):\n x+=8\nprint(x)\n'''"
        r=verified_python_question(q,docker)
        self.assertEqual(len(calls),1)
        self.assertEqual(r["tool_used"],"python_sandbox")
        self.assertTrue(r["tool_verified"])
        self.assertEqual(r["text"],"Ergebnis: 24")
        self.assertEqual(r["error_code"],None)
        fail=verified_python_question(q,lambda code:{"status":"FAILED","error_code":"timeout"})
        self.assertFalse(fail["tool_verified"])
        self.assertEqual(fail["error_code"],"sandbox_error")

    def test_schema_normalizer_fills_presentation_not_missing_rules(self):
        data={"variables":[{"name":"ada","min":1,"max":4}],
              "constraints":[{"description":"Ada ist in Schicht eins","expr":"ada == 1"}]}
        out=normalize_z3_schema(data,objective="count_all")
        self.assertEqual(out["objective"],"count_all")
        self.assertEqual(out["constraints"][0]["rule_id"],"R1")
        with self.assertRaises(ValueError):
            normalize_z3_schema({"variables":[],"constraints":[]})
        with self.assertRaises(ValueError):
            normalize_z3_schema({"variables":[{"name":"ada","min":1,"max":4}],
                                  "constraints":[]})
        with self.assertRaises(ValueError):
            normalize_z3_schema({"variables":[{"name":"ada"}],
                                  "constraints":[{"expr":"ada == 1"}]})

    def test_four_shift_count_has_complete_independent_permutation_certificate(self):
        q=("Vier Mitarbeitende Ada, Ben, Cem und Dora erhalten jeweils genau eine der Schichten 1, 2, 3, 4. "
           "Jede Schicht ist genau einmal besetzt. Ada arbeitet Schicht 1, Ben darf nicht Schicht 2 "
           "und Cem darf nicht Schicht 3 arbeiten. Wie viele zulässige vollständige Schichtpläne gibt es? "
           "Zähle mit Z3 alle Lösungen und gib die Anzahl aus.")
        codes=[]
        def docker(code):
            codes.append(code)
            # independent enumeration: Ada=1; among [2,3,4] Ben!=2, Cem!=3 -> 3 valid
            return {"status":"PASSED","output":"3","exit_code":0}
        result=verified_shift_count(q,docker)
        self.assertTrue(result["tool_verified"])
        self.assertEqual(result["tool_used"],"z3_solver")
        self.assertEqual(result["text"],"Z3: 3")
        self.assertIn("solver.add",codes[0])
        bad=verified_shift_count(q,lambda code:{"status":"PASSED","output":"4","exit_code":0})
        self.assertEqual(bad["error_code"],"checker_veto")
        self.assertFalse(bad["tool_verified"])
        self.assertIsNone(verified_shift_count("Schichtpläne bitte zählen, aber ohne Angaben",docker))

    def test_literal_web_facts_must_cover_each_named_entity(self):
        sources=[
           {"url":"https://example.com/a","page_read":True,
            "page_text":"Firefox is a browser.\nFirefox uses Gecko for rendering web pages."},
           {"url":"https://example.com/b","page_read":True,
            "page_text":"Chrome is a browser.\nChrome is built around Blink, a web rendering engine."},
        ]
        evidence=grounded_entity_facts(("Firefox","Chrome"), sources)
        self.assertEqual(set(evidence),{"Firefox","Chrome"})
        result=deterministic_web_answer(("Firefox","Chrome"),sources)
        self.assertTrue(result["tool_verified"])
        self.assertEqual(result["tool_used"],"direct_reasoning")
        self.assertIn("Firefox",result["text"])
        self.assertIn("Chrome",result["text"])
        self.assertIn("[web1]",result["text"])
        self.assertIn("[web2]",result["text"])
        self.assertIsNone(deterministic_web_answer(("Firefox","Chrome"), sources[:1]))
        self.assertIsNone(deterministic_web_answer(("Firefox","Chrome"),[
             {**sources[0],"page_read":False},sources[1]]))

    def test_grader_separates_answer_correct_from_tool_verified(self):
        case={"category":"mathematics","kind":"integer","gold":10,"tool":"python_sandbox"}
        r=evaluate_case(case,{"text":"Antwort: 10","agent_checked":False},200)
        self.assertTrue(r["answer_correct"])
        self.assertFalse(r["tool_verified"])
        self.assertFalse(r["strict_e2e_pass"])
        self.assertEqual(r["error"],"no_tool_called")
        okay=evaluate_case(case,{"text":"Antwort: 10","tool_used":"python_sandbox",
                                 "tool_verified":True},200)
        self.assertTrue(okay["strict_e2e_pass"])
        self.assertEqual(okay["error"],None)

    def test_web_grader_does_not_require_facts_in_first_180_chars(self):
        case={"category":"web","kind":"web","entities":["Firefox","Chrome"],
              "facts":["Gecko","Blink"]}
        text=("Firefox: "+"Es gibt eine Reihe unterschiedlicher Eigenschaften. "*5+
               "Die Browser-Engine heißt Gecko.\nChrome: verwendet die Blink-Engine.\n"
               "[web1] „Firefox uses the Gecko rendering engine“\n"
               "[web2] „Chrome uses Blink as its rendering engine“")
        r=evaluate_case(case,{"text":text,"tool_used":"direct_reasoning",
                              "agent_checked":True,"sources":[
                                  {"url":"https://a.test","page_read":True},
                                  {"url":"https://b.test","page_read":True}]},200)
        self.assertTrue(r["answer_correct"])
        self.assertTrue(r["strict_e2e_pass"])

    def test_evidence_agent_bypasses_model_veto_for_executable_source(self):
        from neral_evidence_agent import run_evidence_agent
        q=("Welchen ganzzahligen Wert gibt dieses Python-Programm aus?\n"
           "'''python\nx=10\nfor i in range(2):\n x+=7\nprint(x)\n'''")
        calls=[]
        def execute(code):
            calls.append(code)
            return {"status":"PASSED","exit_code":0,"output":"24"}
        def forbidden_model(*args):
            raise AssertionError("Deterministic source must not call LLM")
        r=run_evidence_agent(q,forbidden_model,execute)
        self.assertTrue(r["tool_verified"])
        self.assertEqual(r["tool_used"],"python_sandbox")
        self.assertEqual(len(calls),1)

    def test_web_comparison_subjects_are_parsed_before_generation(self):
        from neral_evidence_agent import _comparison_entities
        question = "Firefox vs Chrome: Vergleiche ihre Browser-Engine anhand tatsächlich gelesener Webseiten."
        self.assertEqual(_comparison_entities(question), ("Firefox", "Chrome"))
        sources = [
            {"url":"https://example.com/1","page_read":True,
             "page_text":"Firefox is a browser.\\nFirefox uses Gecko for rendering pages."},
            {"url":"https://example.com/2","page_read":True,
             "page_text":"Chrome is a browser.\\nChrome uses Blink for rendering pages."},
        ]
        facts = grounded_entity_facts(("Firefox","Chrome"), sources)
        self.assertEqual(set(facts), {"Firefox","Chrome"})
        self.assertEqual(facts["Firefox"]["source"], "web1")
        self.assertEqual(facts["Chrome"]["source"], "web2")

    def test_100_synthetic_raw_case_details_survive_after_run(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.neral_v31_prospective100 import run_live
        def fake_fetch(path, payload=None, csrf=None):
            if path != "/api/neral/chat/three-tier":
                raise AssertionError("Unexpected external call")
            return 200, {"text":"not solved","tool_used":None,
                         "agent_checked":False,"agent_steps":["planner"]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=SimpleNamespace(
                authorize="NERAL_V32_OWNER_APPROVED_NEW_100",
                sealed=root/"frozen.json",progress=root/"checkpoint.jsonl",
                details=root/"benchmark_details.jsonl",output=root/"summary.json")
            with patch("scripts.neral_v31_prospective100._open_client",
                       return_value=(fake_fetch,"dummy-local-session")), \
                 patch("scripts.neral_v31_prospective100.time.sleep"), \
                 patch("scripts.neral_v31_prospective100.secrets.token_hex",
                       return_value="00112233445566778899aabbccddeeff"):
                report=run_live(args)
            lines=args.details.read_text(encoding="utf8").splitlines()
            self.assertEqual(len(lines),100)
            self.assertEqual(report["detail_count"],100)
            self.assertTrue(report["synthetic_individual_rows_retained"])
            self.assertEqual(report["details_sha256"],
                             hashlib.sha256(args.details.read_bytes()).hexdigest())
            sample=json.loads(lines[0])
            self.assertEqual(set(("case","observed","grader","elapsed_seconds"))
                             - set(sample),set())
            self.assertIn("question",sample["case"])
            self.assertIn("answer_correct",sample["grader"])
            self.assertIn("strict_e2e_pass",sample["grader"])

    def test_summary_separates_both_metrics(self):
        cases=build_suite("00112233445566778899aabbccddeeff")
        rows=[{"id":x["id"],"category":x["category"],
               "strict_e2e_pass":i%2==0,"answer_correct":True,
               "tool_verified":i%2==0,"correct":i%2==0,
               "error": None if i%2==0 else "no_tool_called",
               "latency_seconds":1.25} for i,x in enumerate(cases)]
        stats=summarize(cases,rows)
        self.assertEqual(stats["answer_correct"],100)
        self.assertEqual(stats["strict_e2e_pass"],50)
        self.assertEqual(stats["correct"],50)
        self.assertEqual(stats["categories"]["logic"]["answer_correct"],20)

if __name__=="__main__":
    unittest.main()
