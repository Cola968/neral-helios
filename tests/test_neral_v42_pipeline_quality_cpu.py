"""Regression contracts for exact outputs, enforced repair tooling, grounded evidence."""
import unittest
from unittest.mock import patch, Mock
from scripts.neral_v41_blind200_live import _exact_text, grade_response, execute_case
from neral_v42_quality import prompt_for_exact_case, evidence_veto_is_generic, grounded_literal_citations, forced_repair_tool
from neral_evidence_agent import run_evidence_agent


class ExactFinalCPU(unittest.TestCase):
    def test_endresult_parser_accepts_unambiguous_terminal_value(self):
        self.assertEqual(_exact_text("Schritt 1: 17 * 23\nENDERGEBNIS: 391","integer"),"391")
        self.assertEqual(_exact_text("Zuerst prüfen.\nENDERGEBNIS: Ja","yesno"),"ja")
        self.assertIsNone(_exact_text("ENDERGEBNIS: 391\nENDERGEBNIS: 392","integer"))
        self.assertIsNone(_exact_text("Zwischenstand 391\nENDERGEBNIS: 391 oder 392","integer"))
        self.assertIsNone(_exact_text("ENDERGEBNIS: 391\nWeiterer Text","integer"))
        self.assertIsNone(_exact_text("Zahl 391 irgendwo im Text","integer"))

    def test_prompt_only_appended_for_exact_grading_not_sealed_reference_mutation(self):
        case={"id":"x","prompt":"Berechne 17 mal 23.",
              "grader":"independent_exact","sealed_reference":{"kind":"integer","value":"391"}}
        text=prompt_for_exact_case(case)
        self.assertIn("ENDERGEBNIS:",text)
        self.assertNotIn("391",text)
        self.assertEqual(case["prompt"],"Berechne 17 mal 23.")
        other={"prompt":"Fasse diesen Text zusammen","grader":"blind_pairwise"}
        self.assertEqual(prompt_for_exact_case(other),other["prompt"])

    def test_final_marked_answer_must_still_match_sealed_gold(self):
        case={"category":"math_logic","grader":"independent_exact",
              "sealed_reference":{"kind":"integer","value":"391"}}
        self.assertEqual(grade_response(case,{"text":"Rechnung.\nENDERGEBNIS: 391"}),
                         ("pass","independent_exact"))
        self.assertEqual(grade_response(case,{"text":"Rechnung.\nENDERGEBNIS: 392"}),
                         ("fail","exact_mismatch"))

    def test_execute_case_uses_marked_prompt_without_modifying_gold(self):
        class Client:
            def __init__(self):
                self.posts=[]
            def connect(self):
                pass
            def post(self,tier,prompt,history,**kwargs):
                self.posts.append(prompt)
                return 200, {"text":"ENDERGEBNIS: 391"}, 0.1
        case={"id":"x","category":"math_logic","prompt":"Berechne 17 mal 23.",
              "grader":"independent_exact","sealed_reference":{"kind":"integer","value":"391"},
              "history":[]}
        client=Client()
        code,response,_,failure=execute_case(client,"Quartz",case)
        self.assertEqual(code,200)
        self.assertIn("ENDERGEBNIS",client.posts[-1])
        self.assertEqual(case["prompt"],"Berechne 17 mal 23.")
        self.assertIsNone(failure)


class RepairToolCPU(unittest.TestCase):
    def test_forced_tool_for_math_constraints_and_authorized_wiki(self):
        self.assertEqual(forced_repair_tool("Berechne 17 mal 23.",local=False,web=False),"python_sandbox")
        self.assertEqual(forced_repair_tool("Ordne Ada anhand der Schichtbedingungen zu.",local=False,web=False),"z3_solver")
        self.assertEqual(forced_repair_tool("Was ist ein Pulsar?",local=True,web=False),"direct_reasoning")
        self.assertIsNone(forced_repair_tool("Was ist ein Pulsar?",local=False,web=False))
        self.assertIsNone(forced_repair_tool("Wie lautet das Wetter heute?",local=False,web=False))
        self.assertEqual(forced_repair_tool("Was ist ein Pulsar?",local=False,web=True),"direct_reasoning")

    def test_repair_planner_forces_python_and_executes_not_just_claims(self):
        QUESTION = "Ein Paket wiegt 7 kg, ein anderes 12 kg. Bestimme das Gesamtgewicht."
        GOOD = {"locale":"de","calculations":[{"label":"Gewicht","expr":"u1 + u2","unit":"kg"}],"missing":[]}
        OK = {"ok":True,"issue":""}
        proposed=[
            {"tool":"direct_reasoning","parts":["ohne Tool"],"search_query":""},
            {"answer":"Schätzung 19","citations":[]},
            {"ok":False,"issue":"Berechnung nicht ausgeführt"},
            {"tool":"python_sandbox","parts":["wirklich berechnen"],"search_query":""},
            GOOD,OK]
        prompts=[]
        def respond(system,body,budget,schema):
            import json
            prompts.append(json.loads(body))
            return json.dumps(proposed.pop(0))
        tool=Mock(return_value={"status":"PASSED","output":"19"})
        with patch("neral_evidence_agent.quantity_plan",return_value=None), \
             patch("neral_v32_verifier.verified_math_question",return_value=None), \
             patch("neral_v32_verifier.verified_python_question",return_value=None), \
             patch("neral_v32_verifier.verified_shift_count",return_value=None), \
             patch("neral_knowledge_verifier.verified_element_question",return_value=None):
            result=run_evidence_agent(QUESTION,respond,tool,deep=True)
        self.assertGreaterEqual(tool.call_count,1)
        self.assertEqual(result["tool_used"],"python_sandbox")
        self.assertTrue(result["agent_checked"])
        self.assertIn("required_tool",prompts[3])
        self.assertEqual(prompts[3]["required_tool"],"python_sandbox")

    def test_programming_repair_executes_code_in_real_sandbox_contract(self):
        from neral_evidence_agent import _work, AgentError
        import ast
        self.assertEqual(
            forced_repair_tool("Schreibe eine Python-Funktion is_prime(n)", local=False, web=False),
            "python_code_sandbox")
        plan={"tool":"python_code_sandbox","parts":["Implementiere Funktion"]}
        packet={"code":"def is_prime(n):\n    return n > 1"}
        tool=Mock(return_value={"status":"PASSED","output":"NERAL_CODE_SMOKE_OK"})
        result=_work(plan,packet,{"user":"Schreibe is_prime"},tool,set(),{})
        self.assertIn("def is_prime",result)
        self.assertEqual(tool.call_count,1)
        self.assertIn("NERAL_CODE_SMOKE_OK",tool.call_args.args[0])
        with self.assertRaises(AgentError):
            _work(plan,{"code":"def is_prime(n):\n return ("},
                  {"user":"X"},tool,set(),{})
        self.assertEqual(tool.call_count,1)

    def test_unauthorized_web_repair_is_never_assumed_available(self):
        self.assertIsNone(forced_repair_tool("Welche Schlagzeile heute?",local=False,web=False))


class TelemetryWebProbeCPU(unittest.TestCase):
    def test_only_public_web_knowledge_question_triggers_v4(self):
        from scripts.neral_v42_telemetry_probe import web_query_for_diagnostic, DIAGNOSTIC_CASES
        self.assertEqual(web_query_for_diagnostic(
            "knowledge", DIAGNOSTIC_CASES[2][1]),
            "Was ist eine Sonnenfinsternis?")
        self.assertIsNone(web_query_for_diagnostic("knowledge","Geheime Nutzerfrage"))
        self.assertIsNone(web_query_for_diagnostic("math_logic","Berechne 17 x 23"))
        self.assertIsNone(web_query_for_diagnostic("everyday","Erkläre thermische Ausdehnung"))


class GroundedEvidenceCPU(unittest.TestCase):
    def test_actual_web_agent_bypasses_only_false_missing_quote_veto(self):
        import json
        from unittest.mock import Mock
        source_text="Elektrostatik untersucht ruhende elektrische Ladungen."
        def diagnostic_response(extra_claim=False):
            answers=[
                {"tool":"direct_reasoning","parts":["Definition mit Quelle"],},
                {"answer":source_text + (" Pluto ist ein Stern." if extra_claim else ""),
                 "citations":["p1"]},
                {"ok":False,"issue":"Wörtlicher Quellenbeleg fehlt."},
            ]
            def respond(system,payload,budget,schema):
                return json.dumps(answers.pop(0))
            return respond
        search=Mock()
        search.search.return_value={"results":[{"url":"https://example.org/electrostatics"}]}
        page_reader=Mock(return_value=[{"url":"https://example.org/electrostatics",
                                         "page_read":True,"page_text":source_text}])
        kwargs={"web_query":"Was ist Elektrostatik?", "search":search,
                "page_reader":page_reader,"max_attempts":1}
        with patch("neral_v32_verifier.verified_math_question",return_value=None), \
             patch("neral_knowledge_verifier.verified_element_question",return_value=None):
            accepted=run_evidence_agent("Was ist Elektrostatik?",
                                       diagnostic_response(),Mock(),**kwargs)
            rejected=run_evidence_agent("Was ist Elektrostatik?",
                                       diagnostic_response(extra_claim=True),Mock(),**kwargs)
        self.assertTrue(accepted["agent_checked"])
        self.assertFalse(accepted["verified"])
        self.assertFalse(accepted["safe_to_promote"])
        self.assertFalse(accepted["tool_verified"])
        self.assertEqual(accepted["verification_scope"],
                         "literal_source_read_not_independent_fact_proof")
        self.assertEqual(accepted["veto_override_reason_code"],
                         "generic_quote_veto_exact_literal")
        self.assertFalse(rejected["agent_checked"])
        self.assertEqual(rejected["error_code"],"checker_veto")

    def test_literal_wiki_and_read_web_are_allowed_if_checker_only_gives_generic_veto(self):
        packet={"answer":"Elektrostatik untersucht elektrische Ladungen.",
                "citations":[{"source":"wiki1","quote":"Elektrostatik untersucht elektrische Ladungen."}]}
        evidence={"wiki1":"Elektrostatik untersucht elektrische Ladungen."}
        wiki=[{"url":"http://127.0.0.1:9691/content/de/book","title":"Elektrostatik",
               "book":"de","context":"Elektrostatik untersucht elektrische Ladungen."}]
        self.assertTrue(grounded_literal_citations(packet,evidence,[],wiki))
        self.assertTrue(evidence_veto_is_generic("Wörtlicher Quellenbeleg fehlt."))
        self.assertFalse(evidence_veto_is_generic("Unbelegte Behauptung: der Planet besteht aus Gold."))
        web_evidence={"web1":"Ein Komet besteht aus Eis und Staub."}
        web_packet={"answer":"Ein Komet besteht aus Eis und Staub.",
                    "citations":[{"source":"web1","quote":"Ein Komet besteht aus Eis und Staub."}]}
        sources=[{"page_read":True,"url":"https://example.org/comets"}]
        self.assertTrue(grounded_literal_citations(web_packet,web_evidence,sources,[]))
        sources[0]["page_read"]=False
        self.assertFalse(grounded_literal_citations(web_packet,web_evidence,sources,[]))
        sources[0]["page_read"]=True
        self.assertTrue(grounded_literal_citations(
            {"answer":"Ein Komet besteht aus Eis und Staub.","citations":["p1"]},
            web_evidence,sources,[],passages={"p1":("web1","Ein Komet besteht aus Eis und Staub.")}))
        self.assertFalse(grounded_literal_citations(
            {"answer":"Ein Komet besteht aus Eis und Staub.","citations":["p1"]},
            web_evidence,sources,[],passages={"p1":("web1","Nicht in Quelle.")}))
        # An unsupported extra statement is NEVER allowed even with one real quote.
        packet["answer"]="Elektrostatik untersucht elektrische Ladungen. Und Pluto ist ein Stern."
        self.assertFalse(grounded_literal_citations(packet,evidence,[],wiki))

if __name__=="__main__":
    unittest.main()
