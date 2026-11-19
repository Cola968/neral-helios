"""V3 independent provenance/plan checks: a sandbox exit=0 is NOT evidence."""
import unittest
from decimal import Decimal
from neral_v3_agent import (validate_web_pages, plan_request,
                            anchored_numeric_facts, expected_material_value,
                            guard_sandbox_stdout, missing_web_data_reply, try_source_anchored_unit_worker)


QUESTION = ("Suche aktuell den Goldpreis in Euro pro Feinunze und rechne, "
            "was ein 250-Gramm-Goldbarren zum reinen Materialwert kostet "
            "(1 Feinunze = 31,1035 Gramm). Nenne die Quelle.")
PAGE = {"url":"https://prices.example.org/gold","title":"Goldkurs",
        "text":"Goldpreis aktuell: 3.650,20 EUR pro Feinunze (EUR/oz). Stand heute.", "read":True}


class V3AgentGroundingTests(unittest.TestCase):
    def test_planner_restricts_tools_and_bounds_steps(self):
        raw='{"steps":["web_search","web_reader","python_sandbox","validation_checker"],"needs_current_data":true}'
        plan=plan_request(QUESTION, raw, has_web=True)
        self.assertEqual(plan["steps"][0], "web_search")
        self.assertEqual(plan["steps"][-1], "validation_checker")
        self.assertIsNone(plan_request(QUESTION,'{"steps":["shell_exec"]}',has_web=True))

    def test_page_evidence_must_be_an_actual_read_page(self):
        self.assertEqual(len(validate_web_pages([PAGE])),1)
        altered={**PAGE,"read":False}
        with self.assertRaises(ValueError):
            validate_web_pages([altered])

    def test_quote_with_live_euro_ounce_price_is_grounded(self):
        facts=anchored_numeric_facts([PAGE])
        self.assertTrue(any(f["value"]==Decimal("3650.20") and f["unit"]=="EUR/oz"
                            for f in facts))
        expected=expected_material_value(QUESTION,facts)
        self.assertEqual(expected,Decimal("29339.14"))
        self.assertTrue(guard_sandbox_stdout("29339.14", expected=expected))
        self.assertFalse(guard_sandbox_stdout("0",expected=expected))
        self.assertFalse(guard_sandbox_stdout("30000.00",expected=expected))

    def test_price_quoted_as_number_then_euro_symbol_is_ounce_price(self):
        source={**PAGE,"text":(
            "Gold ist aktuell wertvoll. Preis je Feinunze: 3.800,50 €. "
            "Diese Angabe steht im Seiteninhalt, nicht im Such-Snippet.")}
        facts=anchored_numeric_facts([source])
        self.assertTrue(any(f["value"]==Decimal("3800.50") and
                            f["unit"]=="EUR/oz" for f in facts))

    def test_general_unit_worker_executes_only_page_grounded_parameters(self):
        class ActualDocker:
            def __init__(self): self.codes=[]
            def execute_python(self,code):
                self.codes.append(code)
                return {"status":"PASSED","output":"29339.14"}
        docker=ActualDocker()
        result=try_source_anchored_unit_worker(docker, QUESTION,
                                                anchored_numeric_facts([PAGE]))
        self.assertTrue(result["sandbox_executed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["tool_used"],"python_sandbox")
        self.assertIn("29.339,14",result["text"])
        self.assertIn(PAGE["url"],result["text"])
        self.assertEqual(len(docker.codes),1)
        self.assertIn("Decimal",docker.codes[0])
        self.assertIn("3650.20",docker.codes[0])
        self.assertIn("31.1035",docker.codes[0])

    def test_unit_worker_rejects_dummy_and_missing_source_without_execution(self):
        class WrongDocker:
            def __init__(self):self.calls=0
            def execute_python(self,code):
                self.calls+=1
                return {"status":"PASSED","output":"0"}
        wrong=WrongDocker()
        self.assertIsNone(try_source_anchored_unit_worker(wrong,QUESTION,
                         anchored_numeric_facts([PAGE])))
        self.assertEqual(wrong.calls,1)
        absent=WrongDocker()
        self.assertIsNone(try_source_anchored_unit_worker(absent,QUESTION,[]))
        self.assertEqual(absent.calls,0)

    def test_no_web_value_never_becomes_zero_sandbox_result(self):
        empty=[{**PAGE,"text":"Goldpreis auf Anfrage. Kurs wird dynamisch geladen."}]
        facts=anchored_numeric_facts(empty)
        self.assertEqual(facts,[])
        self.assertFalse(guard_sandbox_stdout("0",missing_required=True))
        reply=missing_web_data_reply(QUESTION,empty,"Goldpreis EUR pro Feinunze")
        self.assertIn("nicht",reply["text"].lower())
        self.assertFalse(reply["sandbox_executed"])
        self.assertFalse(reply["verified"])
        self.assertNotIn("Sandbox-Ausgabe (nicht unabhängig geprüft):\\n0",reply["text"])

    def test_unreadable_or_private_pages_cannot_be_evidence(self):
        invalid={**PAGE,"url":"https://127.0.0.1/admin"}
        with self.assertRaises(ValueError):
            validate_web_pages([invalid])


if __name__=="__main__":
    unittest.main()
