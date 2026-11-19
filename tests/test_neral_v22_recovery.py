"""V2.2 regression tests for runtime Docker recovery and honest model fallback."""
import os
import subprocess
import unittest
from unittest.mock import patch

from neral_three_tier_runtime import ThreeTierRuntime
from neral_tir_experimental import _run_ephemeral_python


class Model:
    def __init__(self, replies):
        self.replies=list(replies)
        self.calls=[]
        self.loaded=False
        self.model="qwen3.5:9b+neral-v2-full"
        self.adapter_sha256="a"*64
    def respond_chat(self, turns, *, max_new_tokens, temperature=None, max_time_seconds=None):
        self.loaded=True
        self.calls.append((turns,max_new_tokens))
        answer=self.replies.pop(0)
        if isinstance(answer,Exception):
            raise answer
        return answer
    def unload(self):
        self.loaded=False


def question(text="Wie viel sind 2 plus 3?"):
    return [{"role":"system","content":"NERAL offline"},{"role":"user","content":text}]


class ProbabilityRouting(unittest.TestCase):
    def test_probability_asks_are_not_misrouted_to_z3_by_mindestens(self):
        from neral_three_tier_runtime import _is_constraint_task, _is_free_compute_task
        for user_prompt in (
            "Geburtstagsparadoxon mit 23 Personen: Wahrscheinlichkeit mindestens zwei?",
            "Wie hoch ist die Wahrscheinlichkeit für höchstens drei Treffer?",
            "Wie groß ist die Chance von mindestens einem Fehler in 100 Tests?",
        ):
            self.assertTrue(_is_free_compute_task(user_prompt))
            self.assertFalse(_is_constraint_task(user_prompt))


class SourceGroundedNightAccounting(unittest.TestCase):
    def test_source_plan_is_derived_not_hardcoded(self):
        from neral_three_tier_runtime import _source_person_nights_plan
        user_prompt=("Drei Freundinnen mieten gemeinsam eine Berghütte für 1.260 €. "
            "Nora übernachtet dort die vollen 6 Nächte, Kim reist nach 4 Nächten "
            "wieder ab, und Pia kommt erst zur Hälfte der Zeit dazu, bringt aber "
            "für ihre letzten 2 Nächte noch ihren Bruder mit. Alle Übernachtungsplätze "
            "kosten pro Person und Nacht exakt gleich viel. Wie viel zahlt jede "
            "der drei Freundinnen (Pia inklusive ihres Bruders)?")
        plan=_source_person_nights_plan(user_prompt)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["nights"],{"Nora":6,"Kim":4,"Pia":3,"Bruder":2})
        self.assertEqual(plan["total_nights"],15)
        self.assertEqual(plan["shares"]["Nora"],504)
        self.assertEqual(plan["shares"]["Kim"],336)
        self.assertEqual(plan["shares"]["Pia"],420)

    def test_same_grammar_with_different_names_nights_and_price(self):
        from neral_three_tier_runtime import _source_person_nights_plan
        prompt=("Drei Freunde mieten gemeinsam eine Hütte für 1.400 €. "
            "Eva übernachtet dort die vollen 6 Nächte, Leo reist nach 3 Nächten "
            "wieder ab, und Mia kommt erst zur Hälfte der Zeit dazu, bringt aber "
            "für ihre letzten 2 Nächte noch ihren Bruder mit. "
            "Wie viel zahlt jede Person (Mia inklusive ihres Bruders)?")
        plan=_source_person_nights_plan(prompt)
        self.assertEqual(plan["total_nights"],14)
        self.assertEqual(plan["shares"],{"Eva":600,"Leo":300,"Mia":500})
        self.assertIsNone(_source_person_nights_plan(
            "Eine Berghütte kostet 900 €. Drei Gäste sind dort."))

    def test_rejects_executed_output_that_disagrees_with_original_person_nights(self):
        from neral_three_tier_runtime import (
            _source_person_nights_plan, _source_person_nights_output_ok)
        q=("Drei Freundinnen mieten gemeinsam eine Berghütte für 1.260 €. "
           "Nora übernachtet dort die vollen 6 Nächte, Kim reist nach 4 Nächten "
           "wieder ab, und Pia kommt erst zur Hälfte der Zeit dazu, bringt aber "
           "für ihre letzten 2 Nächte noch ihren Bruder mit. "
           "Pia inklusive ihres Bruders.")
        self.assertIsNotNone(_source_person_nights_plan(q))
        self.assertFalse(_source_person_nights_output_ok(
            q,"Nora=420; Kim=420; Pia=420"))
        self.assertFalse(_source_person_nights_output_ok(
            q,"Nora=252; Kim=420; Pia=588"))
        self.assertTrue(_source_person_nights_output_ok(
            q,"Nora=504; Kim=336; Pia=420"))
        self.assertIsNone(_source_person_nights_output_ok(
            "Warum glitzern Sterne?", "Nora=504; Kim=336; Pia=420"))


class LocaleAndOvernightHints(unittest.TestCase):
    def test_german_euro_thousands_separator_not_misread_as_decimal(self):
        from neral_three_tier_runtime import _source_german_euro_guidance
        hint=_source_german_euro_guidance(
            "Die Berghütte kostet 1.260 € für alle Gäste zusammen.")
        self.assertIn("1.260",hint)
        self.assertIn("1260",hint)
        self.assertNotIn("1.26 Euro",hint)
        self.assertEqual(_source_german_euro_guidance("200 € kosten die Zimmer."),"")

    def test_generic_person_night_hint_is_provided_before_helios_code(self):
        from neral_three_tier_runtime import _source_person_nights_hint
        hint=_source_person_nights_hint(
            "Eine Berghütte kostet 1.260 €. Nora bleibt 6 Nächte, "
            "Kim 4, Pia 3 und ihr Bruder 2 Nächte.")
        self.assertIn("Personennächte",hint)
        self.assertIn("Zusatzgäste",hint)
        self.assertIn("Gesamtpreis",hint)
        self.assertNotIn("504",hint)  # no answer hard-coding


class HeliosFallback(unittest.TestCase):
    def runtime(self,replies,tool):
        h=Model(replies)
        l=Model(["Hallo"])
        l.model="qwen3.5:4b"
        l.adapter_sha256=None
        return ThreeTierRuntime(l,h,execute_python=tool),h

    def test_unavailable_docker_returns_real_helios_answer_with_explicit_reason(self):
        rt,h=self.runtime(["```python\nprint(2+3)\n```","2 plus 3 ergibt 5."],
            lambda _:{"status":"UNAVAILABLE","error_code":"docker_unavailable","error_type":"SandboxUnavailable"})
        out=rt.answer(question(),"Luna")
        self.assertIn("2 plus 3 ergibt 5",out["text"])
        self.assertNotIn("Es liegt kein ausgeführtes Ergebnis vor",out["text"])
        self.assertEqual(out["sandbox_failure_code"],"docker_unavailable")
        self.assertFalse(out["sandbox_executed"])
        self.assertFalse(out["verified"])
        self.assertFalse(out["safe_to_promote"])
        self.assertIsNone(out["tool_used"])
        self.assertEqual(len(h.calls),2)

    def test_code_generation_error_does_not_abort_before_helios_fallback(self):
        rt,h=self.runtime([RuntimeError("HF token budget exhausted"),
                           "Direkte ungeprüfte Erklärung der Aufgabe."],
                          lambda _:self.fail("Never execute model Python on failed generation"))
        out=rt.answer(question(),"Luna")
        self.assertIn("Direkte ungeprüfte Erklärung",out["text"])
        self.assertFalse(out["sandbox_executed"])
        self.assertFalse(out["verified"])
        self.assertEqual(out["sandbox_failure_code"],"runtime_error")
        self.assertEqual(len(h.calls),2)

    def test_successful_docker_with_inconsistent_person_nights_is_not_claimed_as_success(self):
        source=("Drei Freundinnen mieten gemeinsam eine Berghütte für 1.260 €. "
            "Nora übernachtet dort die vollen 6 Nächte, Kim reist nach 4 Nächten "
            "wieder ab, und Pia kommt erst zur Hälfte der Zeit dazu, bringt aber "
            "für ihre letzten 2 Nächte noch ihren Bruder mit. "
            "Pia inklusive ihres Bruders.")
        seen=[]
        def wrong_program(_):
            seen.append(1)
            return {"status":"PASSED","output":"Nora=420; Kim=420; Pia=420"}
        rt,h=self.runtime(["\x60\x60\x60python\nprint('bad')\n\x60\x60\x60"]*3+
                          ["Ich kann das ohne belastbare Rechnung nur ungeprüft beantworten."],
                          wrong_program)
        out=rt.answer(question(source),"Luna",force_sandbox=True)
        self.assertEqual(len(seen),3)
        self.assertFalse(out["sandbox_executed"])
        self.assertEqual(out["sandbox_failure_code"],"source_mismatch")
        self.assertFalse(out["verified"])
        self.assertIn("ungeprüft",out["text"])

    def test_three_invalid_programs_trigger_real_model_fallback(self):
        rt,h=self.runtime(["```python\nraise ValueError(1)\n```"]*3+["Die Aufgabe ist noch nicht sicher gelöst."],
            lambda _:{"status":"FAILED","error_code":"runtime_error","error_type":"RuntimeError","traceback":"ValueError: 1"})
        out=rt.answer(question(),"Luna")
        self.assertIn("noch nicht sicher gelöst",out["text"])
        self.assertEqual(out["sandbox_failure_code"],"runtime_error")
        self.assertEqual(out["sandbox_failure_exception"],"ValueError")
        self.assertEqual(len(h.calls),4)
        self.assertEqual(out["sandbox_attempts"],3)
        self.assertFalse(out["sandbox_executed"])

    def test_model_fallback_failure_reports_unavailability_without_made_up_answer(self):
        rt,h=self.runtime(["```python\nprint(1)\n```"],
            lambda _:{"status":"UNAVAILABLE","error_code":"docker_unavailable"})
        out=rt.answer(question(),"Luna")
        self.assertTrue(out["text"].strip())
        self.assertNotIn("Es liegt kein ausgeführtes Ergebnis vor",out["text"])
        self.assertFalse(out.get("model_fallback_generated",True))
        self.assertFalse(out["sandbox_executed"])

    def test_invalid_fence_and_empty_output_have_distinct_error_codes(self):
        rt,h=self.runtime(["No python","No python","No python","Eine ungeprüfte Antwort."],
                          lambda _:self.fail("invalid source should not execute"))
        x=rt.answer(question(),"Luna")
        self.assertEqual(x["sandbox_failure_code"],"invalid_python_code")
        rt2,h2=self.runtime(["```python\nprint(1)\n```"]*3+["Antwort ohne Ausführung."],
                          lambda _:{"status":"FAILED","error_code":"empty_stdout"})
        y=rt2.answer(question(),"Luna")
        self.assertEqual(y["sandbox_failure_code"],"empty_stdout")


class DockerRecovery(unittest.TestCase):
    def test_daemon_checked_before_image_and_run_without_auto_start_on_linux(self):
        seen=[]
        def fake(cmd,**kwargs):
            seen.append(cmd)
            if cmd[:2]==["docker","info"]:
                return subprocess.CompletedProcess(cmd,0,b"ok",b"")
            if cmd[:3]==["docker","image","inspect"]:
                return subprocess.CompletedProcess(cmd,0,b"ok",b"")
            if cmd[:2]==["docker","run"]:
                return subprocess.CompletedProcess(cmd,0,b"5\n",b"")
            raise AssertionError("unrecognized command "+str(cmd))
        with patch.dict(os.environ,{"NERAL_CODE_SANDBOX_V2":"1"}),patch(
                "neral_tir_experimental.subprocess.run",side_effect=fake):
            got=_run_ephemeral_python("print(2+3)")
        self.assertEqual(got["status"],"PASSED")
        self.assertEqual(seen[0][:2],["docker","info"])
        self.assertEqual(seen[1][:3],["docker","image","inspect"])

    def test_missing_daemon_yields_machine_readable_error(self):
        def unavailable(cmd,**kwargs):
            return subprocess.CompletedProcess(cmd,1,b"",b"not running")
        with patch("neral_tir_experimental.subprocess.run",side_effect=unavailable), patch(
                "neral_docker_runtime._launch_existing_desktop",return_value=False):
            got=_run_ephemeral_python("print(2+3)")
        self.assertEqual(got["status"],"UNAVAILABLE")
        self.assertEqual(got["error_code"],"docker_unavailable")

    def test_python_syntax_error_carries_machine_readable_code(self):
        got=_run_ephemeral_python("if True print(1)")
        self.assertEqual(got["error_code"],"invalid_python_code")


if __name__=="__main__":
    unittest.main()
