"""Regression: exact everyday math/scheduling auto-verify from Luna in real chat contract."""
from __future__ import annotations

import unittest
from neral_three_tier_runtime import ThreeTierRuntime

MATH = "Teile 3.840 € brutto (inkl. 19 % MwSt.) exakt in Netto und Steuer auf und verteile den Nettobetrag im Verhältnis 5 : 3 : 2 auf drei Projektphasen."
PLAN = ("Vier Aufgaben A (3 Std.), B (4 Std.), C (2 Std.) und D (5 Std.) "
        "müssen auf zwei Personen verteilt werden. A muss vor B abgeschlossen sein, "
        "B und C dürfen nicht gleichzeitig laufen, und die Gesamtdauer darf 8 "
        "Stunden nicht überschreiten. Erstelle einen gültigen Zeitplan.")
GOOD = "322689|61311|161344|96807|64538"


class FakeModel:
    def __init__(self, name, sha=None):
        self.model=name
        self.adapter_sha256=sha
        self.loaded=False
        self.calls=[]
    def respond_chat(self, turns, *, max_new_tokens):
        self.calls.append(max_new_tokens)
        self.loaded=True
        return "Grenze: Ein lokales Modell kann keinen mathematischen Beweis liefern.Plan: ungeprüft"
    def unload(self): self.loaded=False


def runtime(docker=None):
    luna=FakeModel("qwen3.5:4b")
    helios=FakeModel("qwen3.5:9b+neral-v2-full", "a"*64)
    return (ThreeTierRuntime(luna, helios, execute_python=docker or
                             (lambda code: {"status":"UNAVAILABLE"})),luna,helios)


def turns(prompt, history=False):
    data=[{"role":"system","content":"System"}]
    if history:
        data += [{"role":"user","content":"Hallo"},{"role":"assistant","content":"Hallo!"}]
    data.append({"role":"user","content":prompt})
    return data


class EverydayLiveChat(unittest.TestCase):
    def test_luna_auto_uses_docker_python_with_cent_exact_totals(self):
        observed=[]
        def docker(code):
            observed.append(code)
            return {"status":"PASSED","output":GOOD}
        rt,luna,helios=runtime(docker)
        result=rt.answer(turns(MATH,history=True),"Luna")
        self.assertTrue(result["verified"])
        self.assertTrue(result["safe_to_promote"])
        self.assertFalse(result["model_generated"])
        self.assertEqual(result["tier"],"Terra")
        self.assertEqual(result["auto_promoted_from"],"Luna")
        self.assertEqual(result["proof_backend"],"trusted_source_arithmetic")
        self.assertIn("3.226,89 €",result["text"])
        self.assertIn("613,11 €",result["text"])
        self.assertIn("1.613,44 €",result["text"])
        self.assertIn("968,07 €",result["text"])
        self.assertIn("645,38 €",result["text"])
        self.assertNotIn("Grenze:",result["text"])
        self.assertEqual(len(observed),1)
        self.assertIn("print(",observed[0])
        self.assertEqual(luna.calls,[])
        self.assertEqual(helios.calls,[])

    def test_docker_wrong_output_does_not_get_badge(self):
        rt,_,_=runtime(lambda code: {"status":"PASSED","output":"322690|613110|0|0|0"})
        result=rt.answer(turns(MATH),"Luna")
        self.assertFalse(result["verified"])
        self.assertNotEqual(result.get("model_origin"),"local_source_verified_tools_only")
        self.assertNotIn("Verifiziert",result["text"])

    def test_docker_offline_never_promotes_untested_answer(self):
        rt,_,_=runtime()
        result=rt.answer(turns(MATH),"Luna")
        self.assertFalse(result["verified"])
        self.assertNotIn("3.226,89 €",result["text"])

    def test_luna_auto_uses_real_z3_and_validated_witness(self):
        rt,luna,helios=runtime()
        result=rt.answer(turns(PLAN,history=True),"Luna")
        self.assertTrue(result["verified"])
        self.assertTrue(result["safe_to_promote"])
        self.assertFalse(result["model_generated"])
        self.assertEqual(result["tier"],"Helios")
        self.assertEqual(result["auto_promoted_from"],"Luna")
        self.assertEqual(result["proof_backend"],"z3_sat_witness")
        for name in "ABCD":
            self.assertRegex(result["text"], rf"\b{name}\s+\d+[–-]\d+")
        self.assertIn("Person 1",result["text"])
        self.assertIn("Person 2",result["text"])
        self.assertNotIn("Grenze:",result["text"])
        self.assertEqual(luna.calls,[])
        self.assertEqual(helios.calls,[])

    def test_unrecognized_extra_constraint_must_not_falsely_verify(self):
        changed=PLAN.replace("Erstelle einen gültigen Zeitplan.",
                             "D darf außerdem erst nach B starten. Erstelle einen gültigen Zeitplan.")
        rt,_,_=runtime()
        result=rt.answer(turns(changed),"Luna")
        self.assertFalse(result["verified"])
        self.assertNotIn("auto_promoted_from",result)

    def test_unrecognized_generic_luna_remains_unpromoted(self):
        rt,luna,helios=runtime()
        result=rt.answer(turns("Was ist das Wetter in Berlin?"),"Luna")
        self.assertEqual(result["tier"],"Luna")
        self.assertFalse(result["verified"])
        self.assertEqual(luna.calls,[96])
        self.assertEqual(helios.calls,[])

    def test_malformed_disclaimer_removed_without_false_verification(self):
        rt,_,_=runtime()
        response=rt.answer(turns("Erstelle einen Reiseplan"),"Luna")
        self.assertFalse(response["verified"])
        self.assertNotIn("Grenze: Ein lokales Modell",response["text"])
        self.assertNotIn("liefern.Plan",response["text"])
        self.assertTrue(response["text"].startswith("Plan:"))


    def test_energy_cost_uses_python_sandbox_and_cent_exact_result(self):
        prompt=("Ein Server verbraucht 340 Watt im Dauerbetrieb. Wie viele Kilowattstunden "
                "sind das in 365 Tagen und was kostet das bei 32,4 Cent pro kWh auf den Cent genau?")
        observed=[]
        def sandbox(code):
            observed.append(code)
            return {"status":"PASSED","output":"2978.4|965.00"}
        rt,luna,helios=runtime(sandbox)
        result=rt.answer(turns(prompt),"Luna")
        self.assertTrue(result["verified"])
        self.assertEqual(result["tool_used"],"python_sandbox")
        self.assertEqual(result["tier"],"Terra")
        self.assertIn("2.978,4 kWh",result["text"])
        self.assertIn("965,00 €",result["text"])
        self.assertEqual(len(observed),1)
        self.assertEqual((luna.calls,helios.calls),([],[]))

    def test_energy_cost_wrong_python_output_not_verified(self):
        prompt=("Ein Server verbraucht 340 Watt im Dauerbetrieb. Wie viele Kilowattstunden "
                "sind das in 365 Tagen und was kostet das bei 32,4 Cent pro kWh auf den Cent genau?")
        rt,_,_=runtime(lambda _:{"status":"PASSED","output":"2978.4|965.01"})
        result=rt.answer(turns(prompt),"Luna")
        self.assertFalse(result["verified"])
        self.assertNotIn("965,00 €",result["text"])

    def test_three_boxes_uniquely_verified_by_z3(self):
        prompt=("Drei Kisten (Rot, Blau, Grün) wiegen zusammen 45 kg. "
                "Blau ist doppelt so schwer wie Rot, und Grün ist 5 kg schwerer "
                "als Blau. Wie schwer ist jede einzelne Kiste?")
        rt,luna,helios=runtime()
        result=rt.answer(turns(prompt),"Luna")
        self.assertTrue(result["verified"])
        self.assertEqual(result["tool_used"],"z3_solver")
        self.assertEqual(result["tier"],"Helios")
        self.assertEqual(result["witness"],{"Rot":"8","Blau":"16","Grün":"21"})
        self.assertNotIn("Grenze:",result["text"])
        self.assertEqual((luna.calls,helios.calls),([],[]))

    def test_puzzle_unknown_extra_constraint_cannot_be_full_source_verified(self):
        prompt=("Drei Kisten (Rot, Blau, Grün) wiegen zusammen 45 kg. "
                "Blau ist doppelt so schwer wie Rot, und Grün ist 5 kg schwerer als Blau. "
                "Zusätzlich ist Rot größer als Grün. Wie schwer ist jede Kiste?")
        rt,_,_=runtime()
        result=rt.answer(turns(prompt),"Luna")
        self.assertFalse(result["verified"])

    def test_unrecognized_complex_question_escalates_to_unverified_helios(self):
        rt,luna,helios=runtime()
        result=rt.answer(turns("Wie viel kostet das bei 50 kWh und 30 Cent?"),"Luna")
        self.assertFalse(result["verified"])
        self.assertEqual(result["tier"],"Helios")
        self.assertEqual(result.get("auto_escalated_from"),"Luna")
        self.assertEqual(luna.calls,[])
        self.assertEqual(helios.calls,[768])


if __name__=="__main__":
    unittest.main()
