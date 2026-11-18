"""V2.1 universal routing and explicit sandbox opt-in contracts (CPU only)."""
import io
import json
import threading
import unittest

from flask import Flask

from neral_three_tier_runtime import ThreeTierRuntime, _is_free_compute_task, register_three_tier_api


class FakeHF:
    def __init__(self, name, responses=()):
        self.model=name
        self.adapter_sha256="a"*64 if "9b" in name else None
        self.loaded=False
        self.responses=list(responses)
        self.calls=[]

    def respond_chat(self, turns, *, max_new_tokens, temperature=None, max_time_seconds=None):
        self.loaded=True
        self.calls.append((turns,temperature))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        return self.responses.pop(0)

    def unload(self):
        self.loaded=False


def fixture():
    low=FakeHF("qwen3.5:4b",["Hallo!"])
    high=FakeHF("qwen3.5:9b+neral-v2-full",[
        "\x60\x60\x60python\nprint(7)\n\x60\x60\x60","7",
        "\x60\x60\x60python\nprint(7)\n\x60\x60\x60","7"
    ])
    executed=[]
    runtime=ThreeTierRuntime(low,high,execute_python=lambda code:
        executed.append(code) or {"status":"PASSED","output":"7"})
    return runtime,low,high,executed


def turns(question):
    return [{"role":"system","content":"Local assistant"},
            {"role":"user","content":question}]


class UniversalRouter(unittest.TestCase):
    def test_noncanonical_quantitative_prompts_route_helios(self):
        cases=[
            "Lena, Tom und Mira teilen sich ein Ferienhaus für 840 €. Lena reist nach der Hälfte der Zeit ab, Tom bleibt die volle Woche und Mira bringt einen Gast für die letzten zwei Tage.",
            "Heute um halb acht los, zwei Stunden später ankommen. Wann?",
            "Welche Reihenfolge hat die kürzeste Wartezeit: A vor B, B nach C?",
            "Wenn die rote Box links steht, muss die blaue rechts sein. Geht das?",
            "Die Chance für rot ist höher als für blau. Welche Wahrscheinlichkeit?",
            "Die Tabelle hat Umsatz und Kosten. Wo liegt der Ausreißer?",
            "Welche Option ist größer oder dauert länger?",
            "Vor drei Tagen war ich dort; nächste Woche bin ich wieder da.",
        ]
        for question in cases:
            with self.subTest(question=question[:30]):
                self.assertTrue(_is_free_compute_task(question))
        for plain in ("Hallo NERAL!", "Erkläre mir den Begriff Magnetfeld.",
                      "Formuliere eine freundliche Einladung ohne Zahlen.",
                      "Was bedeutet Photosynthese?"):
            with self.subTest(plain=plain):
                self.assertFalse(_is_free_compute_task(plain))

    def test_manual_force_bypasses_source_tools_and_luna_for_hello(self):
        runtime,low,high,executed=fixture()
        result=runtime.answer(turns("Hallo!"),"Luna",force_sandbox=True)
        self.assertEqual(result["tier"],"Helios")
        self.assertEqual(result["tool_used"],"helios_code_sandbox")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(executed,["print(7)"])
        self.assertFalse(low.calls)
        self.assertFalse(result["verified"])

    def test_force_flag_strict_http_contract(self):
        runtime,low,high,executed=fixture()
        app=Flask("v21")
        app._neral_cpu_context=lambda turns: None
        register_three_tier_api(app,runtime,busy=threading.Lock())
        client=app.test_client()
        for invalid in ("true",1,None,[],{}):
            response=client.post("/api/neral/chat/three-tier",json={
                "tier":"Luna","prompt":"Hallo!","force_sandbox":invalid})
            self.assertEqual(response.status_code,400)
        response=client.post("/api/neral/chat/three-tier",json={
            "tier":"Luna","prompt":"Hallo!","force_sandbox":True,"history":[]})
        self.assertEqual(response.status_code,200,response.json)
        self.assertTrue(response.json["sandbox_executed"])


if __name__=="__main__":
    unittest.main()
