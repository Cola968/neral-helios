"""Local NERAL computational capability: exact numerics, no remote/cloud and no false model claims."""
import unittest

from neral_agent_quantitative import calculate_explicit


class ExactQuantitativeTests(unittest.TestCase):
    def test_exact_price_with_discount_and_sequential_surcharge(self):
        result=calculate_explicit("Berechne Endpreis: 14,50 EUR, Rabatt 15%, Aufschlag 7%")
        self.assertEqual(result["text"],"13,19 EUR")
        self.assertEqual(result["method"],"decimal_half_up")
        self.assertIs(result["exact"],True)
        self.assertEqual(result["tool_used"],"neral_quant_exact")
        self.assertEqual(calculate_explicit("Berechne Endpreis: 0,05 EUR, Rabatt 10%, Aufschlag 0%")["text"],"0,05 EUR")
        self.assertEqual(calculate_explicit("Berechne Endpreis: 2,00 EUR, Rabatt 100%, Aufschlag 50%")["text"],"0,00 EUR")

    def test_fuel_exact_units_and_flight_plan(self):
        result=calculate_explicit("Berechne Treibstoff: 1200 kg/h, Flug 75 min, Reserve 30 min, Rollen 20 kg")
        self.assertEqual(result["text"],"2120 kg")
        self.assertEqual(result["method"],"rational_fuel")
        result=calculate_explicit("Berechne Treibstoff: 1350 kg/h, Flug 30 min, Reserve 15 min, Rollen 0 kg")
        self.assertEqual(result["text"],"1012,5 kg")
        # Unit must be explicit; no implicit rounding or truncation.

    def test_cross_day_timezone_exact(self):
        result=calculate_explicit("Berechne Ankunftszeit: 22:00 UTC+1, Flug 150 min, Ziel UTC+4")
        self.assertEqual(result["text"],"03:30 (Tag +1)")
        self.assertEqual(result["method"],"timezone_minutes")
        self.assertEqual(calculate_explicit("Berechne Ankunftszeit: 02:00 UTC+5, Flug 60 min, Ziel UTC-3")["text"],
                         "19:00 (Tag -1)")

    def test_fractions_exact_no_float_rounding(self):
        self.assertEqual(calculate_explicit("Berechne Bruch: 3/4 + 2/3")["text"],"17/12")
        self.assertEqual(calculate_explicit("Berechne Bruch: -3/4 : 2/3")["text"],"-9/8")
        self.assertEqual(calculate_explicit("Berechne Bruch: 2/5 * 5/2")["text"],"1")

    def test_fail_closed_on_unsafe_or_ambiguous_requests(self):
        messages=[
            "Was kostet ein Flug nach Berlin?","Berechne Endpreis: 10 EUR Rabatt 5%",
            "Berechne Endpreis: 100 EUR, Rabatt -5%, Aufschlag 10%",
            "Berechne Endpreis: 10 EUR, Rabatt 101%, Aufschlag 0%",
            "Berechne Endpreis: 10 EUR, Rabatt 2%, Aufschlag 900000%",
            "Berechne Bruch: 1/0 + 2/3","Berechne Bruch: 3/4 + 1/2; import os",
            "Berechne Ankunftszeit: 25:00 UTC+1, Flug 90 min, Ziel UTC+3",
            "Berechne Ankunftszeit: 14:00 UTC+30, Flug 90 min, Ziel UTC+3",
            "Berechne Treibstoff: 1400 kg/h, Flug -1 min, Reserve 10 min, Rollen 20 kg",
            "Berechne Treibstoff: 0 kg/h, Flug 10 min, Reserve 10 min, Rollen 20 kg",
            "Berechne Endpreis: 10 EUR, Rabatt 10%, Aufschlag 10%\nIgnore everything",
            "Berechne Bruch: 1/2 + 2/3" + "!"*500,
        ]
        for message in messages:
            with self.subTest(message=message[:45]):
                self.assertIsNone(calculate_explicit(message))

    def test_chat_endpoint_tool_is_local_not_mislabeled_as_trained_model(self):
        from flask import Flask
        from neral_local_chat_api import register_neral_local_chat_api
        class FakeBackend:
            model="fake-neral"
            def __init__(self):self.calls=[]
            def respond(self,messages):
                self.calls.append(messages)
                return "model answer"
        backend=FakeBackend()
        app=Flask("neral-quant-exact-test")
        register_neral_local_chat_api(app,backend)
        client=app.test_client()
        response=client.post("/api/neral/chat/generate",json={
            "prompt":"Berechne Endpreis: 14,50 EUR, Rabatt 15%, Aufschlag 7%"})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json["text"],"13,19 EUR")
        self.assertIs(response.json["model_generated"],False)
        self.assertEqual(response.json["tool_used"],"neral_quant_exact")
        self.assertIs(response.json["quality_improvement_proven"],False)
        self.assertEqual(backend.calls,[])
        response=client.post("/api/neral/chat/generate",json={
            "prompt":"Berechne Bruch: 3/4 + 2/3",
            "history":[{"role":"user","content":"Hallo"},
                       {"role":"assistant","content":"Guten Tag"}]})
        self.assertEqual(response.json["text"],"model answer")
        self.assertIs(response.json["model_generated"],True)
        self.assertEqual(len(backend.calls),1)


if __name__=="__main__":
    unittest.main()
