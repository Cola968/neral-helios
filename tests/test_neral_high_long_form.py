"""CPU-only regression tests for HIGH long-form derivation budget and bounded continuation."""
import json
import unittest
from unittest.mock import patch

from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend

KEPLER = [
    {"role": "system", "content": "Du bist NERAL."},
    {"role": "user", "content": (
        "Leite aus dem Newtonschen Gravitationsgesetz und der Bewegungsgleichung "
        "das dritte Keplersche Gesetz für elliptische Bahnen her. "
        "Begründe alle wesentlichen Zwischenschritte."
    )},
]


class Reply:
    def __init__(self, packet):
        self.packet = packet

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit):
        return json.dumps(self.packet).encode("utf-8")


class Wire:
    def __init__(self, *packets):
        self.packets = list(packets)
        self.calls = []
        self.timeouts = []

    def open(self, request, timeout=0):
        self.calls.append(json.loads(request.data))
        self.timeouts.append(timeout)
        return Reply(self.packets.pop(0))


def packet(reason, content):
    return {"done": True, "done_reason": reason, "message": {"content": content}}


class LongFormHighTests(unittest.TestCase):
    def model(self):
        return OllamaLocalBackend("qwen3.5:9b", inference_profile="measured_nonthinking_2048")

    def test_kepler_derivation_uses_long_high_budget_and_stays_unverified(self):
        wire = Wire(packet("stop", "Für die Ellipse folgt T²=4π²a³/GM."))
        model = self.model()
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            answer = model.respond_high_final(KEPLER)
        self.assertIn("T²=", answer)
        self.assertEqual(len(wire.calls), 1)
        self.assertEqual(wire.calls[0]["options"]["num_predict"], 4096)
        self.assertEqual(wire.calls[0]["options"]["num_ctx"], 8192)
        self.assertEqual(wire.timeouts, [180.0])
        self.assertFalse(wire.calls[0]["think"])
        self.assertTrue(model.last_inference_diagnostic["long_form_budget_used"])
        self.assertFalse(model.last_inference_diagnostic["prose_continuation_succeeded"])

    def test_long_high_response_can_continue_once_without_losing_partial(self):
        part1 = "Schritt 1: Aus der Zentralkraft folgt der Flächensatz.\n"
        part2 = "Schritt 2: Aus der Bahngeometrie folgt T² ∝ a³."
        wire = Wire(packet("length", part1), packet("stop", part2))
        model = self.model()
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            answer = model.respond_high_final(KEPLER)
        self.assertEqual(answer, part1 + part2)
        self.assertEqual(len(wire.calls), 2)
        self.assertEqual(wire.calls[0]["options"]["num_predict"], 4096)
        self.assertEqual(wire.calls[1]["options"]["num_predict"], 1536)
        self.assertEqual(wire.calls[1]["options"]["num_ctx"], 8192)
        self.assertEqual(wire.timeouts, [180.0, 90.0])
        self.assertEqual(wire.calls[1]["options"]["temperature"], 0)
        self.assertFalse(wire.calls[1]["think"])
        self.assertEqual(wire.calls[1]["messages"][-2], {"role": "assistant", "content": part1})
        self.assertEqual(wire.calls[1]["messages"][-1]["role"], "user")
        self.assertTrue(model.last_inference_diagnostic["prose_continuation_succeeded"])
        self.assertNotIn("Zentralkraft", str(model.inference_diagnostic_events))
        self.assertNotIn(KEPLER[-1]["content"], str(model.inference_diagnostic_events))

    def test_second_length_fails_closed_no_partial_and_no_third_call(self):
        wire = Wire(packet("length", "Schritt 1: Aus Newtons Zentralkraft folgt ein erhaltener Drehimpuls.\n"),
                    packet("length", "Schritt 2: noch nicht fertig."))
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            with self.assertRaises(OllamaCompletionError) as raised:
                self.model().respond_high_final(KEPLER)
        self.assertEqual(raised.exception.reason, "token_limit")
        self.assertEqual(len(wire.calls), 2)
        self.assertNotIn("Schritt", str(raised.exception))

    def test_raw_qwen_keeps_2048_and_one_call_even_on_kepler(self):
        wire = Wire(packet("length", "Noch nicht fertig."))
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            with self.assertRaises(OllamaCompletionError):
                self.model().respond_chat(KEPLER)
        self.assertEqual(len(wire.calls), 1)
        self.assertEqual(wire.calls[0]["options"]["num_predict"], 2048)
        self.assertEqual(wire.calls[0]["options"]["num_ctx"], 4096)
        self.assertEqual(wire.timeouts, [90.0])

    def test_regular_high_query_keeps_original_budget(self):
        wire = Wire(packet("stop", "Eine Turbine verdichtet Luft."))
        turns = [KEPLER[0], {"role": "user", "content": "Erkläre den Aufbau eines Triebwerks."}]
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            self.model().respond_high_final(turns)
        self.assertEqual(len(wire.calls), 1)
        self.assertEqual(wire.calls[0]["options"]["num_predict"], 2048)
        self.assertEqual(wire.calls[0]["options"]["num_ctx"], 4096)

    def test_final_answer_task_keeps_existing_numeric_rescue(self):
        turns = [KEPLER[0], {"role": "user", "content": (
            "Beweise die Behauptung. Gib FINAL_ANSWER: <integer> aus."
        )}]
        wire = Wire(packet("length", "Die Rechnung ist fast fertig.\n"),
                    packet("stop", "FINAL_ANSWER: 42"))
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            answer = self.model().respond_high_final(turns)
        self.assertEqual(answer, "FINAL_ANSWER: 42")
        self.assertEqual(wire.calls[0]["options"]["num_predict"], 2048)
        self.assertEqual(wire.calls[1]["options"]["num_predict"], 64)


    def test_high_http_marks_continuation_as_unverified_and_counts_both_calls(self):
        from scripts.neral_same_base_live import BASE_URL, _high_client
        model = self.model()
        wire = Wire(packet("length", "Aus der Gleichung der Bewegung folgt die konstante Flächengeschwindigkeit.\\n"),
                    packet("stop", "Nach Kürzen der Exzentrizität gilt T² ∝ a³."))
        with patch("neral_agent_autonomy.local_only_opener", return_value=wire):
            client, headers = _high_client(model)
            response = client.post(
                "/api/neral/chat/think",
                json={"prompt": KEPLER[-1]["content"]},
                base_url=BASE_URL, headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["calls_used"], 2)
        self.assertTrue(result["completion_continuation_used"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"], "continued_unverified")
        self.assertEqual(len(wire.calls), 2)


if __name__ == "__main__":
    unittest.main()
