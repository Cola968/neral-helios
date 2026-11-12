"""CPU-only RED contracts for opt-in model-selected Python logic exploration.

No benchmark phrases in routing, no GPU usage and no untrusted code on host.
"""
import itertools
import json
import unittest
from unittest.mock import patch

from neral_tir_experimental import run_experimental_tir
from neral_tir_regression_guard import choose_guarded_tir_reply
from neral_agent_autonomy import OllamaLocalBackend


TASK = (
    "Ein fünfstelliger Code verwendet 1, 2, 3, 4 und 5 genau einmal. "
    "Die 2 steht vor der 4, nicht direkt davor. Zwischen der 1 und 5 "
    "steht genau eine Ziffer. Die 3 steht unmittelbar rechts von der 5. "
    "Die 4 steht nicht am Rand. Bestimme alle gültigen Codes."
)


def messages(text=TASK):
    return [{"role": "system", "content": "NERAL"},
            {"role": "user", "content": text}]


def reference_solutions():
    result = []
    for seq in itertools.permutations("12345"):
        p = {digit: seq.index(digit) for digit in seq}
        if (p["4"] - p["2"] >= 2 and
            abs(p["1"] - p["5"]) == 2 and
            p["3"] == p["5"] + 1 and
            p["4"] in (1, 2, 3)):
            result.append("".join(seq))
    return result


class LogicPythonRoutingCPU(unittest.TestCase):
    def test_independent_reference_enumeration_is_unique(self):
        self.assertEqual(reference_solutions(), ["21453"])

    def test_optin_directs_first_decision_to_short_python_not_rejected_schema(self):
        code = ("import itertools, json\n"
                "solutions = []\n"
                "for p in itertools.permutations('12345'):\n"
                "    q = {v: p.index(v) for v in p}\n"
                "    if q['4'] - q['2'] >= 2 and abs(q['1']-q['5']) == 2 and "
                "q['3'] == q['5'] + 1 and q['4'] in (1,2,3):\n"
                "        solutions.append(''.join(p))\n"
                "print(json.dumps({'solutions': solutions}))")
        proposals = []
        tool_invocations = []
        def proposal(turns):
            proposals.append(turns)
            return "NERAL_EXECUTE_JSON: " + json.dumps(
                {"tool": "python_sandbox", "code": code})
        def runner(tool, source):
            tool_invocations.append((tool, source))
            return {"status": "PASSED",
                    "output": json.dumps({"solutions": reference_solutions()}),
                    "stderr": "", "verification_scope": "execution_only"}
        def model(turns):
            # The original raw baseline is only requested after the guard.
            self.assertIn("SANDBOX_OBSERVATION", turns[-1]["content"])
            return "NERAL_FINAL_TEXT: 21453"
        out = run_experimental_tir(
            messages(), model, tool_runner=runner,
            extract_constraints=lambda _: self.fail("No narrow schema on opt-in"),
            python_logic_routing=True, tool_decide=proposal)
        self.assertEqual(len(proposals), 1)
        self.assertIn("itertools", proposals[0][0]["content"])
        self.assertEqual(tool_invocations, [("python_sandbox", code)])
        self.assertEqual(out["text"], "21453")
        self.assertFalse(out["verified"])
        self.assertEqual(out["verification_status"], "tool_observed_unverified")
        self.assertEqual(out["tool_calls_succeeded"], 1)
        self.assertNotIn("generated_code", out)
        assert out["calls_used"] == 2

        guard = choose_guarded_tir_reply(out, messages(), lambda _: "RAW QWEN")
        self.assertEqual(guard["text"], "RAW QWEN")
        self.assertFalse(guard["verified"])
        self.assertEqual(guard["selection_policy"], "raw_qwen_due_to_unverified_tir")

    def test_zero_multiple_malformed_and_non_python_observations_fall_back(self):
        for observation in (
            '{"solutions":[]}', '{"solutions":["21453","31452"]}',
            '{"solutions":["21453","21453"]}', '{"solutions":[true]}',
            '{"solutions":["21453"],"verified":true}', '21453', '{',
        ):
            with self.subTest(observation=observation):
                replies=[]
                def model(turns):
                    replies.append(turns)
                    self.assertEqual(turns, messages())
                    return "RAW QWEN"
                out=run_experimental_tir(
                    messages(), model, python_logic_routing=True,
                    tool_decide=lambda _:'NERAL_EXECUTE_JSON: '+json.dumps(
                        {"tool":"python_sandbox", "code":"solutions=[]\nprint(1)"}),
                    tool_runner=lambda *_:{"status":"PASSED","output":observation})
                self.assertEqual(out["verification_status"],"base_fallback_unverified")
                self.assertEqual(out["text"],"RAW QWEN")
                self.assertEqual(len(replies),1)

    def test_optin_rejects_tool_failures_and_never_uses_host_python(self):
        out = run_experimental_tir(
            messages(), lambda _: "RAW QWEN", python_logic_routing=True,
            tool_decide=lambda _:'NERAL_EXECUTE_JSON: '+json.dumps(
                {"tool":"python_sandbox", "code":"solutions=[]"}),
            tool_runner=lambda *_:{"status":"UNAVAILABLE"})
        self.assertEqual(out["text"], "RAW QWEN")
        self.assertFalse(out["verified"])
        with self.assertRaises(ValueError):
            run_experimental_tir(
                messages(), lambda _: "RAW QWEN",
                python_logic_routing=True)

    def test_default_tir_path_unchanged_without_optin(self):
        calls=[]
        def model(turns):
            calls.append(turns)
            return "NERAL_FINAL_TEXT: Guten Tag"
        result = run_experimental_tir(messages("Hallo"),
                                      model, tool_runner=lambda *_:None)
        self.assertEqual(result["text"], "Guten Tag")
        self.assertEqual(len(calls), 1)

    def test_dedicated_decision_budget_is_smaller_and_context_is_already_4096(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _):
                return json.dumps({"done":True,"done_reason":"stop",
                    "message":{"content":"NERAL_FINAL_TEXT: UNVERIFIED"}}).encode()
        class Local:
            def __init__(self): self.packets=[]
            def open(self, req, timeout):
                self.packets.append(json.loads(req.data))
                return Response()
        local=Local()
        with patch("neral_agent_autonomy.local_only_opener", return_value=local):
            backend=OllamaLocalBackend("qwen3.5:9b",
                                      inference_profile="measured_nonthinking_2048")
            backend.respond_tir_code(messages())
            backend.respond_chat(messages())
        self.assertEqual(local.packets[0]["think"], False)
        self.assertEqual(local.packets[0]["options"]["num_predict"],768)
        self.assertEqual(local.packets[0]["options"]["num_ctx"],4096)
        self.assertEqual(local.packets[1]["options"]["num_predict"],2048)
        self.assertEqual(local.packets[1]["options"]["num_ctx"],4096)


if __name__ == "__main__":
    unittest.main()
