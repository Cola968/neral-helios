"""CPU-only structured adaptive Ollama proposal transport contracts."""
import json
import unittest

from neral_adaptive_ollama import make_local_adaptive_proposer


ORIGINAL = [
    {"role": "system", "content": "NERAL private local identity"},
    {"role": "user", "content": "Count permutations of numbers 1 through 4."},
]


def signal(round_number=0, code=None):
    return {"preferred_strategy": "PYTHON_SANDBOX",
            "allowed_strategies": ("PYTHON_SANDBOX", "DIRECT"),
            "repair_round": round_number, "feedback_code": code}


class AdaptiveProposalTransportCPU(unittest.TestCase):
    def test_one_installed_local_model_call_contains_strategy_and_tool(self):
        calls = []
        def fake_send(payload):
            calls.append(payload)
            return {"done": True, "done_reason": "stop",
                    "message": {"content": json.dumps({
                        "strategy": "PYTHON_SANDBOX", "code": "print(24)"})}}
        model = make_local_adaptive_proposer(
            "neral-qwen35-qlora-v1:latest", transport=fake_send)
        plan = model(ORIGINAL, signal())
        self.assertEqual(plan, {"strategy": "PYTHON_SANDBOX", "code": "print(24)"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "neral-qwen35-qlora-v1:latest")
        self.assertFalse(calls[0]["stream"])
        self.assertFalse(calls[0]["think"])
        self.assertEqual(calls[0]["keep_alive"], "5m")
        self.assertEqual(calls[0]["options"]["temperature"], 0)
        self.assertLessEqual(calls[0]["options"]["num_predict"], 1024)
        self.assertEqual(calls[0]["messages"][-1], ORIGINAL[-1])
        self.assertIn("PYTHON_SANDBOX", calls[0]["messages"][0]["content"])

    def test_repair_uses_same_qlora_no_extra_planner_model(self):
        seen = []
        def send(payload):
            seen.append(payload)
            return {"done": True, "done_reason": "stop",
                    "message": {"content": '{"strategy":"DIRECT"}'}}
        proposer = make_local_adaptive_proposer(
            "neral-qwen35-qlora-v1:latest", transport=send)
        for i in range(3):
            proposer(ORIGINAL, signal(i, "python_runtime_failed" if i else None))
        self.assertEqual(len(seen), 3)
        self.assertEqual({row["model"] for row in seen},
                         {"neral-qwen35-qlora-v1:latest"})
        self.assertIn("python_runtime_failed", seen[-1]["messages"][0]["content"])
        self.assertNotIn("SECRET_TRACEBACK", str(seen))

    def test_no_unapproved_models_or_transport(self):
        for name in ("qwen3.5:9b", "http://remote", "../other", "",
                     "neral-qwen35-qlora-v1:other"):
            with self.assertRaises(ValueError):
                make_local_adaptive_proposer(name, transport=lambda _: {})
        with self.assertRaises(ValueError):
            make_local_adaptive_proposer("neral-qwen35-qlora-v1:latest",
                                        transport="invalid")

    def test_untrusted_signal_values_rejected_without_request(self):
        calls = []
        p = make_local_adaptive_proposer(
            "neral-qwen35-qlora-v1:latest",
            transport=lambda payload: calls.append(payload) or {})
        for invalid in (
            {"preferred_strategy": "UNKNOWN", "allowed_strategies": (),
             "repair_round": 0, "feedback_code": None},
            {"preferred_strategy": "PYTHON_SANDBOX",
             "allowed_strategies": ("PYTHON_SANDBOX", "DIRECT"),
             "repair_round": 3, "feedback_code": None},
            {"preferred_strategy": "PYTHON_SANDBOX",
             "allowed_strategies": ("PYTHON_SANDBOX", "DIRECT"),
             "repair_round": 1, "feedback_code": "SOURCE PRIVATE"},
        ):
            with self.assertRaises(ValueError):
                p(ORIGINAL, invalid)
        self.assertEqual(calls, [])

    def test_only_two_turn_inputs_allowed(self):
        called = []
        p = make_local_adaptive_proposer(
            "neral-qwen35-qlora-v1:latest",
            transport=lambda arg: called.append(arg) or {})
        with self.assertRaises(ValueError):
            p(ORIGINAL + [{"role": "user", "content": "extra"}], signal())
        self.assertEqual(called, [])

    def test_rejects_invalid_or_incomplete_envelope(self):
        for response in (
            {"done": False, "done_reason": "stop",
             "message": {"content": '{"strategy":"DIRECT"}'}},
            {"done": True, "done_reason": "length",
             "message": {"content": '{"strategy":"DIRECT"}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": ""}},
            {"done": True, "done_reason": "stop", "message": {}},
            {"done": True, "done_reason": "stop",
             "message": {"content": '{"strategy":"DIRECT","strategy":"PYTHON_SANDBOX"}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": '{"strategy":"DIRECT","verified":true}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": '{"strategy":"PYTHON_SANDBOX"}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": '{"strategy":"DIRECT","code":"print(1)"}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": '{"strategy":"SHELL","code":"print(1)"}'}},
            {"done": True, "done_reason": "stop",
             "message": {"content": "{}"}},
        ):
            with self.subTest(response=response):
                p = make_local_adaptive_proposer(
                    "neral-qwen35-qlora-v1:latest", transport=lambda _: response)
                with self.assertRaises(ValueError):
                    p(ORIGINAL, signal())

    def test_source_never_gets_removed_for_model(self):
        calls = []
        p = make_local_adaptive_proposer(
            "neral-qwen35-qlora-v1:latest", transport=lambda packet:
            calls.append(packet) or {
                "done": True, "done_reason": "stop",
                "message": {"content": '{"strategy":"DIRECT"}'}})
        p(ORIGINAL, signal())
        self.assertEqual(calls[0]["messages"][1]["content"], ORIGINAL[1]["content"])


if __name__ == "__main__":
    unittest.main()
