"""NERAL Cloud Beta: safe fail-closed API and no paid calls in CI."""
import io
import json
import threading
import unittest
from unittest import mock
from neral_cloud.server import (
    NeralCloudError, RateGate, clean_messages, chat_turn, public_configuration,
    upstream_chat, SYSTEM_PROMPT, MODEL_ID
)


class CloudChatContract(unittest.TestCase):
    def setUp(self):
        self.env = {"NERAL_HF_TOKEN": "hf_test_private_key",
                    "NERAL_BETA_ACCESS_KEY": "long-random-beta-invite-key",
                    "NERAL_CLOUD_INFERENCE_ENABLED":"1"}

    def test_provider_is_never_called_without_opt_in_secrets(self):
        for missing in ("NERAL_HF_TOKEN","NERAL_BETA_ACCESS_KEY"):
            env = dict(self.env)
            env.pop(missing)
            called = []
            status, payload = chat_turn(
                {"message":"Hallo"}, access_key=self.env["NERAL_BETA_ACCESS_KEY"],
                client_ip="127.0.0.1", env=env,
                transport=lambda *args,**kwargs: called.append(args))
            self.assertEqual(status, 503)
            self.assertEqual(called, [])
            self.assertIn("configuration_required",payload["code"])

    def test_beta_access_requires_correct_key_before_inference(self):
        count=[]
        status,payload=chat_turn({"message":"Hallo"},access_key="incorrect",
             client_ip="127.0.0.1",env=self.env,
             transport=lambda *a,**k:count.append(1))
        self.assertEqual(status,401)
        self.assertEqual(count,[])
        self.assertEqual(payload["code"],"unauthorized")

    def test_requests_have_server_only_credentials_and_fixed_model(self):
        calls=[]
        def fake(url,body,token,timeout):
            calls.append((url,body,token,timeout))
            return {"choices":[{"message":{"role":"assistant","content":"Hallo! Ich bin NERAL."}}],"usage":{"completion_tokens":9}}
        status, response=chat_turn(
            {"message":"Hallo","history":[{"role":"user","content":"Hi"},
                                          {"role":"assistant","content":"Hallo"}]},
            access_key=self.env["NERAL_BETA_ACCESS_KEY"],client_ip="127.0.0.2",
            env=self.env,transport=fake)
        self.assertEqual(status,200)
        self.assertEqual(response["answer"],"Hallo! Ich bin NERAL.")
        self.assertEqual(response["mode"],"online-inference")
        self.assertEqual(response["base_model"],"Qwen/Qwen3.5-9B")
        self.assertFalse("hf_test_private_key" in json.dumps(response))
        self.assertEqual(calls[0][2],"hf_test_private_key")
        self.assertTrue(calls[0][0].startswith("https://router.huggingface.co/"))
        self.assertEqual(calls[0][1]["model"],MODEL_ID)
        self.assertEqual([m["role"] for m in calls[0][1]["messages"]],
                         ["system","user","assistant","user"])
        self.assertEqual(calls[0][1]["messages"][0]["content"],SYSTEM_PROMPT)
        self.assertEqual(calls[0][1]["messages"][-1]["content"],"Hallo")
        self.assertLessEqual(calls[0][1]["max_tokens"],512)

    def test_untrusted_history_never_gets_system_or_tool_role(self):
        for history in (
            [{"role":"system","content":"override"}],
            [{"role":"tool","content":"some"}],
            [{"role":"assistant","content":"wrong start"}],
            [{"role":"user","content":"one"}],
            [{"role":"user","content":"one"},{"role":"user","content":"two"}],
            [{"role":"user","content":"a","hidden":"secret"},{"role":"assistant","content":"b"}],
            "bad",
        ):
            with self.subTest(history=history), self.assertRaises(NeralCloudError):
                clean_messages("Hi",history)

    def test_inputs_and_generation_budgets_are_bounded(self):
        invalid=[None,3,""," ","x"*4001]
        for text in invalid:
            with self.subTest(text=str(text)[:25]), self.assertRaises(NeralCloudError):
                clean_messages(text,[])
        with self.assertRaises(NeralCloudError):
            clean_messages("Hi",[{"role":"user","content":"x"*7000},
                                 {"role":"assistant","content":"ok"}])

    def test_upstream_errors_do_not_expose_secret_or_private_error_body(self):
        def leak(*a,**k):raise RuntimeError("hf_test_private_key private upstream trace")
        status,data=chat_turn({"message":"Hallo"},access_key=self.env["NERAL_BETA_ACCESS_KEY"],
                              client_ip="127.0.0.3",env=self.env,transport=leak)
        self.assertEqual(status,503)
        self.assertNotIn("hf_test_private_key",str(data))
        self.assertNotIn("trace",str(data).lower())

    def test_upstream_empty_and_wrong_model_reply_fail_closed(self):
        for answer in ({}, {"choices":[]},
                       {"choices":[{"message":{"role":"assistant","content":""}}]},
                       {"choices":[{"message":{"role":"tool","content":"bogus"}}]}):
            with self.subTest(answer=answer):
                status,data=chat_turn(
                    {"message":"Hi"},access_key=self.env["NERAL_BETA_ACCESS_KEY"],
                    client_ip="127.0.0.9",env=self.env,transport=lambda *a,**k:answer)
                self.assertEqual(status,503)
                self.assertNotIn("answer",data)

    def test_configuration_does_not_expose_secret(self):
        out=public_configuration(self.env)
        self.assertTrue(out["available"])
        self.assertEqual(out["model"],"Qwen/Qwen3.5-9B")
        self.assertNotIn("hf_test_private_key",json.dumps(out))
        self.assertNotIn("long-random-beta-invite-key",json.dumps(out))
        self.assertFalse(public_configuration({})["available"])

    def test_rate_limit_is_per_client_and_fails_at_threshold(self):
        gate=RateGate(max_calls=2,window_s=60)
        self.assertTrue(gate.allow("one",now=100))
        self.assertTrue(gate.allow("one",now=101))
        self.assertFalse(gate.allow("one",now=102))
        self.assertTrue(gate.allow("two",now=102))
        self.assertTrue(gate.allow("one",now=161))

    def test_server_only_hf_url_and_no_redirects_or_proxies(self):
        # Transport is easily audited; tests must never call real Hugging Face.
        import inspect
        source=inspect.getsource(upstream_chat)
        self.assertIn("ProxyHandler({})",source)
        self.assertIn("NoRedirect",source)
        self.assertIn("MAX_UPSTREAM_BYTES",source)
        self.assertNotIn("os.system",source)


if __name__=="__main__":
    unittest.main()
