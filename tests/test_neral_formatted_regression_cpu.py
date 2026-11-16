"""Frozen 100-case *regression* formatting contract, not a new blind gate.

TDD: separate benchmark-only prompt/score path must preserve the original
script and both models' original parameters, models and 100 task hashes.
"""
from __future__ import annotations
import json
import os
import unittest
from unittest.mock import patch
from neral_answer_contract import formatted_benchmark_system, parse_terminal_final_integer
from neral_identity import build_identity_prompt
from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app
from scripts.neral_blind_100_gpu import pair_case
from scripts.neral_blind_100_cases import build,serialize

class FormatContract(unittest.TestCase):
    def test_fixed_identity_and_identical_system_for_both_arms(self):
        source=build_identity_prompt("web","local_foundation_not_neral_adapter")
        baseline=formatted_benchmark_system()
        self.assertTrue(baseline.startswith(source))
        self.assertIn("FINAL_ANSWER:",baseline)
        self.assertIn("one integer",baseline)
        self.assertEqual(baseline,formatted_benchmark_system())
    def test_last_line_only_is_strictly_scored(self):
        accepted={
            "FINAL_ANSWER: 42":42,
            "Some explanation.\nFINAL_ANSWER: -17":-17,
            "short 1+2 reasoning\nFINAL_ANSWER: 3\n":3,
            "FINAL_ANSWER: +0":0,
        }
        for response,gold in accepted.items():
            self.assertEqual(parse_terminal_final_integer(response),gold,response)
        rejected=[
            "42","The answer is 42.","FINAL_ANSWER: 42\nmore text",
            "FINAL_ANSWER: 42\nFINAL_ANSWER: 42",
            "FINAL_ANSWER: 42.0","FINAL_ANSWER: 42 extra",
            "FINAL_ANSWER: 42\nFINAL_ANSWER: 4",
            "FINAL_ANSWER: NaN","FINAL_ANSWER: 42\n\nThink again",
            "FINAL_ANSWER: 100000000000000000000000000000",
            "",
        ]
        for response in rejected:
            self.assertIsNone(parse_terminal_final_integer(response),response)

class FormattedApiGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=create_preview_app(OllamaLocalBackend("qwen3:8b"),
            experimental_high_backend=OllamaLocalBackend(
                "qwen3.5:9b",inference_profile="measured_nonthinking_2048"))
    def request(self,payload):
        with self.app.test_client() as c:
            c.get("/",base_url="http://127.0.0.1:9690")
            token=c.get("/api/neral/preview/session",base_url="http://127.0.0.1:9690").json["csrf"]
            return c.post("/api/neral/chat/think",json=payload,
                headers={"X-Neral-CSRF":token},base_url="http://127.0.0.1:9690")
    def test_experimental_flag_must_be_bool_and_explicitly_authorized(self):
        basic={"prompt":"Count some paths","experimental_tir":True,
                "conservative_tir":True,"dual_routing":True}
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1",
            "NERAL_EXPERIMENTAL_DUAL_ROUTE":"1","NERAL_FORMATTED_EVAL":"0"}):
            self.assertEqual(self.request({**basic,"evaluation_final_answer":True}).status_code,403)
        self.assertEqual(self.request({**basic,"evaluation_final_answer":"yes"}).status_code,400)
        self.assertEqual(self.request({"prompt":"Count some paths",
            "evaluation_final_answer":True}).status_code,400)
    def test_same_formatted_system_enters_unmodified_prod_dual_route(self):
        original={"text":"FINAL_ANSWER: 42","mode":"HIGH","verified":True,
                  "verification_status":"verified","verification_scope":"exact_user_python_execution",
                  "response_source":"verified_tir",
                  "selection_policy":"independently_verified_tir",
                  "tool_model":"neral-qwen35-qlora-v1:latest",
                  "fallback_model":"qwen35-9b-qlora-control-20261112:latest"}
        payload={"prompt":"What is 3*14?","experimental_tir":True,
                 "conservative_tir":True,"dual_routing":True,
                 "evaluation_final_answer":True}
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1",
            "NERAL_EXPERIMENTAL_DUAL_ROUTE":"1","NERAL_FORMATTED_EVAL":"1"}),patch(
            "neral_dual_routing.run_dual_route",return_value=original) as mock:
            result=self.request(payload)
        self.assertEqual(result.status_code,200,result.json)
        messages=mock.call_args.args[0]
        self.assertEqual(messages[0]["content"],formatted_benchmark_system())
        self.assertEqual(messages[-1]["content"],payload["prompt"])

class CompareMatchedBenchmark(unittest.TestCase):
    def test_formatter_is_sent_to_both_control_and_api_without_modifying_task(self):
        item=build()[0]
        ctrl_calls=[]
        def fake_ollama(model,messages,seed):
            ctrl_calls.append((model,messages,seed))
            return "FINAL_ANSWER: "+str(item["gold_integer"]),{
                "load_ms":200,"attempts":1}
        class Response:
            status_code=200
            json={"text":"FINAL_ANSWER: "+str(item["gold_integer"]),
                  "tool_model":"neral-qwen35-qlora-v1:latest",
                  "fallback_model":"qwen35-9b-qlora-control-20261112:latest",
                  "model_digests":{ "qwen35-9b-qlora-control-20261112:latest":
                     "ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9",
                    "neral-qwen35-qlora-v1:latest":
                     "9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8"},
                  "selection_policy":"raw_qwen_already_selected","verified":False,
                  "inference_spans":[]}
        class Client:
            def post(self,path,*,json,headers,base_url):
                self.payload=json
                return Response()
        client=Client()
        with patch("scripts.neral_blind_100_gpu._ollama_chat",side_effect=fake_ollama),patch(
            "scripts.neral_blind_100_gpu.gpu_offload_mib",return_value=4800),patch(
            "scripts.neral_blind_100_gpu.sample_device_mib",return_value=5600):
            outcome=pair_case(item,client,{},formatted_contract=True)
        self.assertEqual(ctrl_calls[0][1][0]["content"],formatted_benchmark_system())
        self.assertTrue(client.payload["evaluation_final_answer"])
        self.assertTrue(outcome["base_correct"])
        self.assertTrue(outcome["dual_correct"])
        self.assertEqual(item,build()[0])
        self.assertEqual(len(build()),100)

if __name__=="__main__":unittest.main()
