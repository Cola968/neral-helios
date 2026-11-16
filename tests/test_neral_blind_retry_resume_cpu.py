"""Red/green contracts for retrying transport failure and exact 26-case resume."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, call

from neral_dual_routing import _ollama_chat, OllamaTransportFailure
from scripts.neral_blind_100_gpu import load_resume, parse_answer
from scripts.neral_blind_100_cases import build, serialize, exclusion
from scripts.neral_generalization_100_run import BASE_TAG

class OllamaTransientRetry(unittest.TestCase):
    MESSAGES=[{"role":"system","content":"stable"},{"role":"user","content":"stable input"}]

    def test_incomplete_then_success_reuses_identical_prompt_and_parameters(self):
        incomplete={"done":False,"done_reason":None,"message":{"content":""}}
        good={"done":True,"done_reason":"stop","message":{"content":"42"},
              "load_duration":4100000000,"total_duration":5000000000}
        with patch("neral_dual_routing._post_chat_with_status",side_effect=[
             (incomplete,200),(good,200)]) as send, patch(
             "neral_dual_routing.time.sleep") as sleep,patch(
             "neral_dual_routing.sample_residency",return_value={"both_resident":False,"models_vram_mib":{}}):
            reply,info=_ollama_chat(BASE_TAG,self.MESSAGES,0)
        self.assertEqual(reply,"42")
        self.assertEqual(info["attempts"],2)
        self.assertEqual(info["failed_attempts"][0]["reason"],"done_false")
        self.assertEqual(info["failed_attempts"][0]["http_status"],200)
        self.assertEqual(info["load_ms"],4100.0)
        self.assertEqual(send.call_count,2)
        first=send.call_args_list[0].args[0]
        second=send.call_args_list[1].args[0]
        self.assertEqual(first,second)
        self.assertEqual(first["messages"],self.MESSAGES)
        self.assertEqual(first["keep_alive"],"5m")
        self.assertEqual(first["options"]["seed"],0)
        self.assertEqual(first["options"]["temperature"],0)
        self.assertEqual(sleep.call_count,1)

    def test_empty_answer_retry_exactly_two_times_and_redacted_failure(self):
        response={"done":True,"done_reason":"stop","message":{"content":""}}
        with patch("neral_dual_routing._post_chat_with_status",
                   return_value=(response,200)) as send,patch(
                   "neral_dual_routing.time.sleep") as sleep:
            with self.assertRaises(OllamaTransportFailure) as got:
                _ollama_chat(BASE_TAG,self.MESSAGES,12)
        self.assertEqual(send.call_count,3)
        self.assertEqual(sleep.call_count,2)
        self.assertEqual(got.exception.diagnostic["reason"],"empty_content")
        self.assertEqual(got.exception.diagnostic["http_status"],200)
        self.assertNotIn("stable input",str(got.exception))
        self.assertEqual(got.exception.diagnostic["attempts"],3)

    def test_token_length_is_not_retryable_by_transport_retries(self):
        packet={"done":True,"done_reason":"length","message":{"content":"42"}}
        with patch("neral_dual_routing._post_chat_with_status",
                   return_value=(packet,200)) as send:
            with self.assertRaises(OllamaTransportFailure) as got:
                _ollama_chat(BASE_TAG,self.MESSAGES,0)
        self.assertEqual(got.exception.diagnostic["reason"],"token_limit")
        self.assertEqual(got.exception.diagnostic["done_reason"],"length")
        self.assertEqual(send.call_count,1)

    def test_network_timeout_retry_then_exact_same_packet(self):
        good={"done":True,"done_reason":"stop","message":{"content":"1"}}
        with patch("neral_dual_routing._post_chat_with_status",
            side_effect=[TimeoutError("private response content"),(good,200)]) as send,patch(
            "neral_dual_routing.time.sleep"),patch(
            "neral_dual_routing.sample_residency",return_value={"both_resident":False,"models_vram_mib":{}}):
            reply,info=_ollama_chat(BASE_TAG,self.MESSAGES,24)
        self.assertEqual(reply,"1")
        self.assertEqual(info["attempts"],2)
        self.assertEqual(info["failed_attempts"][0]["reason"],"timeout")
        self.assertNotIn("private response content",str(info))

class FrozenCheckpoint(unittest.TestCase):
    def test_26_prefix_survives_resume_with_identical_hashes_and_answers(self):
        tasks=build()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/"prior";source.mkdir()
            rows=[];raws=[]
            for item in tasks[:26]:
                answer=str(item["gold_integer"])
                rows.append({"id":item["id"],"category":item["category"],
                   "question_sha256":item["question_sha256"],
                   "base_correct":True,"dual_correct":True,
                   "base_seconds":1.0,"dual_seconds":2.0,
                   "policy":"raw_qwen_already_selected","verified":False,
                   "dual_model_load_ms":[],"gpu_peak_mib":5500,
                   "both_resident":False})
                raws.append({"id":item["id"],"question":item["question"],
                    "gold_integer":item["gold_integer"],
                    "base_reply":answer,"dual_reply":answer})
            (source/"blind100-results.progress.jsonl").write_text(
                "".join(json.dumps(x)+"\n" for x in rows),encoding="utf8")
            (source/"blind100-results.full_raw.jsonl").write_text(
                "".join(json.dumps(x)+"\n" for x in raws),encoding="utf8")
            manifest={"blind100_sha256":hashlib.sha256(serialize(tasks)).hexdigest()}
            (source/"blind100-manifest.json").write_text(json.dumps(manifest))
            imported=load_resume(source,tasks,manifest)
            self.assertEqual(len(imported[0]),26)
            self.assertEqual(len(imported[1]),26)
            self.assertEqual([x["id"] for x in imported[0]],
                [x["id"] for x in tasks[:26]])
            self.assertEqual(imported[1][-1]["question"],tasks[25]["question"])

            raws[-1]["dual_reply"]="wrong"
            (source/"blind100-results.full_raw.jsonl").write_text(
                "".join(json.dumps(x)+"\n" for x in raws),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                load_resume(source,tasks,manifest)

    def test_mismatched_manifest_or_reordered_prefix_rejected(self):
        tasks=build()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"blind100-results.progress.jsonl").write_text("")
            (p/"blind100-results.full_raw.jsonl").write_text("")
            (p/"blind100-manifest.json").write_text('{"blind100_sha256":"bad"}')
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                load_resume(p,tasks,{"blind100_sha256":"good"})


class TerminalTokenLimitScoring(unittest.TestCase):
    """A deterministic exhausted token budget is a wrong answer, never a win."""
    def test_control_token_limit_counts_incorrect_without_changing_decoder(self):
        from scripts.neral_blind_100_gpu import pair_case
        from scripts.neral_generalization_100_run import TUNED_TAG
        item=build()[26]
        class Accepted:
            status_code=200
            json={"text":str(item["gold_integer"]),
                  "tool_model":TUNED_TAG,
                  "fallback_model":BASE_TAG,
                  "model_digests":{
                      BASE_TAG:"ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9",
                      TUNED_TAG:"9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8"},
                  "selection_policy":"independently_verified_tir","verified":True,
                  "ollama_resident_after":{"both_resident":False},
                  "gpu_device_peak_mib":5800,
                  "inference_spans":[]}
        class Client:
            def post(self,*a,**kw):return Accepted()
        failure=OllamaTransportFailure({
            "model":BASE_TAG,"reason":"token_limit","done_reason":"length",
            "http_status":200,"attempts":1,"done":True,
            "attempt_diagnostics":[]})
        with patch("scripts.neral_blind_100_gpu._ollama_chat",side_effect=failure) as fetch, \
             patch("scripts.neral_blind_100_gpu.gpu_offload_mib",return_value=5000), \
             patch("scripts.neral_blind_100_gpu.sample_device_mib",return_value=5750):
            result=pair_case(item,Client(),{})
        self.assertFalse(result["base_correct"])
        self.assertTrue(result["dual_correct"])
        self.assertEqual(result["base_completion_status"],"token_limit")
        self.assertEqual(result["base_done_reason"],"length")
        self.assertEqual(result["base_http_status"],200)
        self.assertEqual(result["raw"]["base_reply"],"")
        self.assertEqual(fetch.call_count,1)

    def test_transport_exception_after_two_retries_remains_fatal(self):
        from scripts.neral_blind_100_gpu import pair_case
        item=build()[26]
        bad=OllamaTransportFailure({
            "model":BASE_TAG,"reason":"done_false","done_reason":"absent_or_other",
            "http_status":200,"attempts":3,"done":False,
            "attempt_diagnostics":[]})
        with patch("scripts.neral_blind_100_gpu._ollama_chat",side_effect=bad), \
             patch("scripts.neral_blind_100_gpu.sample_device_mib",return_value=5700):
            with self.assertRaises(OllamaTransportFailure):
                pair_case(item,object(),{})

if __name__=="__main__":
    unittest.main()
