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

if __name__=="__main__":
    unittest.main()
