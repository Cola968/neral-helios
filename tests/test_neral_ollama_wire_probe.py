"""CPU-only regression for the diagnostic Ollama wire capture (no GPU)."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from scripts.neral_ollama_wire_probe import (
    summarize_ollama_request, summarize_ollama_response, summarize_ollama_error,
    make_capturing_opener,
)


class WireCaptureTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "model": "qwen3.5:9b",
            "messages": [
                {"role": "system", "content": "SYSTEM SECRET"},
                {"role": "user", "content": "PRIVATE PROMPT"},
            ],
            "options": {"num_ctx": 4096, "num_predict": 2048},
            "think": False, "keep_alive": "5m", "stream": False,
        }

    def test_request_schema_reports_roles_and_missing_stop_tools_format(self):
        result = summarize_ollama_request(self.payload)
        self.assertEqual(result["message_roles"], ["system", "user"])
        self.assertEqual(result["message_count"], 2)
        self.assertEqual(result["empty_message_count"], 0)
        self.assertFalse(result["has_top_level_stop"])
        self.assertFalse(result["has_options_stop"])
        self.assertFalse(result["has_format"])
        self.assertFalse(result["has_tools"])
        self.assertTrue(result["valid_message_roles"])
        self.assertNotIn("SYSTEM SECRET", repr(result))
        self.assertNotIn("PRIVATE PROMPT", repr(result))

    def test_bad_messages_are_detected_without_uploading_contents(self):
        self.payload["messages"] = [
            {"role":"system","content":" "},
            {"role":"unknown","content":"PRIVATE PROMPT"},
            {"role":"assistant","content":""},
        ]
        report = summarize_ollama_request(self.payload)
        self.assertFalse(report["valid_message_roles"])
        self.assertEqual(report["empty_message_count"], 2)
        self.assertNotIn("PRIVATE PROMPT",repr(report))

    def test_200_response_records_counts_and_native_tool_presence(self):
        body = {
            "done": True, "done_reason": "stop",
            "eval_count": 19, "prompt_eval_count": 212,
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"function":{"name":"secret"}}],
                "thinking": "NEVER_LOG_THINKING",
            },
        }
        report = summarize_ollama_response(body, status=200)
        self.assertEqual(report["eval_count"], 19)
        self.assertEqual(report["prompt_eval_count"], 212)
        self.assertEqual(report["tool_calls_count"], 1)
        self.assertEqual(report["content_length"], 0)
        self.assertEqual(report["thinking_length"], len("NEVER_LOG_THINKING"))
        self.assertEqual(report["done_reason"], "stop")
        self.assertNotIn("NEVER_LOG_THINKING",repr(report))
        self.assertNotIn("secret",repr(report))

    def test_500_error_metadata_never_contains_raw_prompt(self):
        secret='model runner failed while processing PRIVATE PERSONAL PROMPT'
        raw=json.dumps({"error":secret}).encode()
        report=summarize_ollama_error(raw,status=500)
        self.assertEqual(report["ollama_http_status"],500)
        self.assertEqual(report["ollama_error_json_keys"],["error"])
        self.assertEqual(report["ollama_error_length"],len(secret))
        self.assertEqual(len(report["ollama_error_sha256"]),64)
        self.assertNotIn("PRIVATE PERSONAL PROMPT",repr(report))

    def test_wrapper_preserves_exact_http_200_response_and_url(self):
        events=[]
        raw=json.dumps({"done":True,"done_reason":"stop","message":{"content":"OK"},
            "eval_count":1,"prompt_eval_count":19}).encode()
        class UpstreamResponse:
            status=200
            def __init__(self): self._buf=io.BytesIO(raw)
            def __enter__(self): return self
            def __exit__(self,*a):return False
            def read(self,*a):return self._buf.read(*a)
        class UpstreamOpener:
            def open(self,request,timeout):
                self.request=request
                return UpstreamResponse()
        upstream=UpstreamOpener()
        factory=make_capturing_opener(lambda:upstream,events)
        import urllib.request
        request=urllib.request.Request("http://127.0.0.1:11434/api/chat",
                data=json.dumps(self.payload).encode(),method="POST")
        with factory().open(request,timeout=90) as response:
            self.assertEqual(response.status,200)
            self.assertEqual(response.read(1024),raw)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["eval_count"],1)
        self.assertEqual(events[0]["message_roles"],["system","user"])
        self.assertNotIn("PRIVATE PROMPT",repr(events))
        self.assertEqual(json.loads(upstream.request.data),self.payload)

    def test_owner_local_private_error_file_not_in_public_report(self):
        import tempfile
        from pathlib import Path
        from scripts.neral_tir_smoke3 import save_private_ollama_errors
        private="PRIVATE_TOKEN_123"
        with tempfile.TemporaryDirectory() as temp:
            saved=save_private_ollama_errors(
                [{"http_status":500,"ollama_error_body":private}],
                base_dir=Path(temp))
            self.assertTrue(saved)
            files=list(Path(temp).glob("smoke3-ollama-error-*.json"))
            self.assertEqual(len(files),1)
            self.assertIn(private,files[0].read_text(encoding="utf-8"))

    def test_error_raw_text_is_private_sink_only(self):
        events=[]
        private=[]
        raw=json.dumps({"error":"SECRET_PRIVATE_PROMPT_500"}).encode()
        class RejectingOpener:
            def open(self,request,timeout):
                raise urllib.error.HTTPError(request.full_url,500,"ISE",{},
                                             io.BytesIO(raw))
        import urllib.request
        req=urllib.request.Request("http://127.0.0.1:11434/api/chat",
                  data=json.dumps(self.payload).encode(),method="POST")
        with self.assertRaises(urllib.error.HTTPError):
            make_capturing_opener(lambda:RejectingOpener(),events,
                                  private_errors=private)().open(req,timeout=90)
        self.assertEqual(private[0]["ollama_error_body"],
                         raw.decode("utf-8"))
        self.assertNotIn("SECRET_PRIVATE_PROMPT_500",repr(events))

    def test_wrapper_preserves_http_error_for_existing_backend(self):
        events=[]
        raw=json.dumps({"error":"PRIVATE PROMPT runner failure"}).encode()
        class ErrorOpener:
            def open(self,request,timeout):
                raise urllib.error.HTTPError(
                    request.full_url,500,"Internal Server Error",{},
                    io.BytesIO(raw))
        import urllib.request
        request=urllib.request.Request("http://127.0.0.1:11434/api/chat",
                data=json.dumps(self.payload).encode(),method="POST")
        with self.assertRaises(urllib.error.HTTPError) as got:
            make_capturing_opener(lambda:ErrorOpener(),events)().open(request,timeout=90)
        self.assertEqual(got.exception.code,500)
        self.assertEqual(got.exception.read(),raw)
        self.assertEqual(events[0]["ollama_http_status"],500)
        self.assertNotIn("PRIVATE PROMPT",repr(events))


if __name__=="__main__":
    unittest.main()
