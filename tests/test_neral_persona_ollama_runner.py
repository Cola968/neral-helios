"""No GPU in CI: NERAL live Ollama baseline must never fabricate model answers."""
import json
import tempfile
import unittest
from pathlib import Path
from neral_persona_eval import load_holdout, grade_predictions
from neral_persona_ollama_runner import (
    collect_ollama, make_payload, save_run, LocalRunError,
)


def fake_tags():
    return {"models":[{"name":"qwen3.5:9b","digest":"a"*64}]}


def fake_model(payload):
    return {"model":payload["model"],"done":True,
            "message":{"role":"assistant","content":"NERAL fixture answer"}}


class LocalOllamaEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases=load_holdout()[:3]

    def test_ollama_payload_has_fixed_model_tools_disabled_and_frozen_parameters(self):
        request=make_payload("Was ist 2+2?","qwen3.5:9b")
        self.assertEqual(request["model"],"qwen3.5:9b")
        self.assertEqual(request["tools"],[])
        self.assertEqual(request["options"]["temperature"],0.0)
        self.assertEqual(request["options"]["num_predict"],128)
        self.assertEqual(request["options"]["num_ctx"],4096)
        self.assertFalse(request["stream"])
        self.assertFalse(request["think"])
        for name in ("my_model:latest","http://evil","neral:1","qwen3.5:9b;exit"):
            with self.subTest(name=name),self.assertRaises(LocalRunError):
                make_payload("Hallo",name)
        with self.assertRaises(LocalRunError):
            make_payload("  ","qwen3.5:9b")

    def test_fixture_transport_does_not_count_as_real_model_inference(self):
        calls=[]
        def transport(payload):
            calls.append(payload)
            return fake_model(payload)
        result=collect_ollama(self.cases,model="qwen3.5:9b",
                              model_call=transport,tags_call=fake_tags)
        self.assertEqual(len(calls),len(self.cases))
        self.assertEqual(result["provenance"]["mode"],"simulated_test_transport")
        self.assertEqual(result["provenance"]["model_digest"],"a"*64)
        self.assertFalse(all(x["model_generated"] for x in result["predictions"].values()))
        self.assertIsNone(grade_predictions(self.cases,result["predictions"])["accuracy_pct"])

    def test_missing_model_or_digest_fails_before_first_chat_request(self):
        called=[]
        def record(payload):
            called.append(payload)
            return fake_model(payload)
        for inv in ({"models":[]},{"models":[{"name":"qwen3.5:9b","digest":None}]},{}):
            with self.subTest(tags=inv),self.assertRaises(LocalRunError):
                collect_ollama(self.cases,model="qwen3.5:9b",
                    model_call=record,tags_call=lambda:inv)
        self.assertEqual(called,[])

    def test_wrong_model_or_partial_reply_is_not_accepted(self):
        for reply in (
            {"model":"different:latest","done":True,
             "message":{"role":"assistant","content":"HI"}},
            {"model":"qwen3.5:9b","done":False,
             "message":{"role":"assistant","content":"partial"}},
            {"model":"qwen3.5:9b","done":True,
             "message":{"role":"tool","content":"unauthorized"}},
            {"model":"qwen3.5:9b","done":True,
             "message":{"role":"assistant","content":" "}},
            {"model":"qwen3.5:9b","done":True,
             "message":{"role":"assistant","content":"Hi","tool_calls":[{"function":{"name":"shell"}}]}},
        ):
            with self.subTest(reply=reply),self.assertRaises(LocalRunError):
                collect_ollama(self.cases,model="qwen3.5:9b",
                    model_call=lambda payload:reply,tags_call=fake_tags)

    def test_output_is_json_no_overwrite_and_discloses_fake_adapter(self):
        run=collect_ollama(self.cases,model="qwen3.5:9b",
              model_call=fake_model,tags_call=fake_tags)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"baseline.json"
            save_run(run,output)
            stored=json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stored["provenance"]["mode"],"simulated_test_transport")
            self.assertEqual(stored["provenance"]["model_digest"],"a"*64)
            with self.assertRaises(FileExistsError):
                save_run(run,output)
            with self.assertRaises(LocalRunError):
                save_run(run,Path(tmp)/"report.txt")

    def test_rejects_forged_holdout_and_bounded_case_limit(self):
        with self.assertRaises(LocalRunError):
            collect_ollama(self.cases+[dict(self.cases[0])],model="qwen3.5:9b",
                          model_call=fake_model,tags_call=fake_tags)


if __name__ == "__main__":
    unittest.main()
