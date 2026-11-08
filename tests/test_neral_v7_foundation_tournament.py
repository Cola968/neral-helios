"""Offline tournament contracts: original tasks, localhost-only, no model downloads."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neral_v7_foundation_tournament import (
    OllamaLocal, Refused, generate_cases, evaluate_response,
    compare_local_models, task_digest, _validate_models,
)


class FakeLocal:
    def __init__(self, versions=None):
        self.versions = versions or {
            "qwen3:8b": "sha256:" + "1" * 64,
            "qwen3.5:9b": "sha256:" + "2" * 64,
        }
        self.queries = []
    def inventory(self):
        return self.versions
    def chat(self, model, prompt, *, think, max_tokens):
        self.queries.append((model, prompt, think, max_tokens))
        return {
            "content":"FINAL_ANSWER: 17",
            "reason":"stop",
            "eval_count":20,
            "duration_ns":3_000_000_000,
        }


class OfflineFoundationTournament(unittest.TestCase):
    def test_72_deterministic_distinct_cases_12_families_three_domains(self):
        cases = generate_cases()
        self.assertEqual(cases, generate_cases())
        self.assertEqual(len(cases), 72)
        self.assertEqual(len({c["id"] for c in cases}), 72)
        self.assertEqual(len({c["prompt"] for c in cases}), 72)
        self.assertEqual({d:sum(c["domain"]==d for c in cases)
                          for d in ("math","reasoning","code_trace")},
                         {"math":24,"reasoning":24,"code_trace":24})
        self.assertEqual(len({c["family"] for c in cases}),12)
        self.assertTrue(all(type(c["answer"]) is int and
                            -2_000_000 < c["answer"] < 2_000_000 for c in cases))
        self.assertEqual(len(task_digest(cases)),64)

    def test_grading_accepts_single_terminal_mark_never_last_number_guess(self):
        case={"answer":42}
        self.assertEqual(evaluate_response(case,"We calculate 5 + 37.\nFINAL_ANSWER: 42")["status"],"correct")
        self.assertEqual(evaluate_response(case,"FINAL_ANSWER: 43")["status"],"wrong_final")
        self.assertEqual(evaluate_response(case,"I think 42; maybe 41")["status"],"ungraded")
        self.assertEqual(evaluate_response(case,"FINAL_ANSWER: 42\nFINAL_ANSWER: 42")["status"],"ungraded")
        self.assertEqual(evaluate_response(case,"42")["status"],"ungraded")
        self.assertEqual(evaluate_response(case,"FINAL_ANSWER: 42\nnoise")["status"],"ungraded")
        self.assertEqual(evaluate_response(case,"")["status"],"ungraded")
        self.assertEqual(evaluate_response(case,"FINAL_ANSWER: 42",complete=False)["status"],"incomplete")

    def test_actual_ollama_bare_digest_is_canonicalized_without_guessing(self):
        from scripts.neral_v7_foundation_tournament import canonical_model_digest
        digest="d"*64
        self.assertEqual(canonical_model_digest(digest),"sha256:"+digest)
        self.assertEqual(canonical_model_digest("sha256:"+digest),"sha256:"+digest)
        for bad in ("garbage","d"*63,"D"*64,
                    "sha256:"+"z"*64,"sha512:"+"d"*64):
            with self.subTest(value=bad),self.assertRaises(Refused):
                canonical_model_digest(bad)

    def test_real_ollama_tag_inventory_bare_hash_is_accepted(self):
        from unittest.mock import patch
        import json
        class FakeResponse:
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def read(self,limit):
                return json.dumps({"models":[{"name":"qwen3.5:9b",
                             "digest":"7"*64}]}).encode()
        class FakeOpener:
            def open(self,request,timeout):return FakeResponse()
        with patch("scripts.neral_v7_foundation_tournament._local_opener",
                   return_value=FakeOpener()):
            actual=OllamaLocal().inventory()
        self.assertEqual(actual,{"qwen3.5:9b":"sha256:"+"7"*64})

    def test_local_inventory_enforces_no_proxy_no_redirect_no_download(self):
        from urllib.request import Request
        observed=[]
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self,*_): pass
            def read(self,limit):
                observed.append(("read",limit))
                return json.dumps({"models":[{"name":"qwen3:8b","digest":"sha256:"+"a"*64}]}).encode()
        class FakeOpener:
            def open(self, request, timeout):
                self.request=request
                self.timeout=timeout
                observed.append(("url",request.full_url,request.get_method()))
                return FakeResponse()
        fake=FakeOpener()
        with patch("scripts.neral_v7_foundation_tournament._local_opener",return_value=fake):
            models=OllamaLocal().inventory()
        self.assertEqual(models,{"qwen3:8b":"sha256:"+"a"*64})
        self.assertEqual(observed[0],("url","http://127.0.0.1:11434/api/tags","GET"))
        self.assertLessEqual(observed[1][1],262145)

    def test_reject_noninstalled_or_untrusted_model_tags(self):
        tags={"qwen3:8b":"sha256:"+"a"*64,"qwen3.5:9b":"sha256:"+"b"*64}
        self.assertEqual(_validate_models(["qwen3:8b","qwen3.5:9b"],tags),
                         {"qwen3:8b":tags["qwen3:8b"],"qwen3.5:9b":tags["qwen3.5:9b"]})
        for model_names in (
            ["qwen3:8b"],["qwen3:8b","qwen3:8b"],
            ["qwen3:8b","notinstalled:7b"],
            ["qwen3:8b","../../etc/passwd"],
            ["qwen3:8b","http://example.org"],
            ["qwen3:8b","qwen3:9b","qwen3:4b","qwen3:1b"],
        ):
            with self.subTest(model_names=model_names),self.assertRaises(Refused):
                _validate_models(model_names,tags)

    def test_compare_is_private_and_never_promotes_models(self):
        local=FakeLocal()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"tournament"
            public=compare_local_models(local,["qwen3:8b","qwen3.5:9b"],output,think=False)
            self.assertEqual(public["status"],"NERAL_V7_FOUNDATION_DEV_ONLY")
            self.assertEqual(len(local.queries),144)
            self.assertEqual(len((output/"private-responses.jsonl").read_text().splitlines()),144)
            self.assertTrue((output/"scalar-summary.json").is_file())
            self.assertEqual(public["count_per_model"],72)
            self.assertEqual(public["models"]["qwen3:8b"]["correct"]+
                             public["models"]["qwen3:8b"]["ungraded"]+
                             public["models"]["qwen3:8b"]["wrong_final"]+
                             public["models"]["qwen3:8b"]["incomplete"],72)
            self.assertFalse(public["model_trained"])
            self.assertFalse(public["release_validated"])
            self.assertFalse(public["model_superiority_proven"])
            self.assertFalse(public["adapter_promoted"])
            self.assertNotIn("prompt",json.dumps(public))
            self.assertNotIn("private-responses",json.dumps(public))
            self.assertTrue(all(entry[2] is False and entry[3]==512 for entry in local.queries))
            with self.assertRaises(Refused):
                compare_local_models(local,["qwen3:8b","qwen3.5:9b"],output,think=False)

    def test_abort_on_local_inventory_drift_without_publishing_success(self):
        class Changing(FakeLocal):
            def inventory(self):
                inventory=super().inventory()
                if len(self.queries)>=3:
                    return {**inventory, "qwen3:8b":"sha256:"+"9"*64}
                return inventory
        local=Changing()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"out"
            with self.assertRaises(Refused):
                compare_local_models(local,["qwen3:8b","qwen3.5:9b"],output,
                                     think=False,limit_cases=3)
            self.assertFalse((output/"scalar-summary.json").exists())

    def test_test_limit_cannot_be_misrepresented_as_full_benchmark(self):
        local=FakeLocal()
        with tempfile.TemporaryDirectory() as tmp:
            summary=compare_local_models(local,["qwen3:8b","qwen3.5:9b"],
                                         Path(tmp)/"pilot",think=True,limit_cases=4)
            self.assertEqual(summary["count_per_model"],4)
            self.assertEqual(summary["case_sample"],"pilot_not_full_suite")
            self.assertFalse(summary["model_superiority_proven"])
            self.assertTrue(all(row[2] and row[3]==2048 for row in local.queries))

    def test_safe_generation_uses_no_untrusted_code_execution(self):
        import inspect
        from scripts import neral_v7_foundation_tournament as mod
        text=inspect.getsource(mod)
        self.assertNotIn("eval(",text)
        self.assertNotIn("exec(",text)
        self.assertNotIn("shell=True",text)
        self.assertNotIn("/api/pull",text)
        self.assertNotIn("subprocess",text)


if __name__=="__main__":
    unittest.main()
