"""V2.2 regression tests for runtime Docker recovery and honest model fallback."""
import os
import subprocess
import unittest
from unittest.mock import patch

from neral_three_tier_runtime import ThreeTierRuntime
from neral_tir_experimental import _run_ephemeral_python


class Model:
    def __init__(self, replies):
        self.replies=list(replies)
        self.calls=[]
        self.loaded=False
        self.model="qwen3.5:9b+neral-v2-full"
        self.adapter_sha256="a"*64
    def respond_chat(self, turns, *, max_new_tokens, temperature=None, max_time_seconds=None):
        self.loaded=True
        self.calls.append((turns,max_new_tokens))
        return self.replies.pop(0)
    def unload(self):
        self.loaded=False


def question(text="Wie viel sind 2 plus 3?"):
    return [{"role":"system","content":"NERAL offline"},{"role":"user","content":text}]


class HeliosFallback(unittest.TestCase):
    def runtime(self,replies,tool):
        h=Model(replies)
        l=Model(["Hallo"])
        l.model="qwen3.5:4b"
        l.adapter_sha256=None
        return ThreeTierRuntime(l,h,execute_python=tool),h

    def test_unavailable_docker_returns_real_helios_answer_with_explicit_reason(self):
        rt,h=self.runtime(["```python\nprint(2+3)\n```","2 plus 3 ergibt 5."],
            lambda _:{"status":"UNAVAILABLE","error_code":"docker_unavailable","error_type":"SandboxUnavailable"})
        out=rt.answer(question(),"Luna")
        self.assertIn("2 plus 3 ergibt 5",out["text"])
        self.assertNotIn("Es liegt kein ausgeführtes Ergebnis vor",out["text"])
        self.assertEqual(out["sandbox_failure_code"],"docker_unavailable")
        self.assertFalse(out["sandbox_executed"])
        self.assertFalse(out["verified"])
        self.assertFalse(out["safe_to_promote"])
        self.assertIsNone(out["tool_used"])
        self.assertEqual(len(h.calls),2)

    def test_three_invalid_programs_trigger_real_model_fallback(self):
        rt,h=self.runtime(["```python\nraise ValueError(1)\n```"]*3+["Die Aufgabe ist noch nicht sicher gelöst."],
            lambda _:{"status":"FAILED","error_code":"runtime_error","error_type":"RuntimeError","traceback":"ValueError: 1"})
        out=rt.answer(question(),"Luna")
        self.assertIn("noch nicht sicher gelöst",out["text"])
        self.assertEqual(out["sandbox_failure_code"],"runtime_error")
        self.assertEqual(len(h.calls),4)
        self.assertEqual(out["sandbox_attempts"],3)
        self.assertFalse(out["sandbox_executed"])

    def test_model_fallback_failure_reports_unavailability_without_made_up_answer(self):
        rt,h=self.runtime(["```python\nprint(1)\n```"],
            lambda _:{"status":"UNAVAILABLE","error_code":"docker_unavailable"})
        out=rt.answer(question(),"Luna")
        self.assertTrue(out["text"].strip())
        self.assertNotIn("Es liegt kein ausgeführtes Ergebnis vor",out["text"])
        self.assertFalse(out.get("model_fallback_generated",True))
        self.assertFalse(out["sandbox_executed"])

    def test_invalid_fence_and_empty_output_have_distinct_error_codes(self):
        rt,h=self.runtime(["No python","No python","No python","Eine ungeprüfte Antwort."],
                          lambda _:self.fail("invalid source should not execute"))
        x=rt.answer(question(),"Luna")
        self.assertEqual(x["sandbox_failure_code"],"invalid_python_code")
        rt2,h2=self.runtime(["```python\nprint(1)\n```"]*3+["Antwort ohne Ausführung."],
                          lambda _:{"status":"FAILED","error_code":"empty_stdout"})
        y=rt2.answer(question(),"Luna")
        self.assertEqual(y["sandbox_failure_code"],"empty_stdout")


class DockerRecovery(unittest.TestCase):
    def test_daemon_checked_before_image_and_run_without_auto_start_on_linux(self):
        seen=[]
        def fake(cmd,**kwargs):
            seen.append(cmd)
            if cmd[:2]==["docker","info"]:
                return subprocess.CompletedProcess(cmd,0,b"ok",b"")
            if cmd[:3]==["docker","image","inspect"]:
                return subprocess.CompletedProcess(cmd,0,b"ok",b"")
            if cmd[:2]==["docker","run"]:
                return subprocess.CompletedProcess(cmd,0,b"5\n",b"")
            raise AssertionError("unrecognized command "+str(cmd))
        with patch.dict(os.environ,{"NERAL_CODE_SANDBOX_V2":"1"}),patch(
                "neral_tir_experimental.subprocess.run",side_effect=fake):
            got=_run_ephemeral_python("print(2+3)")
        self.assertEqual(got["status"],"PASSED")
        self.assertEqual(seen[0][:2],["docker","info"])
        self.assertEqual(seen[1][:3],["docker","image","inspect"])

    def test_missing_daemon_yields_machine_readable_error(self):
        def unavailable(cmd,**kwargs):
            return subprocess.CompletedProcess(cmd,1,b"",b"not running")
        with patch("neral_tir_experimental.subprocess.run",side_effect=unavailable):
            got=_run_ephemeral_python("print(2+3)")
        self.assertEqual(got["status"],"UNAVAILABLE")
        self.assertEqual(got["error_code"],"docker_unavailable")

    def test_python_syntax_error_carries_machine_readable_code(self):
        got=_run_ephemeral_python("if True print(1)")
        self.assertEqual(got["error_code"],"invalid_python_code")


if __name__=="__main__":
    unittest.main()
