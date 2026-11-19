"""V3 model-planned non-web tasks are bounded and never called verified."""
import unittest
from neral_three_tier_runtime import ThreeTierRuntime


QUESTION = ("Bei einer internen Reservierung in unserem kleinen Team zahlt "
            "Person A zunächst alle Kosten, Person B erstattet später einen "
            "vereinbarten Anteil, danach wird ein zusätzlicher Rabatt wirksam. "
            "Die vorhandenen Daten sind 420 Euro, 25 Prozent und 10 Prozent. "
            "Welche Zahl muss Person A nach allen Schritten endgültig tragen?")


class ScriptedModel:
    def __init__(self,name,answers,sha=None):
        self.model=name
        self.answers=list(answers)
        self.adapter_sha256=sha
        self.loaded=False
        self.calls=[]
    def respond_chat(self,turns,*,max_new_tokens,temperature=None,max_time_seconds=None):
        self.loaded=True
        self.calls.append([t["content"] for t in turns])
        return self.answers.pop(0) if self.answers else "Fertig."
    def unload(self): self.loaded=False


class V3LocalLoopTests(unittest.TestCase):
    def test_nontrivial_question_without_tool_trigger_uses_model_plan_then_docker(self):
        luna=ScriptedModel("qwen3.5:4b",[])
        helios=ScriptedModel("qwen3.5:9b+neral-v2-full",[
            '{"steps":["python_sandbox","validation_checker"],"needs_current_data":false}',
            "```python\nprint(283.50)\n```",
            "Die tatsächliche Sandbox gab einen Wert aus, nicht unabhängig geprüft."
        ],"a"*64)
        called=[]
        def docker(code):
            called.append(code)
            return {"status":"PASSED","output":"283.50"}
        runtime=ThreeTierRuntime(luna,helios,execute_python=docker)
        turns=[{"role":"system","content":"System"},
               {"role":"user","content":QUESTION}]
        answer=runtime.answer(turns,"Luna")
        self.assertTrue(called)
        self.assertFalse(answer["verified"])
        self.assertEqual(answer["v3_plan"],["python_sandbox","validation_checker"])
        self.assertEqual(answer["tier"],"Helios")
        self.assertEqual(answer["auto_escalated_from"],"Luna")

    def test_short_greeting_keeps_luna_latency(self):
        luna=ScriptedModel("qwen3.5:4b",["Hallo!"])
        helios=ScriptedModel("qwen3.5:9b+neral-v2-full",[],"a"*64)
        runtime=ThreeTierRuntime(luna,helios,execute_python=lambda _:{"status":"UNAVAILABLE"})
        turns=[{"role":"system","content":"System"},
               {"role":"user","content":"Hallo"}]
        answer=runtime.answer(turns,"Luna")
        self.assertEqual(answer["tier"],"Luna")
        self.assertEqual(len(helios.calls),0)


if __name__=="__main__":
    unittest.main()
