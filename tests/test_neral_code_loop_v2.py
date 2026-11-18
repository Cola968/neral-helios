"""NERAL V2 generic code-loop contracts; fake model and sandbox, no host execution.

These tests verify routing, retries, output provenance and response boundaries.
They are NOT claims that the 9B model solves arbitrary unseen tasks.
"""
from __future__ import annotations

import ast
import unittest
from neral_three_tier_runtime import ThreeTierRuntime


CASES = (
    (
        "Ein Kredit beträgt 1200 Euro. Nach dem ersten Jahr fallen 5 Prozent Zinsen an, "
        "danach zahle ich 250 Euro sondertilgend zurück. Nach Jahr zwei fallen wieder "
        "5 Prozent Zinsen auf die Restschuld an. Wie hoch ist sie dann?",
        "from decimal import Decimal\n"
        "debt=Decimal('1200'); debt=debt*Decimal('1.05')-Decimal('250'); "
        "debt=debt*Decimal('1.05')\nprint(f'{debt:.2f}')",
        "1060.50",
    ),
    (
        "Auf 200 Euro kommen zuerst zehn Prozent Nachlass, dann fünf weitere "
        "Prozent auf den reduzierten Preis und zuletzt neunzehn Prozent Umsatzsteuer. "
        "Welcher Endpreis ergibt sich?",
        "from decimal import Decimal\n"
        "p=Decimal('200')*Decimal('.90')*Decimal('.95')*Decimal('1.19')\n"
        "print(f'{p:.2f}')",
        "203.49",
    ),
    (
        "Anna, Ben und Cora besitzen gemeinsam zweiundzwanzig Chips. Ben besitzt "
        "zwei Chips mehr als Anna und Cora doppelt so viele wie Ben. "
        "Wie viele Chips haben alle drei einzeln?",
        "from fractions import Fraction\n"
        "a=Fraction(22-6,4); b=a+2; c=2*b\n"
        "assert a+b+c==22 and b==a+2 and c==2*b\n"
        "print(f'Anna={a}; Ben={b}; Cora={c}')",
        "Anna=4; Ben=6; Cora=12",
    ),
    (
        "Verteile Morgen-, Abend- und Nachtschicht auf Ada, Berta und Cora, "
        "jede Person genau einmal. Ada darf nicht morgens, Berta nicht nachts, "
        "Cora nicht abends. Wie viele gültige Pläne sind möglich?",
        "from itertools import permutations\n"
        "valid=[p for p in permutations(('Ada','Berta','Cora')) "
        "if p[0]!='Ada' and p[1]!='Cora' and p[2]!='Berta']\n"
        "print(len(valid))",
        "2",
    ),
    (
        "Löse die Gleichung drei Viertel mal x plus fünf Sechstel gleich "
        "sieben Drittel nach x auf.",
        "from fractions import Fraction\n"
        "x=(Fraction(7,3)-Fraction(5,6))/Fraction(3,4)\n"
        "assert Fraction(3,4)*x+Fraction(5,6)==Fraction(7,3)\nprint(x)",
        "2",
    ),
)


class FakeModel:
    def __init__(self, model, sha=None, replies=()):
        self.model=model
        self.adapter_sha256=sha
        self.loaded=False
        self.calls=[]
        self.replies=list(replies)

    def respond_chat(self, turns, *, max_new_tokens):
        self.loaded=True
        self.calls.append(([dict(m) for m in turns], max_new_tokens))
        if not self.replies:
            raise AssertionError("unexpected extra Helios model call")
        return self.replies.pop(0)

    def unload(self):
        self.loaded=False


def runtime(replies, tool):
    luna=FakeModel("qwen3.5:4b",replies=["Hallo."])
    helios=FakeModel("qwen3.5:9b+neral-v2-full","a"*64,replies)
    rt=ThreeTierRuntime(luna,helios,execute_python=tool)
    return rt,luna,helios


def turns(text):
    return [{"role":"system","content":"Lokaler Assistent"},
            {"role":"user","content":text}]


class GenericCodeLoop(unittest.TestCase):
    def test_five_free_tasks_execute_model_proposed_python_and_keep_unverified(self):
        for question, program, expected in CASES:
            with self.subTest(question=question[:35]):
                calls=[]
                def tool(code):
                    ast.parse(code)
                    calls.append(code)
                    self.assertEqual(code,program)
                    return {"status":"PASSED","output":expected}
                rt,luna,helios=runtime(["\x60\x60\x60python\n"+program+"\n\x60\x60\x60",
                    "Rechnerisches Ergebnis laut Sandbox: "+expected],tool)
                result=rt.answer(turns(question),"Luna")
                self.assertEqual(calls,[program])
                self.assertEqual(len(helios.calls),2)
                self.assertEqual(luna.calls,[])
                self.assertEqual(result["tier"],"Helios")
                self.assertEqual(result["auto_escalated_from"],"Luna")
                self.assertIs(result["verified"],False)
                self.assertIs(result["sandbox_executed"],True)
                self.assertEqual(result["tool_used"],"helios_code_sandbox")
                self.assertEqual(result["tool_calls_succeeded"],1)
                self.assertEqual(result["proof_backend"],None)
                self.assertIn(expected,result["text"])

    def test_failed_code_can_be_repaired_at_most_twice(self):
        seen=[]
        def tool(code):
            seen.append(code)
            if len(seen)<3:
                return {"status":"FAILED","traceback":"ValueError: incorrect variable"}
            return {"status":"PASSED","output":"7"}
        replies=[
            "\x60\x60\x60python\nprint(int('bad'))\n\x60\x60\x60",
            "\x60\x60\x60python\nraise ValueError('still wrong')\n\x60\x60\x60",
            "\x60\x60\x60python\nprint(3+4)\n\x60\x60\x60",
            "Die Sandbox gab 7 aus.",
        ]
        rt,_,helios=runtime(replies,tool)
        r=rt.answer(turns("Wie viele Stunden sind 3 plus 4 Stunden zusammen?"),"Luna")
        self.assertTrue(r["sandbox_executed"])
        self.assertEqual(r["sandbox_attempts"],3)
        self.assertEqual(len(helios.calls),4)
        self.assertIn("ValueError",str(helios.calls[1][0]))
        self.assertIn("ASCII",str(helios.calls[1][0]))
        self.assertIn("itertools.permutations",helios.calls[0][0][0]["content"])
        self.assertEqual(len(seen),3)

    def test_three_failed_runs_never_invent_a_verified_answer(self):
        rt,_,helios=runtime(["\x60\x60\x60python\nprint(0/0)\n\x60\x60\x60"]*3,
                            lambda _code:{"status":"FAILED","traceback":"ZeroDivisionError"})
        r=rt.answer(turns("Berechne die exakte Summe von 3 und 4 und 5."),"Luna")
        self.assertFalse(r["verified"])
        self.assertFalse(r["sandbox_executed"])
        self.assertNotEqual(r.get("tool_calls_succeeded"),1)
        self.assertEqual(len(helios.calls),3)
        self.assertNotIn("12",r["text"])
        self.assertEqual([x["stage"] for x in r["sandbox_failure_trace"]],
                         ["docker", "docker", "docker"])

    def test_missing_container_does_not_retry_model_or_run_host_python(self):
        seen=[]
        rt,_,helios=runtime(["\x60\x60\x60python\nprint(2+3)\n\x60\x60\x60"],
            lambda code:seen.append(code) or {"status":"UNAVAILABLE","error_type":"SandboxUnavailable"})
        r=rt.answer(turns("Wie viel sind 2 plus 3?"),"Luna")
        self.assertEqual(len(helios.calls),1)
        self.assertEqual(len(seen),1)
        self.assertFalse(r["sandbox_executed"])
        self.assertFalse(r["verified"])

    def test_malformed_first_code_reply_is_repaired_but_never_executed(self):
        calls=[]
        rt,_,helios=runtime([
            "Hier ist die Lösung: 42.",
            "\x60\x60\x60python\nprint(20+22)\n\x60\x60\x60",
            "Laut Sandbox ist es 42.",
        ],lambda code:calls.append(code) or {"status":"PASSED","output":"42"})
        out=rt.answer(turns("Berechne 20 plus 22."),"Luna")
        self.assertEqual(calls,["print(20+22)"])
        self.assertTrue(out["sandbox_executed"])
        self.assertFalse(out["verified"])
        self.assertEqual(out["sandbox_attempts"],2)
        self.assertIn("gültigen Python-Codeblock",str(helios.calls[1][0]))

    def test_non_code_reply_is_not_executed_or_claimed_as_sandbox(self):
        seen=[]
        rt,_,helios=runtime(["Meine Antwort: 42"]*3,lambda c:seen.append(c))
        r=rt.answer(turns("Berechne bitte 20 plus 22."),"Luna")
        self.assertEqual(seen,[])
        self.assertFalse(r["sandbox_executed"])
        self.assertFalse(r["verified"])
        self.assertEqual(len(helios.calls),3)
        self.assertNotIn("42",r["text"])
        self.assertEqual([x["stage"] for x in r["sandbox_failure_trace"]],
                         ["model_format", "model_format", "model_format"])

    def test_two_fenced_blocks_or_surrounding_prose_are_not_accepted(self):
        for text in ("\x60\x60\x60python\nprint(2)\n\x60\x60\x60\n"
                     "\x60\x60\x60python\nprint(3)\n\x60\x60\x60",
                     "Hier mein Skript:\n\x60\x60\x60python\nprint(2)\n\x60\x60\x60"):
            seen=[]
            rt,_,_=runtime([text]*3,lambda code:seen.append(code))
            r=rt.answer(turns("Berechne 1 plus 1."),"Luna")
            self.assertFalse(r["sandbox_executed"])
            self.assertEqual(seen,[])

    def test_simple_non_compute_stays_on_luna(self):
        rt,luna,helios=runtime([],
            lambda _code: (_ for _ in ()).throw(AssertionError("must not execute")))
        r=rt.answer(turns("Hallo, wie geht es dir?"),"Luna")
        self.assertEqual(r["tier"],"Luna")
        self.assertEqual(len(luna.calls),1)
        self.assertEqual(helios.calls,[])
