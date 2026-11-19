"""NERAL V2 generic code-loop contracts; fake model and sandbox, no host execution.

These tests verify routing, retries, output provenance and response boundaries.
They are NOT claims that the 9B model solves arbitrary unseen tasks.
"""
from __future__ import annotations

import ast
import json
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
        "import z3\n"
        "ada=z3.Int('ada'); berta=z3.Int('berta'); cora=z3.Int('cora')\n"
        "all_vars=[ada,berta,cora]\n"
        "solver=z3.Solver()\n"
        "# Regel 1: eindeutige Schichten\n"
        "solver.add(z3.Distinct(*all_vars))\n"
        "# Regel 2: Schichtbereich\n"
        "for v in all_vars: solver.add(z3.And(v>=0,v<=2))\n"
        "# Regel 3: Ada nicht morgens\n"
        "solver.add(ada!=0)\n"
        "# Regel 4: Berta nicht nachts\n"
        "solver.add(berta!=2)\n"
        "# Regel 5: Cora nicht abends\n"
        "solver.add(cora!=1)\n"
        "count=0\n"
        "while solver.check() == z3.sat:\n"
        "    m=solver.model()\n"
        "    a,b,c=[m[v].as_long() for v in all_vars]\n"
        "    assert len({a,b,c})==3\n"
        "    assert all(0<=v<=2 for v in (a,b,c))\n"
        "    assert a!=0\n"
        "    assert b!=2\n"
        "    assert c!=1\n"
        "    count+=1\n"
        "    solver.add(z3.Or([v != m[v] for v in all_vars]))\n"
        "print(count)",
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
        self.temperatures=[]
        self.max_times=[]
        self.replies=list(replies)

    def respond_chat(self, turns, *, max_new_tokens, temperature=None, max_time_seconds=None):
        self.loaded=True
        self.temperatures.append(temperature)
        self.max_times.append(max_time_seconds)
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
            if 'schicht' in question.casefold():
                continue  # Z3 has a separate JSON→compiler regression below.
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

    def test_code_generation_and_repair_are_forced_to_temperature_zero(self):
        rt,_,helios=runtime([
            "\x60\x60\x60python\nraise ValueError('broken')\n\x60\x60\x60",
            "\x60\x60\x60python\nprint(2)\n\x60\x60\x60",
            "Sandbox returned 2",
        ], lambda code: (
            {"status":"FAILED","traceback":"ValueError"} if "raise" in code
            else {"status":"PASSED","output":"2"}))
        result=rt._free_code_loop(turns("Berechne 1 plus 1"),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(helios.temperatures[:2],[0.0,0.0])
        self.assertEqual(helios.max_times[:2],[80.0,80.0])

    def test_z3_shift_generates_json_not_code_and_checks_compiled_models(self):
        proposal = {
            "variables": [
                {"name":"ada","min":0,"max":2},
                {"name":"berta","min":0,"max":2},
                {"name":"cora","min":0,"max":2}],
            "constraints": [
                {"rule_id":"R1","description":"Ada und Berta verschieden","expr":"ada != berta"},
                {"rule_id":"R2","description":"Ada und Cora verschieden","expr":"ada != cora"},
                {"rule_id":"R3","description":"Berta und Cora verschieden","expr":"berta != cora"},
                {"rule_id":"R4","description":"Ada nie Morgen","expr":"ada != 0"},
                {"rule_id":"R5","description":"Berta nie Nacht","expr":"berta != 2"},
                {"rule_id":"R6","description":"Cora nie Abend","expr":"cora != 1"}],
            "objective": "count_all"}
        observed = []
        def sandbox(code):
            ast.parse(code)
            observed.append(code)
            self.assertIn("while solver.check() == z3.sat:",code)
            self.assertEqual(code.count("assert z3.is_true("),1)
            self.assertIn("for rid,rule in zip(rule_ids,rules):",code)
            self.assertIn("solver.add(z3.Or([v != m[v] for v in all_vars]))",code)
            return {"status":"PASSED","output":"2"}
        rt,_,helios=runtime([json.dumps(proposal), "Die Sandbox gab 2 aus."],sandbox)
        question=("Ada, Berta und Cora besetzen Morgen-, Abend- und "
                  "Nachtschicht. Ada darf nicht morgens arbeiten, Berta nicht "
                  "nachts, Cora nicht abends. Wie viele zulässige Schichtpläne?")
        result=rt._free_code_loop(turns(question),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        self.assertFalse(result["verified"])
        self.assertEqual(len(observed),1)
        self.assertNotIn("Python-Codeblock",helios.calls[0][0][0]["content"])
        self.assertIn("JSON-Objekt",helios.calls[0][0][0]["content"])
        self.assertEqual(helios.temperatures[0],0.0)

    def test_z3_bad_json_receives_explicit_error_and_at_most_two_repairs(self):
        payload = {
            "variables":[{"name":"a","min":0,"max":1}],
            "constraints":[{"rule_id":"R1","description":"A ist wahr",
                            "expr":"a == 1"}],
            "objective":"count_all"}
        calls=[]
        rt,_,helios=runtime(["not json", json.dumps(payload), "Ergebnis laut Sandbox"],
                            lambda code:calls.append(code) or
                                {"status":"PASSED","output":"1"})
        r=rt._free_code_loop(turns("Wie viele Schichtpläne für A?"),auto_from="Luna")
        self.assertTrue(r["sandbox_executed"])
        self.assertEqual(r["sandbox_attempts"],2)
        self.assertEqual(len(calls),1)
        self.assertIn("JSON wurde NICHT",str(helios.calls[1][0]))
        self.assertEqual(helios.temperatures[:2],[0.0,0.0])

    def test_z3_extra_schema_key_gets_targeted_repair(self):
        data={"variables":[{"name":"a","min":0,"max":1}],
              "constraints":[{"rule_id":"R1","description":"A ist eins","expr":"a == 1"}],
              "objective":"count_all"}
        with_schema={**data,"schema":{"type":"object"}}
        rt,_,helios=runtime([json.dumps(with_schema),json.dumps(data),
                              "Das Ergebnis lautet 1."],
                             lambda _:{"status":"PASSED","output":"1"})
        result=rt._free_code_loop(
            turns("Wie viele zulässige Schichtpläne für Person A?"),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        # Complete canonical fields are sufficient; schema metadata is ignored.
        self.assertEqual(result["sandbox_attempts"],1)
        self.assertEqual(len(helios.calls),2)

    def test_z3_extra_rules_key_gets_targeted_no_rules_repair(self):
        schema={
            "variables":[{"name":"a","min":0,"max":1}],
            "constraints":[{"rule_id":"R1","description":"Person A erlaubt",
                            "expr":"a == 1"}],
            "objective":"count_all"}
        unexpected={**schema,"rules":["Person A erlaubt"]}
        executed=[]
        rt,_,helios=runtime([
            json.dumps(unexpected),json.dumps(schema),"Die Sandbox zählt eins."
        ],lambda code:executed.append(code) or {"status":"PASSED","output":"1"})
        out=rt._free_code_loop(
            turns("Wie viele zulässige Schichtpläne für Person A?"),auto_from="Luna")
        self.assertTrue(out["sandbox_executed"])
        self.assertEqual(len(executed),1)
        self.assertEqual(out["sandbox_attempts"],2)
        self.assertIn("rules",str(helios.calls[1][0]))
        self.assertIn("verboten",str(helios.calls[1][0]).casefold())

    def test_sympy_symbol_dictionary_keyerror_triggers_precise_repair(self):
        prompt=("Anna, Ben und Cora haben insgesamt 47 Spielchips. "
                "Ben besitzt 5 mehr als Anna. Cora besitzt exakt doppelt "
                "so viele wie Ben. Wie viele Chips hat jede Person?")
        seen=[]
        def tool(code):
            seen.append(code)
            if len(seen)==1:
                return {"status":"FAILED","error_type":"RuntimeError",
                        "traceback":"File /work/run.py line 9: KeyError: 'a'"}
            return {"status":"PASSED","output":"Anna=8; Ben=13; Cora=26"}
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint('bad')\n\x60\x60\x60",
            "\x60\x60\x60python\nprint('fixed')\n\x60\x60\x60",
            "Die Sandbox-Ausgabe ist aufgeführt.",
        ],tool)
        result=rt._free_code_loop(turns(prompt),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(len(seen),2)
        repair=str(helios.calls[1][0])
        self.assertIn("Symbol",repair)
        self.assertIn("result[a]",repair)
        self.assertIn("result['a']",repair)

    def test_general_three_person_relations_produce_exact_source_equations(self):
        from neral_three_tier_runtime import _source_relational_equations
        source=("Mira, Lino und Tessa haben insgesamt 47 Murmeln. "
                "Lino besitzt 5 mehr als Mira. Tessa besitzt exakt "
                "doppelt so viele wie Lino. Wie viele hat jede Person?")
        guidance=_source_relational_equations(source)
        self.assertIn("a+b+c=47", guidance)
        self.assertIn("b=a+5", guidance)
        self.assertIn("c=2*b", guidance)
        self.assertIn("Mira", guidance)
        self.assertIn("Tessa", guidance)
        self.assertIn("sp.solve(eqs,[a,b,c],dict=True)",guidance)
        self.assertIn("assert all(eq.lhs.subs(sol)==eq.rhs.subs(sol) for eq in eqs)",guidance)
        self.assertIn("print(f'Mira={sol[a]}; Lino={sol[b]}; Tessa={sol[c]}')",guidance)
        self.assertIsNone(_source_relational_equations(
            "Anna und Ben haben zusammen sieben Dinge."))

    def test_loan_source_first_year_interest_then_repayment_then_interest(self):
        from neral_three_tier_runtime import _source_loan_equations
        q=("Ein Darlehen von 1600 Euro wird jedes Jahr mit 5 Prozent "
           "verzinst. Nach dem ersten Jahr leiste ich eine Sondertilgung "
           "von 300 Euro. Wie hoch ist die Restschuld nach dem zweiten Jahr?")
        formula=_source_loan_equations(q)
        self.assertIn("P=1600",formula)
        self.assertIn("r=5",formula)
        self.assertIn("T=300",formula)
        self.assertIn("Rest1=P*(1+r/100)",formula)
        self.assertIn("Rest2=(Rest1-T)*(1+r/100)",formula)
        self.assertIsNone(_source_loan_equations("Berechne 17 plus 19"))
        from neral_three_tier_runtime import _source_loan_output_ok
        self.assertTrue(_source_loan_output_ok(q,"1449.00"))
        self.assertTrue(_source_loan_output_ok(q,"Restschuld nach 2 Jahren: 1449,00 Euro"))
        self.assertFalse(_source_loan_output_ok(q,"1449.01"))
        self.assertIsNone(_source_loan_output_ok("Berechne 17 plus 19","1449.00"))

    def test_loan_assertion_error_triggers_source_grounded_repair(self):
        q=("Ein Darlehen von 1600 Euro wird jedes Jahr mit 5 Prozent "
           "verzinst. Nach dem ersten Jahr leiste ich eine Sondertilgung "
           "von 300 Euro. Wie hoch ist die Restschuld nach dem zweiten Jahr?")
        attempts=[]
        def tool(code):
            attempts.append(code)
            if len(attempts)==1:
                return {"status":"FAILED","error_type":"RuntimeError",
                        "traceback":"AssertionError at Rest1*Zinsen+Rest1-Sondertilgung==Rest2"}
            return {"status":"PASSED","output":"1449.00"}
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint('incorrect')\n\x60\x60\x60",
            "\x60\x60\x60python\nprint('1449.00')\n\x60\x60\x60",
            "Die Restschuld ist aus der Sandbox.",
        ],tool)
        result=rt._free_code_loop(turns(q),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(len(attempts),2)
        self.assertIn("Rest2=(Rest1-T)*(1+r/100)",str(helios.calls[0][0]))
        self.assertIn("Sondertilgung",str(helios.calls[1][0]))
        self.assertEqual(result["sandbox_attempts"],2)

    def test_discount_source_hint_compiles_sequential_decimal_template(self):
        from neral_three_tier_runtime import _source_sequential_percent_hint
        q=("Ein Produkt kostet 200 Euro. Zuerst werden 10 Prozent Rabatt "
           "abgezogen, vom neuen Preis weitere 5 Prozent. Danach kommen "
           "19 Prozent Mehrwertsteuer hinzu. Was ist der Endpreis?")
        hint=_source_sequential_percent_hint(q)
        self.assertIn("Decimal('200')",hint)
        self.assertIn("Decimal('10')",hint)
        self.assertIn("Decimal('5')",hint)
        self.assertIn("Decimal('19')",hint)
        self.assertIn("ROUND_HALF_UP",hint)
        self.assertNotIn("203.49",hint)
        self.assertIsNone(_source_sequential_percent_hint("Hallo."))
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint('203.49')\n\x60\x60\x60",
            "Sandbox-Ergebnis liegt vor.",
        ],lambda _:{"status":"PASSED","output":"203.49"})
        response=rt._free_code_loop(turns(q),auto_from="Luna")
        self.assertTrue(response["sandbox_executed"])
        self.assertIn("ROUND_HALF_UP",helios.calls[0][0][0]["content"])

    def test_discount_mismatch_retries_using_original_percentage_source(self):
        from neral_three_tier_runtime import _source_sequential_percent_ok
        source=("Ein Produkt kostet 200 Euro. Zuerst werden 10 Prozent Rabatt "
                "abgezogen, vom neuen Preis weitere 5 Prozent. Danach kommen "
                "19 Prozent Mehrwertsteuer hinzu. Was ist der endgültige Preis "
                "auf den Cent genau?")
        self.assertFalse(_source_sequential_percent_ok(source,"Preis: 200 Euro"))
        self.assertTrue(_source_sequential_percent_ok(source,"Preis: 203.49 Euro"))
        self.assertIsNone(_source_sequential_percent_ok(
            "Warum ist der Himmel blau?","203.49"))
        seen=[]
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint(200)\n\x60\x60\x60",
            "\x60\x60\x60python\nprint('203.49')\n\x60\x60\x60",
            "Die Sandbox lieferte den korrekten Preis.",
        ],lambda code:seen.append(code) or {"status":"PASSED",
            "output":"Preis: 200 Euro" if len(seen)==1 else "Preis: 203.49 Euro"})
        result=rt._free_code_loop(turns(source),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(result["sandbox_attempts"],2)
        self.assertEqual(len(seen),2)
        self.assertIn("Rabatt",str(helios.calls[1][0]))

    def test_relational_word_problem_requires_symbolic_equations(self):
        question=("Anna, Ben und Cora haben insgesamt 31 Spielchips. "
                  "Ben besitzt 3 mehr als Anna. Cora besitzt exakt doppelt "
                  "so viele wie Ben. Wie viele Chips hat jede Person?")
        seen=[]
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint('Anna=4; Ben=7; Cora=14')\n\x60\x60\x60",
            "Die Sandbox hat die Zahlen berechnet.",
        ],lambda code:seen.append(code) or
                  {"status":"PASSED","output":"Anna=4; Ben=7; Cora=14"})
        result=rt._free_code_loop(turns(question),auto_from="Luna")
        self.assertTrue(result["sandbox_executed"])
        instruction=helios.calls[0][0][0]["content"]
        self.assertIn("sympy.Eq",instruction)
        self.assertIn("sympy.solve",instruction)
        self.assertIn("maximal 25",instruction)
        self.assertNotIn("itertools.permutations",instruction)
        self.assertIn("assert",instruction)
        self.assertNotIn("NERAL Z3-JSON-MODUS",instruction)

    def test_fraction_equation_rejects_wrong_executed_number_then_repairs(self):
        from neral_three_tier_runtime import _source_fraction_equation_ok
        source="Bestimme die Lösung x der Gleichung 3/4 mal x plus 5/6 gleich 23/6."
        self.assertFalse(_source_fraction_equation_ok(source,"11/9"))
        self.assertTrue(_source_fraction_equation_ok(source,"4"))
        self.assertIsNone(_source_fraction_equation_ok("Was ist der Sinn des Lebens?","4"))
        seen=[]
        def tool(code):
            seen.append(code)
            return {"status":"PASSED","output":"11/9" if len(seen)==1 else "4"}
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint('11/9')\n\x60\x60\x60",
            "\x60\x60\x60python\nprint(4)\n\x60\x60\x60",
            "Die Sandbox-Ausgabe ist 4.",
        ], tool)
        result=rt._free_code_loop(turns(source),auto_from="Luna")
        self.assertEqual(len(seen),2)
        self.assertTrue(result["sandbox_executed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["sandbox_attempts"],2)
        self.assertIn("Eingabegleichung",str(helios.calls[1][0]))

    def test_z3_prompt_trigger_words_and_normal_arithmetic_boundary(self):
        from neral_three_tier_runtime import _is_constraint_task
        for text in ("Schicht", "Dienstplan", "zuordnen", "verteilen",
                     "nicht zusammen", "nie", "höchstens", "mindestens",
                     "zulässige", "Plan", "Ausschlussbedingung"):
            with self.subTest(text=text):
                self.assertTrue(_is_constraint_task("Bitte "+text+" alle Fälle"))
        self.assertFalse(_is_constraint_task("Berechne 17 plus 19"))

    def test_constraint_code_must_recheck_rules_and_block_enumerated_models(self):
        from neral_three_tier_runtime import _constraint_code_issue
        self.assertIsNotNone(_constraint_code_issue(
            "import z3\nsolver=z3.Solver()\nprint(3)", counted=True))
        self.assertIsNotNone(_constraint_code_issue(
            "import z3\nx=z3.Int('x')\nsolver=z3.Solver()\n"
            "solver.add(x!=0)\nwhile solver.check() == z3.sat:\n"
            " m=solver.model()\n solver.add(z3.Or([x!=m[x]]))\nprint(3)", counted=True))
        self.assertIsNotNone(_constraint_code_issue(
            "from itertools import permutations\nprint(3)", counted=True))

    def test_noncompliant_constraint_script_never_executes_in_docker(self):
        seen=[]
        rt,_,helios=runtime([
            "\x60\x60\x60python\nprint(3)\n\x60\x60\x60"]*3,
            lambda code:seen.append(code) or {"status":"PASSED","output":"3"})
        result=rt._free_code_loop(turns(
            "Wie viele zulässige Schichtpläne gibt es bei diesen Ausschlüssen?"),
            auto_from="Luna")
        self.assertEqual(seen,[])
        self.assertFalse(result["sandbox_executed"])
        self.assertFalse(result["verified"])
        self.assertEqual(helios.temperatures,[0.0,0.0,0.0,0.0])

    def test_sanitized_gpu_failure_categories_without_model_output_leak(self):
        from neral_three_tier_runtime import _safe_runtime_error_code
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("HF response did not finish inside its token/time budget")),
            "hf_time_budget")
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("HF token budget exhausted")), "hf_token_budget")
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("Unclosed private model reasoning")), "hf_reasoning_unclosed")
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("Incomplete or untrusted HF final answer")), "hf_invalid_final")
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("Insufficient free GPU memory; secret text")),
            "gpu_memory")
        self.assertEqual(_safe_runtime_error_code(
            ValueError("Bounded local HF context exceeded")), "context_exceeded")
        self.assertEqual(_safe_runtime_error_code(
            RuntimeError("unrecognized sensitive details")), "runtime_unavailable")

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
        self.assertEqual(len(helios.calls),4)  # one honest Helios fallback attempt
        self.assertNotIn("12",r["text"])
        self.assertEqual([x["stage"] for x in r["sandbox_failure_trace"]],
                         ["docker", "docker", "docker"])

    def test_missing_container_does_not_retry_model_or_run_host_python(self):
        seen=[]
        rt,_,helios=runtime(["\x60\x60\x60python\nprint(2+3)\n\x60\x60\x60"],
            lambda code:seen.append(code) or {"status":"UNAVAILABLE","error_type":"SandboxUnavailable"})
        r=rt.answer(turns("Wie viel sind 2 plus 3?"),"Luna")
        self.assertEqual(len(helios.calls),2)  # one fallback chat attempt
        self.assertEqual(len(seen),1)
        self.assertEqual(r["sandbox_failure_code"],"docker_unavailable")
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
