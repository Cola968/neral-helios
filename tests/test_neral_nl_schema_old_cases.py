"""Old, already-disclosed logic examples: deterministic fixture schemas only.

These tests do NOT run QLoRA or base-Qwen, and cannot prove NL extraction
accuracy. No sealed prospective-100 corpus is imported or opened.
"""
import ast
import json
import re
import unittest
from itertools import permutations

from neral_nl_schema_bridge import assess_dual_extraction
from scripts.neral_blind_100_cases import build as historical_old100

def iv(domain):
    return {"type": "int", "domain": list(domain)}

def bv():
    return {"type": "bool", "domain": [False, True]}

def var(x):
    return {"var": x}

def const(x):
    return {"const": x}

def cmp(a, op, b):
    return {"compare": {"left": a, "op": op, "right": b}}

def doc(variables, constraints):
    return {"version": 2, "variables": variables,
            "constraints": constraints, "query": "count"}

def model_fake(schema):
    def extract(_):
        return json.dumps(schema)
    return extract

class OldKnownLogicProbes(unittest.TestCase):
    def verify_consensus_fixture(self, question, schema, expected):
        result = assess_dual_extraction(question, model_fake(schema), model_fake(schema))
        self.assertEqual(result["status"], "consensus_unverified", result)
        self.assertEqual(result["proof_backend"], "z3_unsat")
        self.assertEqual(result["candidate_count"], expected)
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])

    def test_historical_logic_02_coloring(self):
        old = {row["id"]: row for row in historical_old100()}["blind100-logic-02"]
        question = old["question"]
        n = int(re.search(r"Color vertices 0\.\.(\d+)", question).group(1)) + 1
        edges = ast.literal_eval(question.split("edges ", 1)[1].split(" must differ", 1)[0])
        schema = doc({f"C{i}": iv(range(3)) for i in range(n)},
                     [cmp(var("C0"), "==", const(0))] +
                     [cmp(var(f"C{a}"), "!=", var(f"C{b}")) for a,b in edges])
        self.verify_consensus_fixture(question, schema, old["gold_integer"])

    def test_historical_logic_06_binary_distance(self):
        old = {row["id"]: row for row in historical_old100()}["blind100-logic-06"]
        question = old["question"]
        n, k = map(int, re.search(r"length (\d+) with exactly (\d+) ones", question).groups())
        schema = doc({f"B{i}": bv() for i in range(n)},
            [{"exactly": {"vars": [f"B{i}" for i in range(n)],
                          "value": True, "count": k}},
             cmp(var("B0"), "!=", var("B1"))] +
            [{"not": {"and": [cmp(var(f"B{i}"), "==", const(True)),
                              cmp(var(f"B{i+2}"), "==", const(True))]}}
             for i in range(n-2)])
        self.verify_consensus_fixture(question, schema, old["gold_integer"])

    def test_21453_code_from_prior_pilot_exact_wording(self):
        # This is the already-public reference task from PR #539, not new100.
        question = (
            "Ein fünfstelliger Code verwendet 1, 2, 3, 4 und 5 genau einmal. "
            "Die 2 steht vor der 4, nicht direkt davor. Zwischen der 1 und 5 "
            "steht genau eine Ziffer. Die 3 steht unmittelbar rechts von der 5. "
            "Die 4 steht nicht am Rand. Bestimme alle gültigen Codes."
        )
        names = [f"P{i}" for i in range(1,6)]
        schema = doc({name: iv(range(5)) for name in names}, [
            {"all_different": names},
            cmp(var("P4"), ">=", {"sum": [var("P2"), const(2)]}),
            cmp(var("P2"), "<", var("P4")),
            {"or": [
                cmp(var("P1"), "==", {"sum": [var("P5"), const(2)]}),
                cmp(var("P5"), "==", {"sum": [var("P1"), const(2)]}),
            ]},
            cmp(var("P3"), "==", {"sum": [var("P5"), const(1)]}),
            cmp(var("P4"), "!=", const(0)),
            cmp(var("P4"), "!=", const(4)),
        ])
        # Independent position-permutation oracle verifies exact answer string.
        solutions = []
        for seq in permutations("12345"):
            pos = {digit: seq.index(digit) for digit in seq}
            if (pos["4"] - pos["2"] >= 2
                and abs(pos["1"] - pos["5"]) == 2
                and pos["3"] == pos["5"]+1
                and pos["4"] in (1,2,3)):
                solutions.append("".join(seq))
        self.assertEqual(solutions, ["21453"])
        self.verify_consensus_fixture(question, schema, len(solutions))


if __name__ == "__main__":
    unittest.main()
