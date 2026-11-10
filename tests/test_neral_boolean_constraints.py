"""RED contracts: generic safe Boolean AST within bounded finite-domain solver."""
from __future__ import annotations
import itertools
import json
import unittest

from neral_formal_constraints import (
    PREFIX, InvalidFormalConstraints, try_solve_formal_constraints,
)


def count(variables, bool_rules, *, query="count"):
    question = PREFIX + json.dumps({
        "variables": variables, "rules": [], "bool_rules": bool_rules,
        "query": query,
    }, separators=(",", ":"))
    return try_solve_formal_constraints(question)


class BooleanAlgebraContracts(unittest.TestCase):
    def test_generic_xor_and_predicate_equality(self):
        vars = {v: [False, True] for v in "ABC"}
        expr = {"op":"eq","args":[
            {"op":"xor","args":["A","B","C"]}, True,
        ]}
        self.assertEqual(count(vars,[expr]),4)
        expr["args"][1] = False
        self.assertEqual(count(vars,[expr]),4)

    def test_or_implies_not_equality(self):
        vars={v:[False,True] for v in "ABCD"}
        rules=[
            {"op":"or","args":["A","B"]},
            {"op":"implies","args":["C","D"]},
            {"op":"eq","args":["A",{"op":"not","args":["D"]}]},
        ]
        correct=0
        for A,B,C,D in itertools.product((False,True),repeat=4):
            correct+=bool((A or B) and ((not C) or D) and
                          (A == (not D)))
        self.assertEqual(count(vars,rules),correct)

    def test_threshold_at_least_and_composed_and_not(self):
        vars={v:[False,True] for v in "ABCDE"}
        rules=[
            {"op":"at_least","args":list("ABCDE"),"min":3},
            {"op":"not","args":[{"op":"and","args":["A","E"]}]},
        ]
        truth=sum(
            sum(bits)>=3 and not (bits[0] and bits[-1])
            for bits in itertools.product((False,True),repeat=5)
        )
        self.assertEqual(count(vars,rules),truth)
        self.assertEqual(count(vars,rules,query="satisfiable"),1)

    def test_can_combine_ordering_with_boolean_guards(self):
        vars={"X":[1,2,3],"Y":[1,2,3],"P":[False,True]}
        data={"variables":vars,
              "rules":[{"left":"X","op":"<","right":"Y"}],
              "bool_rules":[{"op":"eq","args":["P",True]}],
              "query":"count"}
        self.assertEqual(try_solve_formal_constraints(
            PREFIX+json.dumps(data)),3)

    def test_rejects_unsafe_ast_and_invalid_domains(self):
        vars={v:[False,True] for v in "AB"}
        bad=[
            {"op":"eval","args":["A"]},
            {"op":"not","args":["A","B"]},
            {"op":"and","args":[]},
            {"op":"or","args":["A","MISSING"]},
            {"op":"implies","args":["A"]},
            {"op":"eq","args":["A",1]},
            {"op":"at_least","args":["A","B"],"min":True},
            {"op":"at_least","args":["A","B"],"min":3},
            {"op":"at_least","args":["A","A"],"min":1},
            {"op":"or","args":["A"],"code":"__import__('os')"},
            {"op":"not","args":[{"op":"not","args":[{"op":"not",
                    "args":[{"op":"not","args":[{"op":"not",
                         "args":["A"]}]}]}]}]},
        ]
        for rule in bad:
            with self.subTest(rule=rule):
                with self.assertRaises(InvalidFormalConstraints):
                    count(vars,[rule])
        with self.assertRaises(InvalidFormalConstraints):
            count({"A":[0,1],"B":[False,True]},
                  [{"op":"eq","args":["A","B"]}])

    def test_boolean_extraction_requires_source_evidence_and_unverified_status(self):
        from neral_turn0_extractor import parse_extracted_constraints
        question = ("A and B can each be True or False. "
                    "A XOR B must be True. "
                    "How many assignments satisfy all conditions?")
        schema = {
            "variables":{"A":[False,True],"B":[False,True]},
            "rules":[],
            "bool_rules":[{"op":"eq","args":[
                {"op":"xor","args":["A","B"]},True]}],
            "query":"count"}
        proposal={
            "kind":"formal", "complete":True, "schema":schema,
            "evidence":["A and B can each be True or False",
                        "A XOR B must be True",
                        "How many assignments satisfy all conditions?"],
        }
        self.assertEqual(parse_extracted_constraints(
            question,json.dumps(proposal)),2)
        proposal["evidence"][1]="Unsupported invented condition"
        self.assertIsNone(parse_extracted_constraints(
            question,json.dumps(proposal)))

    def test_compact_generic_ordering_schema_counts_partial_orders(self):
        import math
        schema={
            "ordering":list("ABCDEFG"),
            "rules":[
                {"left":"A","op":"<","right":"B"},
                {"left":"C","op":"<","right":"D"},
            ],
            "query":"count",
        }
        self.assertEqual(try_solve_formal_constraints(
            PREFIX+json.dumps(schema)),math.factorial(7)//4)
        # No raw variable domains or benchmark-specific labels needed.
        schema["ordering"]=["North","East","South","West"]
        schema["rules"]=[{"left":"East","op":"<","right":"West"}]
        self.assertEqual(try_solve_formal_constraints(
            PREFIX+json.dumps(schema)),12)

    def test_compact_ordering_rejects_ambiguous_or_unsafe_declarations(self):
        for tokens in (["A","A"],[],["A"],["A","B",1],
                       ["A","__import__('os')"],list("ABCDEFGHI")):
            schema={"ordering":tokens,"rules":[],"query":"count"}
            with self.subTest(tokens=tokens), self.assertRaises(InvalidFormalConstraints):
                try_solve_formal_constraints(PREFIX+json.dumps(schema))
        schema={"ordering":["A","B"],"rules":[],"query":"count",
                "variables":{"A":[1,2]}}
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX+json.dumps(schema))

    def test_all_different_full_assignment_uses_bounded_permutations(self):
        import math
        names="ABCDEFG"
        variables={name:list(range(1,8)) for name in names}
        formula={
            "variables":variables,
            "rules":[
                {"left":"A","op":"<","right":"B"},
                {"left":"C","op":"<","right":"D"},
            ],
            "all_different":[list(names)],
            "query":"count",
        }
        # Two independent pairwise precedence constraints: 7!/4.
        self.assertEqual(try_solve_formal_constraints(
            PREFIX+json.dumps(formula)),math.factorial(7)//4)

    def test_all_different_without_full_coverage_stays_bounded(self):
        names="ABCDEFG"
        base={
            "variables":{v:list(range(1,8)) for v in names},
            "rules":[],"query":"count",
        }
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX+json.dumps(base))
        base["all_different"]=[list("ABCDEF")]
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX+json.dumps(base))

    def test_unbounded_permutations_rejected_even_with_all_different(self):
        names="ABCDEFGH"
        schema={
            "variables":{v:list(range(1,10)) for v in names},
            "rules":[],"all_different":[list(names)],"query":"count",
        }
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX+json.dumps(schema))

    def test_development_example_is_generic_not_hardcoded(self):
        variables={v:[False,True] for v in "ABCDEFG"}
        rules=[
            {"op":"eq","args":[{"op":"xor","args":["A","C","F"]},True]},
            {"op":"or","args":["B","D"]},
            {"op":"eq","args":["E",{"op":"not","args":["F"]}]},
            {"op":"not","args":[{"op":"and","args":["A","G"]}]},
            {"op":"implies","args":["C","G"]},
            {"op":"at_least","args":list("ABCDEFG"),"min":4},
        ]
        expected=0
        for bits in itertools.product((False,True),repeat=7):
            A,B,C,D,E,F,G=bits
            expected+=bool(
                (A^C^F) and (B or D) and (E==(not F))
                and not(A and G) and ((not C) or G)
                and sum(bits)>=4
            )
        self.assertEqual(count(variables,rules),expected)


if __name__=="__main__":
    unittest.main()
