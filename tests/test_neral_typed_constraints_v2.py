"""RED/green: exact typed finite-domain counting, never model prose guessing."""
from __future__ import annotations
import itertools
import json
import math
import re
import unittest
from neral_typed_constraints_v2 import solve_typed_constraints, InvalidTypedConstraints, PREFIX
from neral_formal_constraints import try_solve_formal_constraints, InvalidFormalConstraints
from neral_tir_regression_guard import choose_guarded_tir_reply
from scripts.neral_blind_100_cases import build as old100

def variable(values, kind="int"):
    return {"type":kind,"domain":list(values)}
def v(name):
    return {"var":name}
def c(value):
    return {"const":value}
def cmp(lhs,op,rhs):
    return {"compare":{"left":lhs,"op":op,"right":rhs}}
def schema(vars, clauses, *, query="count"):
    return {"version":2,"variables":vars,"constraints":clauses,"query":query}
def tagged(doc):
    return PREFIX+json.dumps(doc,separators=(",",":"),sort_keys=True)

class ModelCounting(unittest.TestCase):
    def solve(self,doc):
        return solve_typed_constraints(tagged(doc))
    def test_graph_three_coloring_cycle_exact_count(self):
        names=[f"C{i}" for i in range(4)]
        vv={x:variable([0,1,2]) for x in names}
        rules=[cmp(v("C0"),"==",c(0))]
        rules.extend(cmp(v(f"C{a}"),"!=",v(f"C{b}"))
                     for a,b in [(0,1),(1,2),(2,3),(3,0)])
        self.assertEqual(self.solve(schema(vv,rules)),6)
        self.assertEqual(self.solve(schema(vv,rules,query="satisfiable")),1)
        rules.append(cmp(v("C1"),"==",v("C0")))
        self.assertEqual(self.solve(schema(vv,rules)),0)
        self.assertEqual(self.solve(schema(vv,rules,query="satisfiable")),0)

    def test_pin_all_different_digit_sum_even_second_and_order(self):
        vv={x:variable(range(6)) for x in "ABCD"}
        rules=[{"all_different":list("ABCD")},
          cmp({"sum":[v(x) for x in "ABCD"]},"==",c(11)),
          cmp(v("A"),"<",v("D")),
          cmp({"mod":{"term":v("B"),"by":2}},"==",c(0))]
        gold=sum(1 for p in itertools.product(range(6),repeat=4)
                 if len(set(p))==4 and sum(p)==11 and p[0]<p[3] and p[1]%2==0)
        self.assertEqual(self.solve(schema(vv,rules)),gold)

    def test_ordering_eight_distinct_ranks_permutation_optimization(self):
        n=8
        vars={f"R{i}":variable(range(1,n+1)) for i in range(n)}
        pairs=[(0,2),(0,4),(1,5),(3,6),(5,7)]
        rules=[{"all_different":list(vars)}]
        rules.extend(cmp(v(f"R{a}"),"<",v(f"R{b}")) for a,b in pairs)
        gold=sum(1 for order in itertools.permutations(range(n)) if all(
            order.index(a)<order.index(b) for a,b in pairs))
        self.assertEqual(self.solve(schema(vars,rules)),gold)

    def test_boolean_exact_cardinality_and_forbidden_distance_two(self):
        n=6;k=2
        vv={f"B{i}":variable([False,True],"bool") for i in range(n)}
        rules=[{"exactly":{"vars":list(vv),"value":True,"count":k}},
           cmp(v("B0"),"!=",v("B1"))]
        for i in range(n-2):
            rules.append({"not":{"and":[
                cmp(v(f"B{i}"),"==",c(True)),
                cmp(v(f"B{i+2}"),"==",c(True))]}})
        gold=sum(1 for vals in itertools.product((False,True),repeat=n)
                 if sum(vals)==k and vals[0]!=vals[1]
                 and all(not(vals[i] and vals[i+2]) for i in range(n-2)))
        self.assertEqual(self.solve(schema(vv,rules)),gold)

    def test_integer_sum_comparison_biimplication_modulo(self):
        vv={x:variable(range(6)) for x in "ABCD"}
        rules=[
          cmp({"sum":[v("A"),v("B")]},"==",c(7)),
          cmp(v("C"),"!=",v("D")),
          {"iff":[cmp(v("A"),"<",v("C")),cmp(v("B"),"<",v("D"))]},
          cmp({"mod":{"term":{"sum":[v("A"),v("D")]},"by":2}},"==",c(0))]
        gold=sum(1 for a,b,c_,d in itertools.product(range(6),repeat=4)
                 if a+b==7 and c_!=d and (a<c_)==(b<d) and (a+d)%2==0)
        self.assertEqual(self.solve(schema(vv,rules)),gold)

    def test_known_old_logic_02_graph_and_06_binary_no_routing_by_question(self):
        known={x["id"]:x for x in old100()}
        graph=known["blind100-logic-02"]
        self.assertIn("Color vertices",graph["question"])
        n=int(re.search(r"Color vertices 0\.\.(\d+)",graph["question"]).group(1))+1
        # Parse only an independently generated TRUSTED TEST FIXTURE. The
        # production solver never reads or regex-routes natural-language text.
        import ast
        edges=ast.literal_eval(graph["question"].split("edges ",1)[1].split(" must differ",1)[0])
        clauses=[cmp(v("C0"),"==",c(0))]
        clauses += [cmp(v(f"C{a}"),"!=",v(f"C{b}")) for a,b in edges]
        self.assertEqual(self.solve(schema(
            {f"C{i}":variable(range(3)) for i in range(n)},clauses)),graph["gold_integer"])
        binary=known["blind100-logic-06"]
        m=re.search(r"length (\d+) with exactly (\d+) ones",binary["question"])
        n,k=map(int,m.groups())
        vv={f"B{i}":variable([False,True],"bool") for i in range(n)}
        clauses=[{"exactly":{"vars":list(vv),"value":True,"count":k}},
                  cmp(v("B0"),"!=",v("B1"))]
        for i in range(n-2):
            clauses.append({"not":{"and":[cmp(v(f"B{i}"),"==",c(True)),
                                          cmp(v(f"B{i+2}"),"==",c(True))]}})
        self.assertEqual(self.solve(schema(vv,clauses)),binary["gold_integer"])

class RejectAmbiguityAndUnsafeInputs(unittest.TestCase):
    def test_type_errors_unknown_vars_and_invalid_modulo(self):
        base=schema({"A":variable([0,1]),"B":variable([False,True],"bool")},
                    [cmp(v("A"),"==",c(1))])
        invalid=[
           {**base,"constraints":[cmp(v("A"),"==",c(True))]},
           {**base,"constraints":[cmp(v("Z"),"==",c(0))]},
           {**base,"constraints":[cmp({"sum":[v("B"),c(1)]},"==",c(2))]},
           {**base,"constraints":[cmp({"mod":{"term":v("A"),"by":0}},"==",c(0))]},
           {**base,"constraints":[cmp(v("B"),"<",c(True))]},
           {**base,"constraints":[{"all_different":["A","B"]}]},
           {**base,"constraints":[{"exactly":{"vars":["B","A"],"value":True,"count":1}}]},
           {**base,"constraints":[{"all_different":["A","A"]}]},
           {**base,"constraints":[cmp(v("A"),"==",c(1))],"ignored":"extra"},
           {**base,"constraints":[]},
           {**base,"variables":{"A":{"type":"int","domain":[True,1]}}},
           {**base,"variables":{"A":variable([0,1])},"query":"unknown"},
           {**base,"constraints":[{"iff":[cmp(v("A"),"==",c(1))]}]},
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(InvalidTypedConstraints):
                solve_typed_constraints(tagged(item))

    def test_reject_unsafe_large_search_spaces_and_schema_size(self):
        vars={f"A{i}":variable(range(12)) for i in range(8)}
        with self.assertRaisesRegex(InvalidTypedConstraints,"Search space"):
            solve_typed_constraints(tagged(schema(vars,[cmp(v("A0"),"==",c(0))])))
        with self.assertRaises(InvalidTypedConstraints):
            solve_typed_constraints(PREFIX+" "*12000)
        with self.assertRaises(InvalidTypedConstraints):
            solve_typed_constraints(PREFIX+'{"version":2,"version":2}')
        with self.assertRaises(InvalidTypedConstraints):
            solve_typed_constraints(tagged(schema({"X":variable([0])},
                                                 [cmp(v("X"),"==",c(0))]*33)))
    def test_no_natural_language_v2_promotion_or_partial_json(self):
        doc=schema({"A":variable(range(3))},[cmp(v("A"),"==",c(2))])
        self.assertIsNone(solve_typed_constraints("Count solutions of "+tagged(doc)))
        self.assertIsNone(try_solve_formal_constraints("Count solutions of "+tagged(doc)))
        with self.assertRaises(InvalidTypedConstraints):
            solve_typed_constraints(tagged(doc)[:-7])
    def test_formal_solver_dispatch_and_guard_deny_model_authored_schema(self):
        doc=schema({"X":variable(range(5))},[cmp(v("X"),">",c(2))])
        self.assertEqual(try_solve_formal_constraints(tagged(doc)),2)
        fake={"text":"2","mode":"HIGH","verified":True,
              "model_generated":True,"verification_status":"formal_schema_exact",
              "verification_scope":"explicit_finite_domain_schema",
              "tool_used":"neral_logic_exact","tool_calls_succeeded":1,
              "calls_used":1,"seconds":0.01}
        selected=choose_guarded_tir_reply(fake,[
            {"role":"system","content":"Base"},{"role":"user","content":"Some text puzzle"}],
            lambda _: "FINAL_ANSWER: 3")
        self.assertFalse(selected["verified"])
        self.assertEqual(selected["text"],"FINAL_ANSWER: 3")

if __name__=="__main__":
    unittest.main()
