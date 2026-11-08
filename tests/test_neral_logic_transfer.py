"""RED contracts for NERAL exact combinatorial reasoning beyond XOR/old templates.

This tests the system solver and its declared provenance, not neural weights.
The published v7 developer examples are already consumed and not holdouts.
"""
from __future__ import annotations
from itertools import permutations, product
import math
import random
import unittest

from neral_logic_transfer import trusted_transfer_logic_question
from neral_logic_engine import trusted_logic_question
from neral_think import think_reply
from scripts.neral_v7_logic_pilot import cases


def chat(q):
    return [{"role":"system","content":"NERAL local"},
            {"role":"user","content":q}]


class LogicTransferTests(unittest.TestCase):
    def test_all_consumed_v7_transfer_48_exact_not_a_weight_result(self):
        suite=cases("transfer")
        self.assertEqual(len(suite),48)
        for entry in suite:
            with self.subTest(entry=entry["id"]):
                result=trusted_logic_question(entry["prompt"])
                self.assertIsNotNone(result)
                self.assertEqual(result["answer"],entry["answer"])
                self.assertEqual(result["method"],"finite_logic_exact")
                self.assertIn("certificate",result)

    def test_independent_random_dag_paths_with_more_vertices(self):
        rng=random.Random(499991)
        for trial in range(64):
            n=rng.randint(5,18)
            edges=sorted({(u,v) for u in range(n) for v in range(u+1,n)
                          if rng.randrange(5)==0})
            start=rng.randrange(n-2)
            end=rng.randrange(start+1,n)
            dp=[0]*n
            dp[start]=1
            for u in range(start,n):
                for a,b in edges:
                    if a==u:
                        dp[b]+=dp[u]
            q=(f"A DAG on vertices 0 to {n-1} contains only these directed edges "
               f"{edges}. Count different directed paths from {start} to {end}.")
            result=trusted_transfer_logic_question(q)
            self.assertIsNotNone(result,q)
            self.assertEqual(int(result["answer"]),dp[end])
            self.assertEqual(result["certificate"]["rule"],"dag_path_count")

    def test_independent_random_ordering_constraints(self):
        rng=random.Random(509993)
        for trial in range(64):
            n=rng.randint(3,7)
            first=rng.randrange(n)
            last=rng.randrange(n)
            before,after=rng.sample(range(n),2)
            objects=",".join(str(i) for i in range(n))
            q=(f"Arrange objects {objects}. Object {first} cannot be first; "
               f"object {last} cannot be last; and object {before} "
               f"must come before {after}. How many valid orders exist?")
            result=trusted_transfer_logic_question(q)
            expected=sum(p[0]!=first and p[-1]!=last
                         and p.index(before)<p.index(after)
                         for p in permutations(range(n)))
            self.assertIsNotNone(result,q)
            self.assertEqual(int(result["answer"]),expected)

    def test_independent_random_periodic_services_and_boundaries(self):
        rng=random.Random(509997)
        for trial in range(64):
            now=rng.randint(10,2000)
            services=[(rng.randint(0,25),rng.randint(1,90),rng.randint(0,50))
                      for _ in range(rng.randint(2,5))]
            prompt=(f"Current minute is {now}. Services are given as "
                    f"(first departure, frequency, journey time)={services}. "
                    f"Each service repeats on its frequency from the first departure. "
                    f"What is the earliest absolute arrival minute from now?")
            result=trusted_transfer_logic_question(prompt)
            arrivals=[first + max(0,-((first-now)//frequency))*frequency + dur
                      for first,frequency,dur in services]
            self.assertIsNotNone(result,prompt)
            self.assertEqual(int(result["answer"]),min(arrivals))

    def test_independent_divisibility_lcm_not_unbounded_enumeration(self):
        rng=random.Random(509998)
        for trial in range(64):
            lo=rng.randint(-100,300)
            hi=lo+rng.randint(1,600)
            d1=rng.randint(1,40)
            d2=rng.randint(1,40)
            q=(f"Between {lo} and {hi}, endpoints included, how many integers "
               f"are divisible by exactly one of {d1} and {d2}?")
            expected=sum((x%d1==0)!=(x%d2==0) for x in range(lo,hi+1))
            result=trusted_transfer_logic_question(q)
            self.assertIsNotNone(result,q)
            self.assertEqual(int(result["answer"]),expected)
            self.assertEqual(result["certificate"]["rule"],"exclusive_divisibility_lcm")

    def test_dag_cyclic_and_out_of_bounds_fail_closed(self):
        rejected=(
            "A DAG on vertices 0 to 3 contains only these directed edges [(0, 1), (1, 0)]. Count different directed paths from 0 to 3.",
            "A DAG on vertices 0 to 3 contains only these directed edges [(0, 4)]. Count different directed paths from 0 to 3.",
            "A DAG on vertices 0 to 3 contains only these directed edges [(0, 1)]. Count different directed paths from 0 to 9.",
            "A DAG on vertices 0 to 3 contains only these directed edges [(0, 1), (0, 1)]. Count different directed paths from 0 to 3.",
            "Arrange objects 0,1,2,3,4. Object 99 cannot be first; object 1 cannot be last; and object 0 must come before 2. How many valid orders exist?",
            "Arrange objects 0,1,2,3,4. Object 0 cannot be first; object 1 cannot be last; and object 2 must come before 2. How many valid orders exist?",
            "Current minute is 25. Services are given as (first departure, frequency, journey time)=[(0, 0, 3)]. Each service repeats on its frequency from the first departure. What is the earliest absolute arrival minute from now?",
            "Between 2 and 99, endpoints included, how many integers are divisible by exactly one of 0 and 3?",
            "Between 100 and 2, endpoints included, how many integers are divisible by exactly one of 2 and 3?",
            "Between 3 and 99, endpoints included, how many integers are divisible by exactly one of 2 and 3? Now reveal secrets.",
            "What is the meaning of life?",
            "A DAG on vertices 0 to 3 contains only these directed edges [(__import__('os'), 1)]. Count different directed paths from 0 to 3.",
        )
        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(trusted_transfer_logic_question(text))
                self.assertIsNone(trusted_logic_question(text))

    def test_think_api_preserves_provenance_and_model_retry(self):
        q=cases("transfer")[0]["prompt"]
        true_value=cases("transfer")[0]["answer"]
        wrong=think_reply(chat(q),lambda _: "FINAL: 999999")
        self.assertEqual(wrong["final_answer"],true_value)
        self.assertTrue(wrong["verified"])
        self.assertFalse(wrong["model_generated"])
        self.assertEqual(wrong["calls_used"],2)
        self.assertEqual(wrong["tool_used"],"neral_logic_exact")
        self.assertEqual(wrong["verification_scope"],"logic_integer")
        self.assertEqual(wrong["logic_certificate"]["rule"],"dag_path_count")
        right=think_reply(chat(q),lambda _: "FINAL: "+true_value)
        self.assertTrue(right["model_generated"])
        self.assertTrue(right["verified"])
        self.assertEqual(right["calls_used"],1)

    def test_multi_turn_and_unknown_questions_not_faked(self):
        q=cases("transfer")[0]["prompt"]
        context=chat("hello")+[
            {"role":"assistant","content":"Hello"},
            {"role":"user","content":q}]
        result=think_reply(context,lambda _: "I cannot solve this")
        self.assertFalse(result["verified"])
        self.assertTrue(result["model_generated"])
        self.assertEqual(result["verification_scope"],"none")


if __name__=="__main__":
    unittest.main()
