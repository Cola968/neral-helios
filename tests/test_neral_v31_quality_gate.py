"""NERAL V3.1 prospective quality gate: no fabricated success or partial results."""
import json
import unittest

from scripts.neral_v31_quality_gate import (
    build_cases, score_case, summarize, validate_case_manifest,
)

class V31ProspectiveCases(unittest.TestCase):
    def test_exactly_100_unique_with_20_per_category_and_frozen_oracles(self):
        cases=build_cases()
        self.assertEqual(len(cases),100)
        self.assertEqual(cases,build_cases())
        self.assertEqual(len({x["id"] for x in cases}),100)
        self.assertEqual(len({x["question_sha256"] for x in cases}),100)
        self.assertEqual({cat:sum(x["category"]==cat for x in cases)
                          for cat in ("math","logic","programming","web","knowledge")},
                         {cat:20 for cat in ("math","logic","programming","web","knowledge")})
        self.assertEqual(validate_case_manifest(cases),100)

    def test_math_and_programming_need_exact_integer_not_plausible_prose(self):
        c=next(x for x in build_cases() if x["category"]=="math")
        ok={"text":str(c["gold"]),"seconds":0.3,"agent_checked":True}
        self.assertTrue(score_case(c,ok,200)["passed"])
        for text in ("The answer might be "+str(c["gold"]),
                     "Wrong 2; correct "+str(c["gold"]),
                     "42"+str(c["gold"]), ""):
            self.assertFalse(score_case(c,{**ok,"text":text},200)["passed"])

    def test_web_requires_real_read_pages_and_each_literal_named_quote(self):
        c=next(x for x in build_cases() if x["category"]=="web")
        left,right=c["entities"]
        ok={"agent_checked":True,"tool_used":"direct_reasoning",
            "text":f"{left}: belegt. {right}: belegt.\n\n[web1] „{left} offers useful tools.“\n\n[web2] „{right} offers useful tools.“",
            "sources":[{"page_read":True,"url":"https://a.example.org/"},
                       {"page_read":True,"url":"https://b.example.org/"}],
            "seconds":1.25}
        self.assertTrue(score_case(c,ok,200)["passed"])
        for patch in (
            {"agent_checked":False},
            {"tool_used":None},
            {"sources":[{"page_read":False,"url":"https://a.example.org/"},{"page_read":False,"url":"https://b.example.org/"}]},
            {"text":f"{left} vs {right}\n[web1] „{left} offers useful tools.“"},
            {"text":ok["text"].replace(f"[web2]", "[web99]")},
            {"text":ok["text"].replace(f"[web2] „{right}",f"[web2] „Unknown")},
            {"text":ok["text"]+"\n99% guaranteed."},
        ):
            self.assertFalse(score_case(c,{**ok,**patch},200)["passed"])

    def test_web_failure_is_not_mistaken_for_verified_fact(self):
        c=next(x for x in build_cases() if x["category"]=="web")
        bad={"agent_checked":True,"tool_used":"direct_reasoning",
             "text":"Confident answer with no actual cited pages.",
             "sources":[],"seconds":1.0}
        row=score_case(c,bad,200)
        self.assertFalse(row["passed"])
        self.assertTrue(row["critical_unverified"])

    def test_knowledge_demands_exact_expected_term(self):
        c=next(x for x in build_cases() if x["category"]=="knowledge")
        answer={"text":c["gold"],"seconds":0.1}
        self.assertTrue(score_case(c,answer,200)["passed"])
        self.assertFalse(score_case(c,{**answer,"text":c["gold"]+" vielleicht"},200)["passed"])

    def test_failed_http_zero_latency_and_timeouts_cannot_pass(self):
        c=build_cases()[0]
        for data,status in (({"text":str(c["gold"]),"seconds":0.2},503),
                            ({"text":str(c["gold"]),"seconds":-1},200),
                            ({"text":str(c["gold"]),"seconds":None},200)):
            self.assertFalse(score_case(c,data,status)["passed"])

    def test_aggregate_rejects_incomplete_cases_and_false_100(self):
        cases=build_cases()
        with self.assertRaises(ValueError):
            summarize(cases,[])
        rows=[score_case(c,{"text":str(c.get("gold","")),"seconds":0.1},200) for c in cases]
        result=summarize(cases,rows)
        self.assertEqual(result["total"],100)
        self.assertEqual(result["categories"]["web"]["n"],20)
        self.assertFalse(result["threshold_90_of_100_met"])
        self.assertFalse(result["production_promotion_allowed"])
        self.assertIn("median",result["latency_seconds"])
        self.assertIn("p95",result["latency_seconds"])

    def test_manifest_rejects_duplicates_and_altered_gold(self):
        cases=build_cases()
        with self.assertRaises(ValueError):
            validate_case_manifest(cases[:-1]+[cases[0]])
        altered=[dict(x) for x in cases]
        altered[0]["gold"]=99_999_999
        with self.assertRaises(ValueError):
            validate_case_manifest(altered)

if __name__=="__main__":
    unittest.main()
