"""Regression gates for the native Arctus 350M, with no GPU/model calls."""
from __future__ import annotations

import math
import unittest

from frontier350.score_chat350_capabilities import score, score_one, compare
from frontier350.release_check350 import check


def cases():
    return [
        {"id":"case-1","category":"de","scoring":{"type":"exact","value":"Ja"}},
        {"id":"case-2","category":"math","scoring":{"type":"integer","value":42}},
        {"id":"case-3","category":"code","scoring":{"type":"contains_all","values":["def","return"]}},
        {"id":"case-4","category":"de","scoring":{"type":"contains_any","values":["Berlin","Hamburg"]}},
    ]


def result(pass_de: int = 2, pass_math: int = 1, pass_code: int = 1):
    cats={"de":pass_de/2*100,"math":pass_math*100,"code":pass_code*100}
    return {"overall_pct":(pass_de+pass_math+pass_code)/4*100,
            "by_category":{k:{"cases":2 if k=="de" else 1,
                               "passed":pass_de if k=="de" else pass_math if k=="math" else pass_code,
                               "pct":v} for k,v in cats.items()}}


class ArctusScoreIntegrityTests(unittest.TestCase):
    def test_duplicate_report_id_can_never_overwrite_failed_prediction(self):
        report={"rows":[
            {"id":"case-1","text":"Nein"},
            {"id":"case-1","text":"Ja"},
            {"id":"case-2","text":"42"},
            {"id":"case-3","text":"def f(): return 42"},
            {"id":"case-4","text":"Berlin"},
        ]}
        with self.assertRaises(ValueError):
            score(report,cases())

    def test_duplicate_reference_case_ids_rejected(self):
        duplicated=cases()+[cases()[0]]
        with self.assertRaises(ValueError):
            score({"rows":[{"id":c["id"],"text":"Ja"} for c in cases()]},duplicated)

    def test_missing_nontext_or_empty_report_rejected(self):
        with self.assertRaises(ValueError):
            score({"rows":[]},[])
        for invalid in (
            {"rows":[{"id":"case-1","text":None},{"id":"case-2","text":"42"},{"id":"case-3","text":"foo"},{"id":"case-4","text":"Berlin"}]},
            {"rows":[{"id":"case-1","text":True},{"id":"case-2","text":"42"},{"id":"case-3","text":"foo"},{"id":"case-4","text":"Berlin"}]},
            {"rows":[{"id":"case-1"},{"id":"case-2","text":"42"},{"id":"case-3","text":"foo"},{"id":"case-4","text":"Berlin"}]},
        ):
            with self.subTest(data=invalid["rows"][0]),self.assertRaises(ValueError):
                score(invalid,cases())

    def test_no_empty_vacuous_contains_scoring(self):
        for spec in (
            {"type":"contains_all","values":[]},
            {"type":"contains_all","values":"abc"},
            {"type":"contains_any","values":[]},
            {"type":"contains_any","values":["x"],"and_values":[]},
        ):
            with self.subTest(spec=spec),self.assertRaises(ValueError):
                score_one("arbitrary",spec)

    def test_any_category_regression_blocks_checkpoint_promotion(self):
        # A 10-point decline previously slipped through a '-20pp hard gate'.
        base={
            "overall_pct":70.,
            "by_category":{
                "math":{"cases":20,"passed":14,"pct":70.0},
                "de":{"cases":10,"passed":7,"pct":70.0},
            },
        }
        cand={
            "overall_pct":70.,
            "by_category":{
                "math":{"cases":20,"passed":12,"pct":60.0},
                "de":{"cases":10,"passed":9,"pct":90.0},
            },
        }
        assessment=compare(base,cand)
        self.assertFalse(assessment["category_no_regression"])
        self.assertFalse(assessment["capability_no_regression"])
        self.assertEqual(assessment["category_delta_pp"]["math"],-10.0)

    def test_scores_require_same_categories_and_finite_values(self):
        base=result()
        wrong={"overall_pct":75,"by_category":{"other":{"cases":4,"passed":3,"pct":75.}}}
        with self.assertRaises(ValueError):
            compare(base,wrong)
        for bad in (float("nan"),float("inf"),-1.,101.,True,"100"):
            damaged={**base,"overall_pct":bad}
            with self.subTest(bad=repr(bad)),self.assertRaises(ValueError):
                compare(base,damaged)

    def test_counts_must_match_reported_accuracy(self):
        base=result(pass_de=1)
        fake={"overall_pct":100.0, "by_category":base["by_category"]}
        with self.assertRaises(ValueError):
            compare(base,fake)


class ArctusReleaseFailClosedTests(unittest.TestCase):
    def valid_evidence(self):
        return (
            {"gates":{"pass":True}},
            {"decision":"manual_review_required"},
            {"comparison":{
                "capability_no_regression":True,
                "category_no_regression":True,
            },"cases":30},
            {"slice_no_regression":True},
        )

    def test_valid_completed_gates_still_wait_for_manual_review(self):
        r=check(*self.valid_evidence())
        self.assertTrue(r["machine_gates_pass"])
        self.assertFalse(r["automatic_release"])
        self.assertEqual(r["decision"],"manual_release_review_required")

    def test_truthy_strings_and_numbers_cannot_be_treated_as_gates(self):
        from copy import deepcopy
        for idx,field in ((0,"pass"),(2,"capability_no_regression"),(2,"category_no_regression"),(3,"slice_no_regression")):
            for bad in ("false",1,[],{"approved":True}):
                args=list(map(deepcopy,self.valid_evidence()))
                if idx==0:args[idx]["gates"][field]=bad
                elif idx==2:args[idx]["comparison"][field]=bad
                else:args[idx][field]=bad
                with self.subTest(gate=field,bad=bad):
                    self.assertFalse(check(*args)["machine_gates_pass"])

    def test_missing_category_nonregression_or_case_count_blocks_release(self):
        from copy import deepcopy
        for value in (None,False):
            args=list(map(deepcopy,self.valid_evidence()))
            if value is None: del args[2]["comparison"]["category_no_regression"]
            else:args[2]["comparison"]["category_no_regression"]=value
            self.assertEqual(check(*args)["decision"],"reject")
        args=list(map(deepcopy,self.valid_evidence()))
        args[2]["cases"]=29
        self.assertFalse(check(*args)["machine_gates_pass"])

    def test_optional_external_report_cannot_bypass_gate_via_string(self):
        args=self.valid_evidence()
        self.assertFalse(check(*args, external_decision={"pass":"false"})["machine_gates_pass"])
        self.assertFalse(check(*args, require_external=True)["machine_gates_pass"])


if __name__ == "__main__":
    unittest.main()
