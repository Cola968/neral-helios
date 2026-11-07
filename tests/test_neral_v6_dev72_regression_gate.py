"""CPU-only regression contracts: 72-case dev signal is never release proof."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v6_dev72_regression_gate import analyze_dev72, analyze_file


def report():
    return {
        "schema":1,"status":"NERAL_V6_NEW_STRUCTURAL_DEV_72_THREE_WAY",
        "total":72,"task_families":12,
        "suite_sha256":"4b9a302a4cdb07b1e9ed3601cc1382a200ef4e4048a3afac29e72a4324f902dd",
        "v5_adapter_sha256":"2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5",
        "v6_adapter_sha256":"8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd",
        "release_validated":False,"weights_promoted":False,
        "independent_benchmark_proven":False,
        "scores":{
          "qwen_fewshot":{"correct":29,"total":72,"incomplete":0,"ungraded_completed":0,
                          "per_domain":{"code_trace":5,"math":9,"reasoning":15}},
          "neral_v5":{"correct":24,"total":72,"incomplete":0,"ungraded_completed":7,
                      "per_domain":{"code_trace":3,"math":7,"reasoning":14}},
          "neral_v6":{"correct":21,"total":72,"incomplete":0,"ungraded_completed":19,
                      "per_domain":{"code_trace":3,"math":4,"reasoning":14}},
        },
        "paired":{
          "v5_vs_qwen":{"improvements":3,"regressions":8},
          "v6_vs_qwen":{"improvements":2,"regressions":10},
          "v6_vs_v5":{"improvements":0,"regressions":3},
        }
    }


class Dev72RegressionTests(unittest.TestCase):
    def test_real_values_raise_promotion_veto(self):
        result=analyze_dev72(report())
        self.assertEqual(result["v6_minus_v5_pp"],-4.167)
        self.assertEqual(result["v6_minus_qwen_pp"],-11.111)
        self.assertEqual(result["v6_math_minus_v5"],-3)
        self.assertEqual(result["v6_ungraded_minus_v5"],12)
        self.assertFalse(result["developer_no_regression_gate_passed"])
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["can_promote_adapter"])
        self.assertEqual(result["status"],"NERAL_V6_DEV_REGRESSION_WARNING")

    def test_no_good_accuracy_guarantees_release(self):
        data=report()
        for name in data["scores"]:
            data["scores"][name]["correct"]=60
            data["scores"][name]["per_domain"]={"code_trace":20,"math":20,"reasoning":20}
            data["scores"][name]["ungraded_completed"]=0
        data["paired"]={
          "v5_vs_qwen":{"improvements":0,"regressions":0},
          "v6_vs_qwen":{"improvements":0,"regressions":0},
          "v6_vs_v5":{"improvements":0,"regressions":0}
        }
        result=analyze_dev72(data)
        self.assertTrue(result["developer_no_regression_gate_passed"])
        self.assertFalse(result["can_promote_adapter"])

    def test_invalid_score_partition_rejected(self):
        data=report()
        data["scores"]["neral_v6"]["ungraded_completed"]=100
        with self.assertRaises(ValueError):
            analyze_dev72(data)

    def test_tampered_per_domain_counts_rejected(self):
        data=report()
        data["scores"]["neral_v6"]["per_domain"]["math"]=99
        with self.assertRaises(ValueError):
            analyze_dev72(data)

    def test_mismatched_pair_counts_rejected(self):
        data=report()
        data["paired"]["v6_vs_v5"]["regressions"]=1
        with self.assertRaises(ValueError):
            analyze_dev72(data)

    def test_unexpected_adapter_hash_rejected(self):
        data=report()
        data["v6_adapter_sha256"]="b"*64
        with self.assertRaises(ValueError):
            analyze_dev72(data)

    def test_false_release_flags_required(self):
        for field in ("release_validated","weights_promoted","independent_benchmark_proven"):
            data=report()
            data[field]=True
            with self.assertRaises(ValueError):
                analyze_dev72(data)

    def test_missing_variant_rejected(self):
        data=report()
        del data["scores"]["qwen_fewshot"]
        with self.assertRaises(ValueError):
            analyze_dev72(data)

    def test_read_only_report_export_does_not_contain_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/"summary.json"
            output=Path(folder)/"decision.json"
            original.write_text(json.dumps(report()),encoding="utf8")
            decision=analyze_file(original,output)
            self.assertTrue(output.exists())
            self.assertFalse(decision["can_promote_adapter"])
            self.assertNotIn("prompt",output.read_text("utf8"))
            with self.assertRaises(ValueError):
                analyze_file(original,output)

if __name__=="__main__":
    unittest.main()
