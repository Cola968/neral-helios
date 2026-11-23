"""Final 3+3 source-proof and owner GPU accounting contracts (CPU only)."""
from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from scripts.neral_v42_telemetry_probe import grade_live_probe
from scripts.neral_v42_gpu_memory import (
    parse_compute_rows, read_vram_snapshot, validate_owner_snapshot,
)
from neral_evidence_agent import run_evidence_agent


PROMPT = (
    "In welchem Jahr landete Apollo 11 erstmals Menschen auf dem Mond? "
    "Nenne die Jahreszahl und belege sie mit einem wörtlichen Zitat "
    "aus einer tatsächlich gelesenen Quelle."
)
URL = "https://de.wikipedia.org/wiki/Apollo_11"
QUOTE = "Apollo 11 landete am 20. Juli 1969 auf dem Mond."


class YearSourceFinalCPU(unittest.TestCase):
    def test_live_success_propagates_year_and_citation_proof(self):
        page = {"url":URL, "page_read":True, "page_text":QUOTE,
                "year_clip_code":"year_preserved_in_extract"}
        def plan(*_):
            return json.dumps({"tool":"direct_reasoning", "parts":["Jahr"], "search_query":""})
        class Search:
            def search(self, _):
                return {"results":[{"url":URL,"title":"Apollo 11"}]}
        answer = run_evidence_agent(PROMPT, plan, Mock(),
            web_query="Apollo 11 erste Mondlandung NASA Jahr", search=Search(),
            page_reader=lambda *_:[page], max_attempts=1)
        self.assertTrue(answer.get("source_year_match"), answer)
        self.assertEqual(answer.get("year_source_pipeline_code"), "candidate_available")
        self.assertIs(answer.get("year_quote_in_final"), True)
        self.assertEqual(grade_live_probe("knowledge", 200, answer),
                         "exact_year_and_read_https_quote")

    def test_wrong_or_unverified_year_is_not_falsely_graded(self):
        correct = {
            "text": "1969\n\n[web1] „" + QUOTE + "“",
            "source_year_match": True,
            "year_source_pipeline_code": "candidate_available",
            "year_quote_in_final": True,
            "sources": [{"url": URL, "page_read": True}],
        }
        self.assertEqual(grade_live_probe("knowledge", 200, correct),
                         "exact_year_and_read_https_quote")
        for patch in (
            {"text": correct["text"].replace("1969\n\n", "1961\n\n")},
            {"year_quote_in_final": False},
            {"source_year_match": False},
            {"year_source_pipeline_code": "year_event_not_contiguous"},
            {"sources": [{"url":"http://de.wikipedia.org/wiki/Apollo_11","page_read":True}]},
            {"sources": [{"url":URL,"page_read":False}]},
            {"text": "1969"},
        ):
            with self.subTest(patch=patch):
                self.assertNotEqual(grade_live_probe("knowledge", 200, {**correct, **patch}),
                                    "exact_year_and_read_https_quote")

    def test_everyday_needs_actual_physical_explanation_and_math_exactness(self):
        good = {"text": "Der Wasserdampf aus der Dusche trifft auf den kalten Spiegel. "
                        "Dort kondensiert er zu kleinen Wassertröpfchen.", "error_code":None}
        self.assertEqual(grade_live_probe("everyday",200,good),
                         "two_sentence_condensation_explanation")
        self.assertNotEqual(grade_live_probe("everyday",200,{"text":"Ja."}),
                            "two_sentence_condensation_explanation")
        self.assertEqual(grade_live_probe("math_logic",200,
            {"text":"17 * 23 = 391\n391 - 46 = 345\nENDERGEBNIS: 345"}),
            "exact_numeric_match")
        self.assertNotEqual(grade_live_probe("math_logic",200,
            {"text":"ENDERGEBNIS: 344"}), "exact_numeric_match")


class VramIsolationCPU(unittest.TestCase):
    def test_compute_rows_parse_numeric_and_wddm_unknown_fail_closed(self):
        self.assertEqual(parse_compute_rows("111, 4100 MiB\n112, 1500 MiB\n345, N/A\n"),
                         {111:4100, 112:1500, 345:None})
        with self.assertRaises(ValueError):
            parse_compute_rows("not a pid, 100 MiB")

    def test_wddm_bracketed_na_is_not_parse_error(self):
        self.assertEqual(parse_compute_rows("111, [N/A]\n333, 1500 MiB\n"), 
                         {111: None, 333: 1500})

    def test_wddm_unknown_uses_measured_cuda_telemetry_as_fallback(self):
        snapshot={"neral_process_mib":None,
                  "process_memory_status":"wddm_memory_unreported",
                  "system_total_mib":11500}
        self.assertFalse(validate_owner_snapshot(snapshot, process_cap_mib=9500))
        self.assertTrue(validate_owner_snapshot(
            snapshot, process_cap_mib=9500, cuda_reserved_mib=7800))
        self.assertFalse(validate_owner_snapshot(
            snapshot, process_cap_mib=9500, cuda_reserved_mib=9800))
        self.assertFalse(validate_owner_snapshot(
            {"neral_process_mib":None,
             "process_memory_status":"nvidia_compute_query_unavailable"},
            process_cap_mib=9500,cuda_reserved_mib=7600))

    def test_process_and_system_usage_are_separate(self):
        calls=[]
        def fake_run(args, **kwargs):
            calls.append(args)
            if "--query-gpu=memory.used" in args:
                return Mock(returncode=0, stdout="11870\n")
            return Mock(returncode=0, stdout="111, 4200 MiB\n112, 3100 MiB\n345, 1500 MiB\n")
        snapshot=read_vram_snapshot((111,112), runner=fake_run)
        self.assertEqual(snapshot["neral_process_mib"], 7300)
        self.assertEqual(snapshot["system_total_mib"], 11870)
        self.assertEqual(snapshot["other_compute_process_count"], 1)
        self.assertEqual(len(calls),2)

    def test_unreported_process_memory_is_unknown_not_zero(self):
        def fake_run(args, **kwargs):
            return Mock(returncode=0, stdout="9000\n" if "--query-gpu=memory.used" in args
                        else "111, N/A\n")
        snapshot=read_vram_snapshot((111,), runner=fake_run)
        self.assertIsNone(snapshot["neral_process_mib"])
        self.assertFalse(validate_owner_snapshot(snapshot, process_cap_mib=9500))

    def test_over_process_cap_fails_even_with_lower_total_limit(self):
        snapshot={"neral_process_mib":9700,"system_total_mib":12000}
        self.assertFalse(validate_owner_snapshot(snapshot, process_cap_mib=9500))
        self.assertTrue(validate_owner_snapshot({"neral_process_mib":7800,
                                                  "system_total_mib":12000},
                                                 process_cap_mib=9500))


if __name__ == "__main__":
    unittest.main()
