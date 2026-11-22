"""CPU-only attribution regressions. Synthetic data; no GPU, network or private cases."""
import unittest
from unittest.mock import Mock

from neral_v42_quality import (stock_stdout_diagnostic_code,
                               year_source_pipeline_code)
from neral_page_reader import MainText, read_top_pages
from scripts.neral_v42_telemetry_probe import DIAGNOSTIC_CASES
from test_neral_three_tier_api import make_runtime

MATH = DIAGNOSTIC_CASES[1][1]
YEAR = DIAGNOSTIC_CASES[2][1]
URL = "https://example.org/apollo-11"


class StockFailureAttributionCPU(unittest.TestCase):
    def test_parser_reports_shape_not_sensitive_stdout(self):
        samples = {
            "Gesamt: 391\nErgebnis: 345": "accepted",
            "345": "accepted",
            "346": "wrong_value",
            "Gesamt: 391\nErgebnis: 345\nZusatz: 12": "extra_lines",
            "Gesamt: 391\nErgebnis: 346": "wrong_value",
            "Gesamt: 391\nAntwort: 345": "unsupported_shape",
            "": "empty_stdout",
        }
        for stdout, expected in samples.items():
            with self.subTest(stdout=stdout):
                self.assertEqual(
                    stock_stdout_diagnostic_code(stdout, succeeded=True, prompt=MATH),
                    expected)
        self.assertEqual(stock_stdout_diagnostic_code(
            "345", succeeded=False, prompt=MATH), "execution_failed")
        self.assertEqual(stock_stdout_diagnostic_code(
            "secret personal output", succeeded=True, prompt="Hallo"), "not_stock_task")

    def test_real_quartz_failure_exposes_only_enum_and_attempt_count(self):
        runtime, _, high, _ = make_runtime()
        calls = []
        def generate(*args, **kwargs):
            high.loaded = True
            calls.append(1)
            code = "print('Gesamt: 391')\\nprint('Antwort: 345')"
            return "```python\n" + code.replace("\\n","\n") + "\n```"
        high.respond_chat = generate
        runtime.execute_python = Mock(return_value={
            "status":"PASSED", "output":"Gesamt: 391\nAntwort: 345"})
        result = runtime.answer([{"role":"system","content":"Nur die aktuelle Aufgabe"},
                                 {"role":"user","content":MATH}], "Quartz")
        self.assertTrue(result["sandbox_executed"])
        self.assertEqual(result["math_stdout_diagnostic_code"], "unsupported_shape")
        self.assertEqual(result["sandbox_attempts"],1)
        self.assertNotIn("391",result["math_stdout_diagnostic_code"])
        self.assertEqual(len(calls), 2)


class YearEvidenceAttributionCPU(unittest.TestCase):
    def test_page_stage_separates_missing_year_and_missing_compact_citation(self):
        from neral_v42_quality import unique_read_year_source
        valid = "Apollo 11 landed the first humans on the Moon.\nThe mission took place in July 1969."
        read = lambda t: [{"url":URL,"page_text":t,"page_read":True}]
        self.assertEqual(year_source_pipeline_code(YEAR, [], {}), "no_read_page")
        self.assertEqual(year_source_pipeline_code(YEAR, read(
            "Apollo 11 landed on the Moon."), {}), "year_missing_in_extract")
        too_far = "Apollo 11 landed on the Moon.\nIt was a historic flight.\nThe date was in 1969."
        self.assertIsNone(unique_read_year_source(YEAR,read(too_far)))
        self.assertEqual(year_source_pipeline_code(
            YEAR, read(too_far), {"web1":too_far}), "year_event_not_contiguous")
        self.assertEqual(year_source_pipeline_code(
            YEAR, read(valid), {"web1":"Apollo 11 landed."}),
            "quote_missing_from_evidence")
        self.assertEqual(year_source_pipeline_code(
            YEAR, read(valid), {"web1":valid}), "candidate_available")

    def test_reader_records_year_clipped_without_storing_page(self):
        markup = ("<main>" + "<p>Navigation note %d. This is filler.</p>" * 0 +
                  "".join("<p>Landing timeline note %d, no date.</p>" % i
                          for i in range(190)) +
                  "<p>Apollo 11 returned to Earth in 1969.</p></main>")
        parser = MainText()
        parser.feed(markup)
        excerpt = parser.text("Apollo 11 first moon landing year")
        self.assertLessEqual(len(excerpt),4600)
        self.assertIn(parser.year_clip_code,
                      {"year_preserved_in_extract", "year_lost_in_clip"})
        self.assertNotIn("1969", parser.year_clip_code)

        result = read_top_pages(
            [{"url": URL,"title":"Synthetic","snippet":"No evidence"}],
            "Apollo 11 year", fetch=lambda url,q:{
                "url":url,"text":"Apollo 11 landed on the Moon in 1969.",
                "year_clip_code":"year_preserved_in_extract"})
        self.assertEqual(result[0]["year_clip_code"],"year_preserved_in_extract")
        self.assertEqual(result[0]["page_text"],"Apollo 11 landed on the Moon in 1969.")

    def test_year_clip_stage_outweighs_ambiguous_reader_output(self):
        page = {"url": URL, "page_read": True,
                "page_text":"Apollo 11 lunar mission facts.",
                "year_clip_code":"year_lost_in_clip"}
        self.assertEqual(year_source_pipeline_code(
            YEAR,[page],{"web1":page["page_text"]}),"year_lost_in_clip")


if __name__ == "__main__":
    unittest.main()
