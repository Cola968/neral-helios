"""Offline tests of the live smoke-test acceptance gate."""
import unittest
from unittest.mock import Mock

from scripts.neral_v43_live_probe import evaluate_case


class LiveProbeTests(unittest.TestCase):
    def evaluate(self, status, sources):
        route = Mock(side_effect=lambda question, *, benchmark=False:
                     None if benchmark else question)
        return evaluate_case("test", "public question", route,
                             Mock(return_value={"status": status, "sources": sources}))

    def test_requires_success_and_at_least_one_read_page(self):
        for status, sources in (
            ("empty", []),
            ("headlines_only", [{"page_read": False}]),
            ("ok", []),
            ("ok", [{"page_read": False}]),
            ("timeout", [{"page_read": True}]),
        ):
            with self.subTest(status=status, sources=sources):
                self.assertFalse(self.evaluate(status, sources)["passed"])
        self.assertTrue(self.evaluate("ok", [{"page_read": True}])["passed"])

    def test_probe_exports_only_recognized_nonnegative_integer_stage_counts(self):
        route = Mock(side_effect=lambda question, *, benchmark=False:
                     None if benchmark else question)
        lookup = Mock(return_value={"status": "headlines_only", "sources": [],
                                    "stage_counts": {"reader_candidates": 2,
                                                     "reader_read_bodies": False,
                                                     "rejected_relevance": -1,
                                                     "private-provider": "private-url-sentinel"}})
        row = evaluate_case("test", "public question", route, lookup)
        self.assertEqual(row["stage_counts"], {"reader_candidates": 2})
        self.assertFalse(row["passed"])

    def test_failure_stage_reports_last_observed_boundary_not_a_cause(self):
        route = lambda question, *, benchmark=False: None if benchmark else question
        for status, counts, expected in (
            ("headlines_only", {}, "stage_not_measured"),
            ("headlines_only", {"search_results": 0}, "no_article_candidates"),
            ("empty", {"search_results": 4, "topic_candidates": 0}, "no_topic_candidates"),
            ("empty", {"search_results": 4, "topic_candidates": 1, "eligible_candidates": 0},
             "no_eligible_candidates"),
            ("headlines_only", {"eligible_candidates": 1, "reader_candidates": 1,
                                "reader_read_bodies": 0, "discovery_read_bodies": 0},
             "reader_no_usable_body"),
            ("empty", {"reader_read_bodies": 1, "rejected_relevance": 1}, "body_not_accepted"),
            ("empty", {"discovery_read_bodies": 1}, "body_not_accepted"),
            ("unavailable", {"search_results": 0}, "lookup_unavailable"),
            ("timeout", {"reader_candidates": 1}, "lookup_timeout"),
        ):
            with self.subTest(status=status, expected=expected):
                result = {"status": status, "sources": [],
                          "stage_counts": {**counts, "payload": "private-sentinel"}}
                row = evaluate_case("test", "public question", route, lambda _: result)
                self.assertEqual(row["failure_stage"], expected)
                self.assertNotIn("private-sentinel", str(row))
                self.assertFalse(row["passed"])
        self.assertIsNone(self.evaluate("ok", [{"page_read": True}]).get("failure_stage"))

    def test_never_searches_if_offline_isolation_fails(self):
        lookup = Mock()
        result = evaluate_case("test", "public question", Mock(return_value="q"), lookup)
        self.assertEqual(result["status"], "isolation_failed")
        self.assertFalse(result["passed"])
        lookup.assert_not_called()

    def test_extraction_metric_requires_explicit_untruncated_read(self):
        rows = [{"page_read": True, "article_chars": 400},
                {"page_read": True, "article_chars": 400, "article_truncated": True},
                {"page_read": True, "article_chars": 400, "article_truncated": False}]
        self.assertEqual(self.evaluate("ok", rows)["untruncated_extractions"], 1)


if __name__ == "__main__":
    unittest.main()
