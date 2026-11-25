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
