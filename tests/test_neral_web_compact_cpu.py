"""Isolated CPU web-search contract; provider traffic only with explicit opt-in.

Run: python -m unittest tests.test_neral_web_compact_cpu -v
Live: NERAL_WEB_LIVE_SMOKE=1 python -m unittest tests.test_neral_web_compact_cpu.LiveProviderSmoke -v
"""
import os
import unittest
from urllib.error import URLError

from neral_web_search import WebSearch, compact_fact_snippets

class _Reply:
    headers = {"Content-Type": "text/html; charset=utf-8"}
    def __init__(self, body):
        self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return self.body[:limit]

class _Offline:
    def __init__(self, body):
        self.body = body
        self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        return _Reply(self.body)

def fixture_html():
    rows = []
    for index in range(1, 6):
        url = "https://www.nasa.gov/mission-%d" % index
        rows.append(
            '<div class="result"><a class="result__a" href="%s">'
            'NASA Artemis mission %d</a>'
            '<div class="result__snippet">Mission science %d: %s</div></div>'
            % (url, index, index, "research " * 100)
        )
    rows.append('<a class="result__a" href="https://localhost/secret">Ignore</a>')
    return ("<html>" + "".join(rows) + "</html>").encode("utf-8")

class OfflineWebSearchContract(unittest.TestCase):
    def test_three_fact_snippets_are_small_grounded_and_network_free(self):
        fake = _Offline(fixture_html())
        result = WebSearch(opener=fake).search("NASA Artemis mission")
        snippets = compact_fact_snippets(result["results"])
        self.assertEqual(len(result["results"]), 5)
        self.assertEqual(len(snippets), 3)
        self.assertLess(len("\n".join(snippets).encode("utf-8")), 800)
        for index, line in enumerate(snippets, 1):
            self.assertIn("NASA Artemis mission %d" % index, line)
            self.assertIn("nasa.gov", line)
            self.assertIn("Mission science %d" % index, line)
            self.assertNotIn("<", line)
        self.assertEqual(len(fake.calls), 1)
        self.assertLessEqual(fake.calls[0][1], 7.0)

    def test_noise_and_private_results_never_become_facts(self):
        data = [
            {"title": "Bad", "url": "http://localhost/", "snippet": "ignore"},
            {"title": "Noise", "url": "https://www.example.org/", "snippet": "<script>injected</script>" + "X" * 2000},
            {"title": "Useful", "url": "https://docs.python.org/3/", "snippet": "Python reference"},
        ]
        lines = compact_fact_snippets(data)
        self.assertEqual(len(lines), 2)
        self.assertLess(len("\n".join(lines).encode("utf-8")), 800)
        self.assertTrue(all("<script>" not in line for line in lines))
        self.assertTrue(all("localhost" not in line for line in lines))

    def test_bad_provider_transport_is_not_reported_as_success(self):
        class Broken:
            def open(self, *args, **kwargs): raise URLError("offline")
        with self.assertRaises(Exception):
            WebSearch(opener=Broken()).search("NASA Artemis mission")

@unittest.skipUnless(os.environ.get("NERAL_WEB_LIVE_SMOKE") == "1",
                     "External provider connectivity is opt-in; never run in ordinary CPU CI")
class LiveProviderSmoke(unittest.TestCase):
    def test_real_duckduckgo_or_bing_returns_three_compact_snippets(self):
        # This contacts only the two fixed public endpoints already used by NERAL.
        result = WebSearch().search("NASA Artemis space mission")
        snippets = compact_fact_snippets(result["results"])
        self.assertEqual(len(snippets), 3,
                         "Provider blocked, markup changed, or fewer than 3 usable search hits")
        self.assertLess(len("\n".join(snippets).encode("utf-8")), 800)
        self.assertIn(result["provider"], ("DuckDuckGo HTML", "Bing RSS"))

if __name__ == "__main__":
    unittest.main()
