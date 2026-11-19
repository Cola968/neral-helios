import json
import unittest
from unittest.mock import Mock, patch

from neral_evidence_agent import run_evidence_agent
from neral_page_reader import MainText, _public_address, read_top_pages


class EvidenceAgentTest(unittest.TestCase):
    def setUp(self):
        self.question = "250 Gramm zu 31,1035 Gramm je Feinunze. Was kostet das Gold in Euro?"
        self.plan = {"tool": "python_sandbox", "parts": ["Materialwert"], "inputs": [
            {"name": "grams", "need": "Barrenmasse"},
            {"name": "ounce", "need": "Gramm je Feinunze"},
            {"name": "price", "need": "Goldpreis EUR/Feinunze"},
        ]}
        self.worker = {"bindings": {
            "grams": {"source": "user", "quote": "250 Gramm", "token": "250", "locale": "de"},
            "ounce": {"source": "user", "quote": "31,1035 Gramm je Feinunze", "token": "31,1035", "locale": "de"},
            "price": {"source": "web1", "quote": "Gold: 3.367,35 EUR pro Feinunze", "token": "3.367,35", "locale": "de"},
        }, "calculations": [{"label": "Materialwert", "expr": "grams / ounce * price", "unit": "EUR"}], "missing": []}
        self.pages = Mock(return_value=[{"url": "https://gold.example.com/price", "page_read": True,
                                        "page_text": "Gold: 3.367,35 EUR pro Feinunze"}])
        self.search = Mock()
        self.search.search.return_value = {"results": [{"url": "https://gold.example.com/price"}]}
        self.execute = Mock(return_value={"status": "PASSED", "output": "27065.6839262456"})

    def run_agent(self, packets):
        model = Mock(side_effect=[json.dumps(packet) for packet in packets])
        return run_evidence_agent(self.question, model, self.execute, web_query="Goldpreis",
                                  search=self.search, page_reader=self.pages)

    def test_read_then_compute_with_exact_quote(self):
        result = self.run_agent([self.plan, self.worker, {"ok": True, "issue": ""}])
        self.assertTrue(result["agent_checked"])
        self.assertFalse(result["verified"])
        self.assertIn("27065,68 EUR", result["text"])
        self.assertIn("Decimal('3367.35')", self.execute.call_args.args[0])
        self.search.search.assert_called_once_with("Goldpreis")

    def test_missing_price_never_reaches_sandbox(self):
        del self.worker["bindings"]["price"]
        self.worker["missing"] = ["price"]
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertEqual(result["agent_attempts"], 3)
        self.assertIn("Goldpreis EUR/Feinunze", result["text"])
        self.execute.assert_not_called()

    def test_invented_literal_never_reaches_sandbox(self):
        self.worker["calculations"][0]["expr"] = "grams / ounce * 0"
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertFalse(result["agent_checked"])
        self.execute.assert_not_called()

    def test_checker_rejects_and_repairs(self):
        result = self.run_agent([self.plan, self.worker, {"ok": False, "issue": "Einheit erklären"},
                                 self.plan, self.worker, {"ok": True, "issue": ""}])
        self.assertTrue(result["agent_checked"])
        self.assertEqual(result["agent_attempts"], 2)
        self.search.search.assert_called_once()


class PageReaderTest(unittest.TestCase):
    def test_extracts_text_without_navigation_or_scripts(self):
        p = MainText()
        p.feed('<body><nav>Menü</nav><main><script>code()</script>'
               '<table><tr><td>1 Feinunze</td><td>3.367,35 EUR</td></tr></table>'
               '<p hidden>versteckt</p><p>Messung</p></main><footer>Links</footer>')
        self.assertEqual(p.text("Gold EUR"), "| 1 Feinunze | 3.367,35 EUR\nMessung")

    def test_uses_only_three_public_unique_results(self):
        urls = ["https://127.0.0.1/", "https://a.example.com/", "https://a.example.com/",
                "https://b.example.com/", "https://c.example.com/", "https://d.example.com/"]
        fetch = Mock(side_effect=lambda url, _: {"url": url, "text": "x" * 3100})
        pages = read_top_pages([{"url": url} for url in urls], "topic", fetch=fetch)
        self.assertEqual([page["url"] for page in pages], urls[1:2] + urls[3:5])
        self.assertTrue(all(len(page["page_text"]) == 3000 for page in pages))

    def test_rejects_nonpublic_dns_answers(self):
        addresses = [(0, 0, 0, "", ("93.184.215.14", 443)), (0, 0, 0, "", ("127.0.0.1", 443))]
        with patch("neral_page_reader.socket.getaddrinfo", return_value=addresses):
            with self.assertRaises(ValueError):
                _public_address("mixed.example.com")


if __name__ == "__main__":
    unittest.main()
