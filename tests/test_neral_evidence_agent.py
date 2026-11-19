import json
import unittest
from unittest.mock import Mock, patch

from neral_evidence_agent import run_evidence_agent
from neral_quantity_plans import quantity_plan
from neral_page_reader import MainText, _public_address, read_top_pages


class EvidenceAgentTest(unittest.TestCase):
    def setUp(self):
        self.question = "250 Gramm zu 31,1035 Gramm je Feinunze. Was kostet das Gold in Euro?"
        self.plan = {"tool": "python_sandbox", "parts": ["Materialwert"]}
        self.worker = {"locale": "de", "calculations": [
            {"label": "Materialwert", "expr": "u1 / u2 * w1_1", "unit": "EUR"}], "missing": []}
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
        self.worker["missing"] = ["Goldpreis EUR/Feinunze"]
        self.worker["calculations"] = []
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertEqual(result["agent_attempts"], 3)
        self.assertIn("Goldpreis EUR/Feinunze", result["text"])
        self.execute.assert_not_called()

    def test_invented_literal_never_reaches_sandbox(self):
        self.worker["calculations"][0]["expr"] = "u1 / u2 * 0"
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertFalse(result["agent_checked"])
        self.execute.assert_not_called()

    def test_web_calculation_cannot_use_only_user_numbers(self):
        self.worker["calculations"][0]["expr"] = "u1 / u2"
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertIn("externen Quelle", result["text"])
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


class ExactQuantityTest(unittest.TestCase):
    def test_rate_uses_same_metal_currency_and_quote_basis(self):
        question = "Was ist der Materialwert eines 125-Gramm-Silberbarrens in Euro (1 Feinunze = 31,1035 Gramm)?"
        evidence = {"user": question, "web1": "| GOLD | 3.360,00 €/oz\n| SILBER | 44,00 $/oz | 40,50 €/oz | 1,30 €/g"}
        plan, numbers = quantity_plan(question, evidence)
        self.assertEqual(numbers["price"]["token"], "40,50")
        self.assertEqual(plan["calculations"][1]["expr"], "mass / grams_per_ounce * price")
        self.assertIsNone(quantity_plan(question, {"user": question, "web1": "SILBER 44,00 $/oz"}))

    def test_inventory_requires_the_complete_source(self):
        question = ("Jan stellt 4 Kisten mit je 10 Flaschen in den Raum. 3 Flaschen zerbrechen. "
                    "In jedes übrige Glas füllt er 0,25 Liter Saft. Der Behälter enthält vorher 20 Liter. "
                    "Was bleibt im Behälter? Wie viele gefüllte Flaschen stehen bereit?")
        _, numbers = quantity_plan(question, {"user": question})
        self.assertEqual(numbers["lost"]["token"], "3")
        self.assertIsNone(quantity_plan(question + " Weitere 2 bleiben leer.", {"user": question}))


if __name__ == "__main__":
    unittest.main()
