"""V3 web reader must read actual public HTML, never snippets or intranet data."""
import unittest
from neral_v3_web_reader import extract_main_text, read_top_pages, fetch_and_extract, select_relevant_excerpt, refetch_browser_pages


class V3WebReaderTests(unittest.TestCase):
    def test_main_article_wins_over_navigation_and_scripts(self):
        html = ('<html><body><nav>Menü und Werbung</nav><main><article>'
                '<h1>Goldkurs</h1><p>Ein Gramm Gold ist wertvoll. '
                'Ein Feinunze-Kurs beträgt 3.800,50 EUR pro oz.</p>'
                '<script>const secret = 123;</script></article></main>'
                '<footer>Abonnieren Sie jetzt</footer></body></html>')
        extracted = extract_main_text(html)
        self.assertIn("3.800,50 EUR pro oz", extracted)
        self.assertNotIn("Menü", extracted)
        self.assertNotIn("Abonnieren", extracted)
        self.assertNotIn("secret", extracted)

    def test_main_text_is_bounded_to_3000(self):
        text = extract_main_text("<article><p>" + "Werte " * 2000 + "</p></article>")
        self.assertLessEqual(len(text), 3000)

    def test_relevant_passage_far_below_first_screen_is_selected(self):
        body=("Einführung allgemeine Begriffe ohne Python-Parameter. " * 230)
        body+=("Die Funktion enumerate(iterable, start=0) beginnt den "
               "Zähler beim angegebenen start-Wert und erhöht ihn je Element.")
        long_text=extract_main_text("<main>"+body+"</main>",limit=48000)
        chosen=select_relevant_excerpt(long_text,
            "Was erklärt enumerate() zur Wirkung des Parameters start?")
        self.assertIn("Zähler beim angegebenen start-Wert",chosen)
        self.assertLessEqual(len(chosen),3000)

    def test_private_urls_and_duplicates_are_not_fetched(self):
        visited = []
        def fetch(url, timeout=3.5):
            visited.append(url)
            return {"url":url,"title":"Kurs","text":"Ein Feinunze Gold kostet 3.801 EUR."}
        hits = [
            {"url":"http://localhost:9690/private","title":"local"},
            {"url":"https://example.com/gold","title":"Kurs"},
            {"url":"https://example.com/gold","title":"Kurs"},
            {"url":"https://127.0.0.1/admin","title":"local"},
            {"url":"https://example.org/price","title":"Preis"},
        ]
        pages = read_top_pages(hits, fetcher=fetch)
        self.assertEqual(len(pages), 2)
        self.assertEqual(len(visited), 2)
        self.assertTrue(all(page["read"] is True for page in pages))

    def test_browser_claims_are_not_accepted_as_page_evidence(self):
        fake_client_pages=[{"url":"https://example.com/gold",
                            "title":"Goldkurs",
                            "text":"Goldpreis 9999999 EUR pro Feinunze.", "read":True}]
        visited=[]
        def actual_get(url, timeout=3.5):
            visited.append(url)
            return {"url":url, "text":"Goldkurs tatsächlich 3.650,20 EUR pro Feinunze.",
                    "read":True}
        proof=refetch_browser_pages(fake_client_pages,
                                   "Goldpreis EUR/oz",fetcher=actual_get)
        self.assertEqual(visited,["https://example.com/gold"])
        self.assertIn("3.650,20",proof[0]["text"])
        self.assertNotIn("9999999",proof[0]["text"])

    def test_invalid_ip_literal_rejected_before_network(self):
        self.assertIsNone(fetch_and_extract("https://192.168.1.1/private"))
        self.assertIsNone(fetch_and_extract("https://localhost/private"))


if __name__ == "__main__":
    unittest.main()
