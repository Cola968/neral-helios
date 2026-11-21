"""No network, model or GPU required."""
import unittest
from unittest.mock import Mock, patch
import neral_local_wiki as wiki

RSS = b"""<rss><channel><item><title>Elektrostatik</title><link>/content/wiki_de/Elektrostatik</link></item><item><title>Elektrostatik (Physik)</title><link>/content/wiki_de/Physik</link></item></channel></rss>"""
HTML = b"""<html><nav><p>Menu noise</p></nav><table class="infobox"><tr><th>Teilgebiet</th><td>Physik</td></tr></table><p>Elektrostatik ist ein Teilgebiet der Physik und untersucht ruhende elektrische Ladungen sowie ihre Wechselwirkungen.</p><h2>Weitere Details</h2><p>Not the introduction</p></html>"""

class OfflineWikiTests(unittest.TestCase):
    def test_ranked_leads_and_infobox(self):
        with patch.object(wiki, "_get", side_effect=[RSS, HTML, HTML]):
            hits = wiki.local_wiki_search("Was ist Elektrostatik?")
        self.assertEqual(len(hits), 2)
        self.assertIn("ruhende elektrische Ladungen", hits[0]["context"])
        self.assertIn("Teilgebiet", hits[0]["context"])
        self.assertNotIn("Menu noise", hits[0]["context"])
        self.assertNotIn("Not the introduction", hits[0]["context"])

    def test_empty_results(self):
        with patch.object(wiki, "_get", return_value=b"<rss><channel/></rss>"):
            self.assertEqual(wiki.local_wiki_search("Was ist Elektrostatik?"), [])

    def test_timeout(self):
        with patch.object(wiki, "_get", side_effect=TimeoutError):
            self.assertEqual(wiki.local_wiki_search("Was ist Elektrostatik?"), [])

    def test_unrelated_title_is_rejected(self):
        with patch.object(wiki, "_get", return_value=RSS):
            self.assertEqual(wiki.local_wiki_search("Was ist Photosynthese?"), [])

    def test_external_link_is_rejected(self):
        bad = RSS.replace(b"/content/wiki_de/", b"https://example.org/")
        with patch.object(wiki, "_get", return_value=bad) as get:
            self.assertEqual(wiki.local_wiki_search("Was ist Elektrostatik?"), [])
            self.assertEqual(get.call_count, 1)

    def test_offline_hit_avoids_web(self):
        local = Mock(return_value=[{"title": "Elektrostatik"}])
        web = Mock(return_value={"text": "web"})
        render = Mock(return_value={"text": "offline"})
        self.assertEqual(wiki.lookup_with_fallback("Was ist Elektrostatik?", web, render, local=local),
                         {"text": "offline"})
        web.assert_not_called()

    def test_empty_and_timeout_use_existing_web_callback(self):
        for local in (Mock(return_value=[]), Mock(side_effect=TimeoutError)):
            web = Mock(return_value={"text": "V4"})
            render = Mock()
            self.assertEqual(wiki.lookup_with_fallback("Was ist Elektrostatik?", web, render, local=local),
                             {"text": "V4"})
            web.assert_called_once()
            render.assert_not_called()

    def test_fresh_query_skips_local_search(self):
        local = Mock()
        web = Mock(return_value={"text": "V4"})
        for q in ("Wie hoch ist der Goldpreis heute?", "Wer ist Bundeskanzler?", "What is the latest Python version?"):
            self.assertTrue(wiki.eligible_question(q))
            self.assertEqual(wiki.lookup_with_fallback(q, web, Mock(), local=local), {"text": "V4"})
        local.assert_not_called()

    def test_nonknowledge_work_stays_on_old_route(self):
        for q in ("Hallo", "Berechne 5 + 7", "Schreibe Python-Code", "Was ist meine Adresse?"):
            self.assertFalse(wiki.eligible_question(q))

if __name__ == "__main__":
    unittest.main()
