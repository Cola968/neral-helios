"""Topic-independent publisher discovery without aggregator access."""
import unittest
from datetime import datetime, timezone
from neral_news_articles import publisher_links, publisher_document_links, find_publisher_article
from neral_v43_live_web import route_live_query

HOME="https://publisher.example.com/"
TITLE="NASA meldet Fortschritte bei der Artemis Mission"
URL=HOME+"wissenschaft/nasa-meldet-fortschritte-bei-der-artemis-mission"
HIT={"publisher_url":HOME,"url":"https://news.google.com/rss/articles/fixture","title":TITLE+" - Publisher","published_at":datetime.now(timezone.utc).isoformat()}
P1="Die NASA hat Fortschritte bei der Artemis Mission bekannt gegeben. Die Raumfahrtagentur beschreibt neue Tests und den aktuellen Stand der Arbeiten."
P2="Die nächsten Schritte sollen technische Komponenten und das Zusammenspiel der Systeme prüfen. Das Team bereitet weitere Untersuchungen vor."
ARTICLE=f"<article><h1>{TITLE}</h1><p>{P1}</p><p>{P2}</p></article>"

class GenericDiscoveryTests(unittest.TestCase):
    def test_slug_only_link_is_a_candidate_not_evidence(self):
        markup=f'<a href="{URL}">Weiterlesen</a><a href="/sport/fussball">Weiterlesen</a>'
        self.assertEqual(publisher_links(markup,HOME,HIT["title"]),[URL])

    def test_public_news_sitemap_matches_title(self):
        xml=f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"><url><loc>{URL}</loc><news:news><news:title>{TITLE}</news:title></news:news></url><url><loc>https://external.example.com/other</loc><news:news><news:title>{TITLE}</news:title></news:news></url></urlset>'
        self.assertEqual(publisher_document_links(xml,HOME,HIT["title"]),[URL])

    def test_public_rss_matches_title_not_unrelated_link(self):
        xml=f'<rss><channel><item><title>{TITLE}</title><link>{URL}</link></item><item><title>Fußball</title><link>{HOME}sport</link></item></channel></rss>'
        self.assertEqual(publisher_document_links(xml,HOME,HIT["title"]),[URL])

    def test_sitemap_fallback_reads_real_article(self):
        def fetch(url):
            if url==HOME:
                return {"url":HOME,"markup":"<html><body>Startseite</body></html>"}
            if url==HOME+"sitemap.xml":
                return {"url":url,"markup":f"<urlset><url><loc>{URL}</loc></url></urlset>"}
            if url==URL:
                return {"url":URL,"markup":ARTICLE}
            return None
        result=find_publisher_article(HIT,"Aktuelle Nachrichten zur NASA",fetch=fetch)
        self.assertIsNotNone(result)
        self.assertTrue(result["page_read"])
        self.assertIn(P2,result["page_text"])

    def test_no_fabricated_article_when_sitemap_link_wrong(self):
        def fetch(url):
            if url==HOME:
                return {"url":HOME,"markup":"<html></html>"}
            if url==HOME+"sitemap.xml":
                return {"url":url,"markup":f"<urlset><url><loc>{URL}</loc></url></urlset>"}
            if url==URL:
                return {"url":URL,"markup":"<article><h1>Fußball</h1><p>"+P1+"</p><p>"+P2+"</p></article>"}
            return None
        self.assertIsNone(find_publisher_article(HIT,"Aktuelle Nachrichten zur NASA",fetch=fetch))

    def test_benchmark_never_routes_public_news(self):
        for q in ("Aktuelle Nachrichten zur NASA","Wie ist das Wetter morgen in Berlin?","Welche Version von Python ist aktuell?"):
            self.assertIsNone(route_live_query(q,benchmark=True))

if __name__=="__main__":
    unittest.main()
