"""Topic-independent publisher discovery without aggregator access."""
import unittest
from datetime import datetime, timezone, timedelta
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

    def test_current_feed_does_not_make_a_reposted_old_article_fresh(self):
        def discover(date, second_date=None):
            meta = f'<meta property="article:published_time" content="{date}">'
            if second_date:
                meta += f'<meta name="pubdate" content="{second_date}">'
            def fetch(url):
                if url == HOME:
                    return {"url": HOME, "markup": f'<a href="{URL}">{TITLE}</a>'}
                if url == URL:
                    return {"url": URL, "markup": meta + ARTICLE}
                return None
            return find_publisher_article(HIT, "Aktuelle Nachrichten zur NASA", fetch=fetch)
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=30)).isoformat()
        fresh = (now - timedelta(days=1)).isoformat()
        self.assertIsNone(discover(old))
        self.assertIsNone(discover(fresh, old))
        result = discover(fresh)
        self.assertEqual(result["published_at"], fresh)
        self.assertEqual(result["date_origin"], "publisher_metadata")

    def test_relative_feed_and_article_links_use_actual_document_urls(self):
        index = HOME + "public/news/"
        feed = index + "feeds/updates.xml"
        article = index + "stories/mission-report"
        pages = {
            HOME: {"url": index, "markup": '<link rel="alternate" type="application/rss+xml; charset=utf-8" href="feeds/updates.xml">'},
            feed: {"url": feed, "markup": f"<rss><channel><item><title>{TITLE}</title><link>../stories/mission-report</link></item></channel></rss>"},
            article: {"url": article, "markup": ARTICLE},
        }
        result = find_publisher_article(HIT, "Aktuelle Nachrichten zur NASA", fetch=pages.get)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], article)
        self.assertTrue(result["page_read"])

    def test_relative_feed_discovery_rejects_off_publisher_document_redirect(self):
        feed = HOME + "feed.xml"
        attempted = []
        def fetch(url):
            attempted.append(url)
            if url == HOME:
                return {"url": HOME, "markup": '<link rel="alternate" type="application/rss+xml" href="feed.xml">'}
            if url == feed:
                return {"url": "https://other.example.com/feed.xml", "markup": (
                        f"<rss><channel><item><title>{TITLE}</title>"
                        f"<link>{URL}</link></item></channel></rss>")}
            if url == URL:
                return {"url": URL, "markup": ARTICLE}
            return None
        self.assertIsNone(find_publisher_article(HIT, "Aktuelle Nachrichten zur NASA", fetch=fetch))
        self.assertNotIn(URL, attempted)

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
