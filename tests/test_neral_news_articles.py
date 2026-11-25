"""Regression contracts for public publisher article resolution."""
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock
from neral_news_articles import publisher_links, article_text, find_publisher_article

HOME="https://publisher.example.com/"
URL=HOME+"news/airport-growth"
HEAD="Flughafen Berlin Brandenburg plant neue Flugverbindungen"
BODY="Der Flughafen Berlin Brandenburg plant neue Flugverbindungen. Die Fluggesellschaften haben neue Ziele angekündigt. Reisende erhalten zusätzliche Auswahl. "
HIT={"publisher_url":HOME,"title":HEAD+" - Publisher","published_at":datetime.now(timezone.utc).isoformat(),"url":"https://news.google.com/rss/articles/opaque"}

class PublicPublisherTests(unittest.TestCase):
    def test_index_match_same_publisher(self):
        html=f'<a href="/news/airport-growth"><h2>{HEAD}</h2></a><a href="https://external.example.com/other">{HEAD}</a>'
        self.assertEqual(publisher_links(html,HOME,HIT["title"]),[URL])

    def test_unrelated_or_unsafe_links_rejected(self):
        html='<a href="/sports">Sport Nachrichten</a><a href="https://127.0.0.1/">{}</a>'.format(HEAD)
        self.assertEqual(publisher_links(html,HOME,HIT["title"]),[])

    def test_visible_article_only(self):
        html=f'<html><head><title>{HEAD}</title></head><nav>Navigation</nav><article><h1>{HEAD}</h1><p>{BODY*3}</p><p>{BODY*2}</p></article><footer>Footer</footer></html>'
        result=article_text(html)
        self.assertIn(HEAD,result["title"])
        self.assertIn(BODY.strip(),result["text"])
        self.assertNotIn("Navigation",result["text"])
        self.assertNotIn("Footer",result["text"])

    def test_paywall_and_landing_page_are_not_articles(self):
        self.assertIsNone(article_text('<html><title>Google News</title><p>Continue to article</p></html>'))
        self.assertIsNone(article_text('<article><h1>News</h1><p>Nur für Abonnenten. Jetzt anmelden.</p></article>'))

    def test_real_article_requires_matching_body_not_only_headline(self):
        index=f'<a href="/news/airport-growth">{HEAD}</a>'
        article=f'<article><h1>{HEAD}</h1><p>{BODY*3}</p><p>{BODY*2}</p></article>'
        fetch=Mock(side_effect=[{"url":HOME,"markup":index},{"url":URL,"markup":article}])
        result=find_publisher_article(HIT,"Aktuelle Nachrichten Flughafen BER",fetch=fetch)
        self.assertEqual(result["url"],URL)
        self.assertIn(BODY.strip(),result["page_text"])
        self.assertTrue(result["page_read"])

    def test_mismatched_redirect_rejected(self):
        index=f'<a href="/news/airport-growth">{HEAD}</a>'
        article=f'<article><h1>{HEAD}</h1><p>{BODY*3}</p><p>{BODY*2}</p></article>'
        def fetch(url):
            if url==HOME:
                return {"url":HOME,"markup":index}
            if url==URL:
                return {"url":"https://other.example.com/news","markup":article}
            return None
        self.assertIsNone(find_publisher_article(HIT,"Aktuelle Nachrichten Flughafen BER",fetch=fetch))

    def test_ordinary_aggregator_http_redirect_to_publisher(self):
        article=f'<article><h1>{HEAD}</h1><p>{BODY*3}</p><p>{BODY*2}</p></article>'
        def fetch(url):
            if url==HIT["url"]:
                return {"url":URL,"markup":article}
            return None
        result=find_publisher_article(HIT,"Aktuelle Nachrichten Flughafen BER",fetch=fetch)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"],URL)

    def test_multi_source_fallback_after_first_four_fail(self):
        from neral_news_articles import discover_publisher_articles
        hits=[{**HIT,"url":HIT["url"]+str(i)} for i in range(6)]
        def resolver(hit,query):
            if hit is hits[5]:
                return {**hit,"url":URL,"page_read":True,"article_chars":500}
            return None
        out=discover_publisher_articles(hits,"Aktuelle Nachrichten Flughafen BER",resolver=resolver)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]["url"],URL)

if __name__=="__main__":
    unittest.main()
