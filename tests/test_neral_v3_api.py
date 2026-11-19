"""V3 end-to-end HTTP contract: a read page is not a search-engine snippet."""
import unittest
from unittest.mock import patch
from flask import Flask
from neral_web_api import register_web_search_api

URL="https://example.org/oz"
TEXT="Goldpreis aktuell: 3.650,20 EUR pro Feinunze. Dieser Wert steht im Haupttext."

class FakeSearch:
    def search(self,query,on_provider=None):
        return {"query":query,"provider":"DuckDuckGo HTML",
                "providers_contacted":["DuckDuckGo HTML"],
                "results":[{"url":URL,"title":"Goldpreis Feinunze",
                            "snippet":"Goldpreis vom Markt"}]}

class V3WebAPITests(unittest.TestCase):
    def setUp(self):
        app=Flask(__name__)
        app.testing=True
        def fake_page(url,timeout=3.5):
            return {"url":url,"text":TEXT,"read":True}
        register_web_search_api(app,service=FakeSearch(),page_reader=fake_page)
        self.client=app.test_client()

    def test_search_response_contains_fetched_main_text_and_original_url(self):
        with patch("neral_web_api.verify_public_https_url",side_effect=lambda url:url):
            response=self.client.post("/api/web/search",json={
                "query":"Goldpreis Feinunze","verify_links":True})
        self.assertEqual(response.status_code,200)
        data=response.json
        self.assertEqual(data["pages"][0]["text"],TEXT)
        self.assertEqual(data["pages"][0]["url"],URL)
        self.assertEqual(data["page_read"],"main_text_https")
        self.assertIn("snippet",data["previews"][0])
        self.assertNotEqual(data["previews"][0]["snippet"],TEXT)

    def test_unreadable_page_never_gets_fake_page_proof(self):
        app=Flask(__name__+"missing")
        app.testing=True
        register_web_search_api(app,service=FakeSearch(),
                                page_reader=lambda url,timeout=3.5:None)
        with patch("neral_web_api.verify_public_https_url",side_effect=lambda url:url):
            response=app.test_client().post("/api/web/search",json={
                "query":"Goldpreis Feinunze","verify_links":True})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json["pages"],[])
        self.assertEqual(response.json["page_read"],"not_available")


if __name__=="__main__":unittest.main()
