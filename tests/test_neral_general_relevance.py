"""Independent regression cases for the topic gate; no real network."""
import unittest
from datetime import datetime,timezone
from unittest.mock import Mock
from neral_web_search import live_topic_matches, live_topic_hits, news_query_variants
from neral_v43_live_web import search_live_sources

NOW=datetime.now(timezone.utc).isoformat()
def hit(title,slug,body=""):
    return {"title":title,"url":"https://example.org/"+slug,"snippet":body,
            "published_at":NOW}
def page(h,body):
    return {**h,"page_read":True,"page_text":body}
class TopicGate(unittest.TestCase):
    def test_software_version_requires_named_product(self):
        q="Welche Version von Python ist aktuell?"
        self.assertFalse(live_topic_matches(q,"Version Rechtschreibung Bedeutung"))
        self.assertTrue(live_topic_matches(q,"Python release downloads and versions"))
    def test_mission_is_not_same_as_agency(self):
        q="Welche aktuellen Nachrichten gibt es zur NASA Artemis Mission?"
        self.assertFalse(live_topic_matches(q,"NASA sucht Partner für einen Kernreaktor"))
        self.assertTrue(live_topic_matches(q,"NASA prepares Artemis II crew"))
    def test_location_is_not_same_as_category(self):
        q="Wie ist das Wetter morgen in Berlin?"
        self.assertFalse(live_topic_matches(q,"Wetter aktuell und Wettervorhersage"))
        self.assertTrue(live_topic_matches(q,"Berlin weather tomorrow forecast"))
    def test_german_english_technology_synonyms(self):
        q="Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?"
        self.assertFalse(live_topic_matches(q,"K+S Aktie steigt"))
        self.assertTrue(live_topic_matches(q,"KI-Dokumentation spart Zeit"))
        self.assertTrue(live_topic_matches(q,"Artificial intelligence in healthcare"))
    def test_named_league_is_not_replaced_by_generic_sport(self):
        q = "Welche aktuellen Nachrichten gibt es zur Fußball Bundesliga?"
        self.assertFalse(live_topic_matches(q, "Football: Premier League results"))
        self.assertTrue(live_topic_matches(q, "Bundesliga: Ergebnisse vom Samstag"))
        variants = news_query_variants(q)
        self.assertTrue(variants)
        self.assertTrue(all("bundesliga" in query.casefold() for query in variants))
        self.assertTrue(live_topic_matches("Aktuelle Nachrichten über Fußball", "Football news"))
        self.assertFalse(live_topic_matches("Latest football news about Bundesliga", "Premier League football"))
        self.assertTrue(live_topic_matches("Latest football news about Bundesliga", "Bundesliga results"))
    def test_politics_requires_actual_institution(self):
        q="Welche aktuellen Nachrichten gibt es zum Deutschen Bundestag?"
        self.assertFalse(live_topic_matches(q,"Der deutsche Sozialstaat wird in der Ukraine verteidigt"))
        self.assertTrue(live_topic_matches(q,"Bundestag debattiert Haushalt"))
    def test_metadata_rejects_wrong_search_hits(self):
        q="Welche Version von Python ist aktuell?"
        wrong=hit("Version Rechtschreibung","version")
        good=hit("Python downloads","python")
        self.assertEqual(live_topic_hits([wrong,good],q),[good])
    def test_body_not_metadata_is_evidence(self):
        q="Welche Version von Python ist aktuell?"
        good=hit("Python release","python")
        engine=Mock()
        engine.search.return_value={"provider":"fixture","results":[good]}
        out=search_live_sources(engine,q,reader=Mock(return_value=[page(good,"Windows Update ist verfügbar.")]))
        self.assertEqual(out["status"],"empty")
        self.assertFalse(out["context"])
    def test_wrong_first_three_pages_do_not_prevent_later_match(self):
        q="Welche Version von Python ist aktuell?"
        hits=[hit("Python releases "+str(i),"python-"+str(i)) for i in range(5)]
        engine=Mock()
        engine.search.return_value={"provider":"fixture","results":hits}
        def reader(batch,query):
            return [page(h,"Python stable release 3.14.1." if h is hits[4] else "Windows Update.") for h in batch[:3]]
        out=search_live_sources(engine,q,reader=reader)
        self.assertEqual(out["status"],"ok")
        self.assertEqual([x["url"] for x in out["sources"]],[hits[4]["url"]])
    def test_version_page_must_be_about_the_requested_product(self):
        q="Welche Version von Python ist aktuell?"
        wrong=hit("PyCharm","pycharm","Python release versions")
        right=hit("Python releases","python")
        self.assertEqual(live_topic_hits([wrong,right],q),[right])
    def test_version_source_must_have_actual_release_number(self):
        q="Welche Version von Python ist aktuell?"
        good=hit("Python releases","python")
        engine=Mock()
        engine.search.return_value={"provider":"fixture","results":[good]}
        out=search_live_sources(engine,q,reader=Mock(return_value=[page(good,"Python has many releases and versions.")]))
        self.assertEqual(out["status"],"empty")
    def test_irrelevant_feed_headline_is_not_displayed(self):
        q="Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?"
        wrong={**hit("Düngerkonzern Aktie steigt","stock"),"provider":"Google News RSS"}
        right={**hit("KI-Dokumentation spart Zeit","ki"),"provider":"Google News RSS"}
        engine=Mock()
        engine.search.return_value={"provider":"fixture","results":[],"headlines":[wrong,right]}
        out=search_live_sources(engine,q,reader=Mock())
        self.assertEqual(out["status"],"headlines_only")
        self.assertEqual([x["url"] for x in out["sources"]],[right["url"]])

if __name__=="__main__":
    unittest.main()
