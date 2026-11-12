"""Real-time provider and reachable-source events for NERAL search."""
import json
import unittest
from unittest.mock import patch
from flask import Flask
from neral_web_api import register_web_search_api

class FakeLiveSearch:
    def __init__(self): self.calls=[]
    def search(self,query,on_provider=None):
        self.calls.append(query)
        if on_provider:
            on_provider("DuckDuckGo HTML")
            on_provider("Bing RSS")
        return {"query":query,"providers_contacted":["DuckDuckGo HTML","Bing RSS"],
          "results":[{"title":"Airport BER","url":"https://airport.example/ber","snippet":"BER flights"},
                     {"title":"Unsafe","url":"https://localhost/private","snippet":"Private"},
                     {"title":"Other","url":"https://space.example/news","snippet":"Space news"}]}

class LiveSourcesTests(unittest.TestCase):
    def setUp(self):
        self.app=Flask(__name__);self.fake=FakeLiveSearch();register_web_search_api(self.app,self.fake)
        self.client=self.app.test_client()
    def events(self,data):
        response=self.client.post('/api/web/search/stream',json=data)
        return response,[json.loads(chunk) for chunk in response.response]
    def test_stream_emits_only_contacted_providers_and_reachable_sources(self):
        def reachable(url):return url if url.startswith('https://airport.example/') else None
        with patch('neral_web_api.verify_public_https_url',side_effect=reachable):
            response,events=self.events({'query':'Suche BER','verify_links':True})
        self.assertEqual(response.status_code,200)
        self.assertEqual([e['name'] for e in events if e['type']=='provider'],['DuckDuckGo HTML','Bing RSS'])
        sources=[e['source'] for e in events if e['type']=='source']
        self.assertEqual(len(sources),1)
        self.assertTrue(sources[0]['reachable'])
        result=next(e['result'] for e in events if e['type']=='result')
        self.assertEqual(result['results'],sources)
        self.assertEqual(events[-1]['type'],'end')
    def test_invalid_requests_produce_no_search_or_provider_events(self):
        for payload in ({'query':'x','verify_links':True},{'query':'BER','verify_links':False},
                        {'query':'BER','verify_links':True,'url':'https://localhost'},
                        {'verify_links':True}):
            response=self.client.post('/api/web/search/stream',json=payload)
            self.assertEqual(response.status_code,400)
        self.assertEqual(self.fake.calls,[])
    def test_search_error_is_in_stream_without_claiming_a_source(self):
        class Broken(FakeLiveSearch):
            def search(self,query,on_provider=None):raise RuntimeError('private diagnostic')
        app=Flask(__name__);register_web_search_api(app,Broken())
        response=app.test_client().post('/api/web/search/stream',json={'query':'Artemis','verify_links':True})
        payload=[json.loads(line) for line in response.response]
        self.assertEqual(response.status_code,200)
        self.assertEqual([x['type'] for x in payload],['error','end'])
        self.assertNotIn('private diagnostic',repr(payload))


    def test_provider_fallback_when_ddg_hits_exist_but_none_are_reachable(self):
        class Scripted:
            def __init__(self): self.bing_calls=[]
            def search(self, query, on_provider=None):
                if on_provider: on_provider("DuckDuckGo HTML")
                return {"query":query,"provider":"DuckDuckGo HTML",
                  "providers_contacted":["DuckDuckGo HTML"],
                  "results":[{"title":"Blocked","url":"https://example.org/blocked"}]}
            def search_bing(self, query, on_provider=None):
                self.bing_calls.append(query)
                if on_provider: on_provider("Bing RSS")
                return [{"title":"Berlin Airport","url":"https://example.org/ber",
                         "snippet":"Terminals and flights"}]
        for stream in (True, False):
            service=Scripted()
            app=Flask(__name__);register_web_search_api(app,service)
            url="/api/web/search/stream" if stream else "/api/web/search"
            with patch('neral_web_api.verify_public_https_url',
                       side_effect=lambda link: link if link.endswith("/ber") else None):
                response=app.test_client().post(url,json={"query":"Berlin Airport","verify_links":True})
                if stream:
                    events=[json.loads(line) for line in response.response]
                    result=next(x["result"] for x in events if x["type"]=="result")
                    self.assertEqual([x["name"] for x in events if x["type"]=="provider"],
                                     ["DuckDuckGo HTML","Bing RSS"])
                else:
                    result=response.get_json()
            self.assertEqual(response.status_code,200)
            self.assertEqual(service.bing_calls,["Berlin Airport"])
            self.assertEqual(result["results"][0]["title"],"Berlin Airport")
            self.assertEqual(result["provider"],"Bing RSS")
            self.assertEqual(result["providers_contacted"],["DuckDuckGo HTML","Bing RSS"])


    def test_checked_offtopic_ddg_hits_do_not_block_relevant_bing_fallback(self):
        class Scripted:
            def search(self,query,on_provider=None):
                if on_provider: on_provider("DuckDuckGo HTML")
                return {"query":query,"provider":"DuckDuckGo HTML",
                        "providers_contacted":["DuckDuckGo HTML"],
                        "results":[{"url":"https://unrelated.example/1","title":"Rezept für Pfannkuchen","snippet":"Frühstück"},
                                   {"url":"https://unrelated.example/2","title":"Fernsehprogramm","snippet":"TV"}]}
            def search_bing(self,query,on_provider=None):
                if on_provider: on_provider("Bing RSS")
                return [{"url":"https://x.ai/grok","title":"Grok - xAI","snippet":"KI-Assistent Grok"}]
        app=Flask(__name__);register_web_search_api(app,Scripted())
        with patch('neral_web_api.verify_public_https_url',side_effect=lambda url:url):
            resp=app.test_client().post('/api/web/search/stream',
                          json={"query":"grok ai","verify_links":True})
            events=[json.loads(x) for x in resp.response]
        sources=[x["source"] for x in events if x["type"]=="source"]
        result=next(x["result"] for x in events if x["type"]=="result")
        self.assertEqual([x["name"] for x in events if x["type"]=="provider"],
                         ["DuckDuckGo HTML","Bing RSS"])
        self.assertEqual([x["title"] for x in sources],["Grok - xAI"])
        self.assertEqual([x["title"] for x in result["results"]],["Grok - xAI"])


    def test_public_discovered_result_survives_failed_link_probe_for_all_searches(self):
        """An unverified search-engine result is still an actual search result."""
        examples=[
            ("Flughafen Frankfurt","Frankfurt Airport – Fraport","https://www.frankfurt-airport.com/en.html"),
            ("NASA Artemis","Artemis NASA Missions","https://www.nasa.gov/humans-in-space/artemis/"),
            ("Python Dokumentation","Python Documentation","https://docs.python.org/3/"),
            ("Wetter Berlin","Berlin Weather Forecast","https://example.org/berlin-weather"),
        ]
        for query,title,url in examples:
            class Search:
                def search(self,query,on_provider=None):
                    if on_provider:on_provider("DuckDuckGo HTML")
                    return {"query":query,"provider":"DuckDuckGo HTML",
                        "providers_contacted":["DuckDuckGo HTML"],
                        "results":[{"title":title,"url":url,"snippet":"Official information"},
                                   {"title":"Unrelated TV cooking","url":"https://example.org/cooking","snippet":"Dinner"}]}
                def search_bing(self,query,on_provider=None):
                    if on_provider:on_provider("Bing RSS")
                    return []
            app=Flask(__name__);register_web_search_api(app,Search())
            with patch("neral_web_api.verify_public_https_url",return_value=None):
                response=app.test_client().post("/api/web/search/stream",
                    json={"query":query,"verify_links":True})
                events=[json.loads(line) for line in response.response]
            result=next(x["result"] for x in events if x["type"]=="result")
            candidates=[x["source"] for x in events if x["type"]=="candidate"]
            self.assertEqual(result["results"],[],query)
            self.assertEqual([x["url"] for x in result["previews"]],[url],query)
            self.assertEqual([x["url"] for x in candidates],[url],query)
            self.assertTrue(result["previews"][0]["discovered"])
            self.assertFalse(result["previews"][0]["reachable"])
            self.assertTrue(result["previews"][0]["topically_relevant"])
            self.assertLess(next(i for i,x in enumerate(events) if x["type"]=="candidate"),
                            next(i for i,x in enumerate(events) if x["type"]=="result"))

    def test_unrelated_previews_are_not_passed_to_any_model_prompt(self):
        class Search:
            def search(self,query,on_provider=None):
                return {"query":query,"provider":"DuckDuckGo HTML",
                    "providers_contacted":["DuckDuckGo HTML"],
                    "results":[{"url":"https://example.org/tv","title":"TV-Programm Berlin",
                                "snippet":"Fernsehserien"},
                               {"url":"http://localhost/private","title":"Frankfurt Airport",
                                "snippet":"untrusted internal link"}]}
            def search_bing(self,query,on_provider=None):
                return []
        app=Flask(__name__);register_web_search_api(app,Search())
        with patch("neral_web_api.verify_public_https_url",return_value=None):
            answer=app.test_client().post("/api/web/search",
                json={"query":"Flughafen Frankfurt","verify_links":True}).get_json()
        self.assertEqual(answer["results"],[])
        self.assertEqual(answer["previews"],[])


    def test_wikipedia_reference_fallback_after_offtopic_bing_results(self):
        class Three:
            def search(self,query,on_provider=None):
                if on_provider:on_provider("DuckDuckGo HTML")
                if on_provider:on_provider("Bing RSS")
                return {"query":query,"provider":"Bing RSS",
                    "providers_contacted":["DuckDuckGo HTML","Bing RSS"],
                    "results":[{"title":"DAX Kurs","url":"https://example.org/boerse",
                                "snippet":"Deutsche Aktien"}]}
            def search_wikipedia(self,query,on_provider=None):
                if on_provider:on_provider("Wikipedia DE")
                self.last=query
                return [{"title":"Flughafen Frankfurt Main","url":"https://de.wikipedia.org/wiki/Flughafen_Frankfurt_Main",
                         "snippet":"Internationaler Flughafen in Frankfurt"}]
        service=Three()
        app=Flask(__name__);register_web_search_api(app,service)
        with patch("neral_web_api.verify_public_https_url",
                   side_effect=lambda url: url):
            response=app.test_client().post("/api/web/search/stream",
              json={"query":"Flughafen Frankfurt","verify_links":True})
            events=[json.loads(x) for x in response.response]
        result=next(x["result"] for x in events if x["type"]=="result")
        self.assertEqual(service.last,"Flughafen Frankfurt")
        self.assertEqual([x["name"] for x in events if x["type"]=="provider"],
            ["DuckDuckGo HTML","Bing RSS","Wikipedia DE"])
        self.assertEqual([x["title"] for x in result["results"]],["Flughafen Frankfurt Main"])
        self.assertEqual(result["results"][0]["provider"],"Wikipedia DE")
