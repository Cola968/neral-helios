"""Free web search: parser, provider isolation, quotas, HTTP, billing and agent consent."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request, ProxyHandler

from flask import Flask
from neral_web_search import (WebSearch, WebSearchError, WebSearchInputError,
                              WebSearchRateLimitError, parse_search_results,
                              safe_result_url, fixed_search_opener, NoRedirect)
from neral_web_api import register_web_search_api
from billing.entitlements import entitlements_for

HTML = '''<html><div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.org%2Fnews">Example <b>title</b></a><a class="result__snippet">Science &amp; news.</a></div>
<div class="result"><a class="result__a" href="https://docs.python.org/3/">Python Docs</a><div class="result__snippet">Official docs</div></div>
<div class="result"><a class="result__a" href="javascript:alert(1)">Bad</a></div>
<div class="result"><a class="result__a" href="https://localhost/private">Private</a></div></html>'''

class Response:
    headers = {"Content-Type": "text/html; charset=utf-8"}
    def __init__(self, body): self.body=body
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def read(self,n): return self.body[:n]

class Opener:
    def __init__(self,body=HTML.encode()): self.body=body;self.calls=[]
    def open(self,request,timeout): self.calls.append((request,timeout));return Response(self.body)

class WebSearchContracts(unittest.TestCase):
    def test_results_are_cited_and_untrusted(self):
        rows=parse_search_results(HTML)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0],{"title":"Example title","url":"https://www.example.org/news","snippet":"Science & news."})
        self.assertEqual(rows[1]["url"],"https://docs.python.org/3/")

    def test_never_link_to_local_or_unsafe_urls(self):
        rejected=["http://example.org", "javascript:alert(1)", "file:///etc/passwd",
                  "https://localhost/a","https://api.local/a","https://127.0.0.1/a",
                  "https://10.0.0.1/a", "https://[::1]/", "https://user:p@example.org/",
                  "https://example.org:8080/","https://evil/","https://foo..bar.com",
                  "https://example.org/%0aSet-Cookie:secret","https://example.org\\@evil.test/"]
        for value in rejected:
            with self.subTest(value=value): self.assertIsNone(safe_result_url(value))
        self.assertEqual(safe_result_url("https://example.org/a"),"https://example.org/a")

    def test_only_explicit_query_is_sent_to_fixed_website(self):
        opener=Opener()
        result=WebSearch(opener=opener).search("  aktuelle   Raumfahrt News  ")
        self.assertEqual(result["query"],"aktuelle Raumfahrt News")
        self.assertTrue(result["external_request"])
        self.assertFalse(result["model_generated"])
        self.assertEqual(len(result["results"]),2)
        self.assertEqual(len(opener.calls),1)
        request, timeout=opener.calls[0]
        self.assertTrue(request.full_url.startswith("https://html.duckduckgo.com/html/?q="))
        self.assertIn("aktuelle+Raumfahrt+News",request.full_url)
        self.assertLessEqual(timeout,8)
        self.assertEqual(request.get_method(),"GET")
        self.assertNotIn("PRIVATE",request.full_url)

    def test_bad_queries_do_not_leave_host(self):
        opener=Opener();search=WebSearch(opener=opener)
        for q in (None, "", " ", "a", "x"*161, "one\ntwo", "\x00secret"):
            with self.subTest(query=q),self.assertRaises(WebSearchInputError):search.search(q)
        self.assertEqual(opener.calls,[])

    def test_provider_failures_sanitized(self):
        class Broken:
            def open(self,request,timeout): raise URLError("/private/my-key")
        with self.assertRaises(WebSearchError) as exc:WebSearch(opener=Broken()).search("space news")
        self.assertNotIn("private",str(exc.exception))
        self.assertNotIn("my-key",str(exc.exception))

    def test_response_size_limited(self):
        with self.assertRaises(WebSearchError):
            WebSearch(opener=Opener(b"x"*600_002)).search("space news")

    def test_search_proxies_and_redirects_disabled(self):
        opener=fixed_search_opener()
        self.assertTrue(any(isinstance(h,NoRedirect) for h in opener.handlers))
        self.assertFalse(any(isinstance(h,ProxyHandler) and h.proxies for h in opener.handlers))
        with self.assertRaises(WebSearchError):
            NoRedirect().redirect_request(Request("https://html.duckduckgo.com/html/?q=x"),None,302,"redirect",{},"https://evil.example/leak")

    def test_rate_limit_is_resource_protection_not_plan_paywall(self):
        opener=Opener();engine=WebSearch(opener=opener,max_per_minute=2)
        engine.search("first search");engine.search("second search")
        with self.assertRaises(WebSearchRateLimitError):engine.search("third search")
        self.assertEqual(len(opener.calls),2)
        for plan in ("free","plus","pro"):
            self.assertIs(entitlements_for(plan)["web_search"],True)

    def test_empty_provider_page_does_not_invent_results(self):
        data=WebSearch(opener=Opener(b"<html>no matches</html>")).search("space news")
        self.assertEqual(data["results"],[])


class HttpSearchContracts(unittest.TestCase):
    def test_explicit_only_and_free_status(self):
        app=Flask("web-search-api-test")
        class Search:
            def __init__(self):self.calls=[]
            def search(self,query):
                self.calls.append(query)
                return {"query":query,"results":[],"model_generated":False}
        fake=Search()
        register_web_search_api(app,fake)
        client=app.test_client()
        status=client.get("/api/web/search/status").json
        self.assertTrue(status["free_for_all"])
        self.assertFalse(status["requires_api_key"])
        self.assertFalse(status["live_verified"])
        self.assertEqual(client.post("/api/web/search",json={"query":"NASA"}).status_code,200)
        self.assertEqual(fake.calls,["NASA"])
        for extra in ({"query":"NASA","history":["secret"]},{"query":"NASA","plan":"pro"},[],{"history":[] }):
            self.assertEqual(client.post("/api/web/search",json=extra).status_code,400)
        self.assertEqual(client.post("/api/web/search",data="bad").status_code,415)
        self.assertEqual(fake.calls,["NASA"])

    def test_main_app_inherits_cookie_and_host_origin_guards(self):
        import app as runtime
        c=runtime.app.test_client()
        self.assertEqual(c.post("/api/web/search",json={"query":"NASA"},base_url="http://localhost").status_code,401)
        c.get("/",base_url="http://localhost")
        with patch.object(runtime.web_search_service,"search",return_value={"query":"NASA","results":[]}):
            self.assertEqual(c.post("/api/web/search",json={"query":"NASA"},base_url="http://localhost").status_code,200)
            for headers in ({"Origin":"https://evil.invalid"},{"Host":"evil.invalid"},{"Sec-Fetch-Site":"cross-site"}):
                self.assertEqual(c.post("/api/web/search",json={"query":"NASA"},headers=headers,base_url="http://localhost").status_code,403)


class ModelAgentConsentContracts(unittest.TestCase):
    class Backend:
        def __init__(self):self.responses=[];self.calls=[]
        def respond(self,messages):
            self.calls.append(messages)
            return json.dumps(self.responses.pop(0))

    def test_model_cannot_send_queries_without_explicit_user_consent(self):
        from neral_agent_autonomy import AutonomousAgent
        with tempfile.TemporaryDirectory() as folder:
            b=self.Backend()
            b.responses=[{"action":"web_search","query":"secret_contents_of_private_file"}, {"action":"finish","summary":"Not allowed"}]
            class Search:
                calls=[]
                def search(self,q):self.calls.append(q);return {"results":[]}
            fake=Search()
            agent=AutonomousAgent(Path(folder),b,web_search_service=fake)
            result=agent.run(agent.start("Inspect project")["id"])
            self.assertEqual(result["status"],"completed")
            self.assertEqual(fake.calls,[])
            self.assertIn("blocked",str(result["history"]))
            with self.assertRaises(ValueError):agent.start("Inspect",web_query="NASA")

    def test_approved_exact_query_is_used_once_even_if_model_supplies_other_text(self):
        from neral_agent_autonomy import AutonomousAgent
        with tempfile.TemporaryDirectory() as folder:
            b=self.Backend()
            b.responses=[{"action":"web_search","query":"PRIVATE_ENV_VARIABLE"},
                         {"action":"web_search","query":"PRIVATE_ENV_VARIABLE"},
                         {"action":"finish","summary":"Cited sources; one web request."}]
            class Search:
                def __init__(self):self.calls=[]
                def search(self,q):
                    self.calls.append(q)
                    return {"query":q,"provider":"fixture","results":[{"title":"NASA","url":"https://nasa.gov","snippet":"A"}]}
            fake=Search()
            agent=AutonomousAgent(Path(folder),b,web_search_service=fake)
            run=agent.run(agent.start("Find space news",allow_web_search=True,web_query="  NASA news  ")["id"])
            self.assertEqual(run["status"],"completed")
            self.assertEqual(fake.calls,["NASA news"])
            self.assertIn("external_untrusted",str(run["history"]))
            self.assertNotIn("PRIVATE_ENV_VARIABLE",str(run["history"]))
            self.assertTrue(run["web_search_used"])




BING_XML = '''<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Artemis &amp; NASA</title><link>https://www.nasa.gov/news</link><description>Latest launch &amp; science</description></item>
<item><title>Unsafe</title><link>https://localhost/private</link><description>Ignore</description></item>
</channel></rss>'''


class BingFallbackTests(unittest.TestCase):
    def test_rss_parser_rejects_private_destinations(self):
        from neral_web_search import parse_bing_rss
        rows = parse_bing_rss(BING_XML)
        self.assertEqual(rows, [{"title":"Artemis & NASA","url":"https://www.nasa.gov/news",
                                 "snippet":"Latest launch & science"}])

    def test_bing_is_used_with_exact_same_query_when_ddg_returns_zero(self):
        class Routing:
            def __init__(self): self.calls = []
            def open(self, request, timeout):
                self.calls.append(request.full_url)
                if "duckduckgo.com" in request.full_url:
                    return Response(b"<html>blocked</html>")
                rss = Response(BING_XML.encode("utf-8"))
                rss.headers = {"Content-Type":"application/rss+xml; charset=utf-8"}
                return rss
        opener = Routing()
        data = WebSearch(opener=opener).search("Artemis NASA")
        self.assertEqual(data["provider"],"Bing RSS")
        self.assertEqual(len(data["results"]),1)
        self.assertEqual(len(opener.calls),2)
        self.assertIn("duckduckgo.com",opener.calls[0])
        self.assertIn("www.bing.com/search",opener.calls[1])
        self.assertTrue(all("Artemis+NASA" in url for url in opener.calls))
        self.assertEqual(data["providers_contacted"], ["DuckDuckGo HTML","Bing RSS"])

    def test_bing_rss_localized_preserves_exact_query(self):
        from urllib.parse import urlsplit,parse_qs
        class Capturing:
            def __init__(self):self.urls=[]
            def open(self,req,timeout):
                self.urls.append(req.full_url)
                return type('Response',(),{
                    'headers':{'Content-Type':'text/xml'},
                    '__enter__':lambda self:self,
                    '__exit__':lambda self,*a:False,
                    'read':lambda self,n:BING_XML.encode()})()
        for query in ('Flughafen Frankfurt','Python Dokumentation','NASA Artemis'):
            stub=Capturing()
            self.assertTrue(WebSearch(opener=stub).search_bing(query))
            params=parse_qs(urlsplit(stub.urls[0]).query)
            self.assertEqual(params['q'],[query])
            self.assertEqual(params['cc'],['de'])
            self.assertEqual(params['setlang'],['de'])


    def test_wikipedia_search_uses_only_fixed_endpoint_and_safe_clean_content(self):
        from urllib.parse import parse_qs,urlsplit
        import json
        response=json.dumps({"query":{"search":[
             {"title":"Flughafen Frankfurt Main","snippet":"<span>Internationaler</span> Flughafen"},
             {"title":"NASA Artemis","snippet":"<script>bad()</script> Space program"},
             {"title":"","snippet":"Empty"}]}}).encode()
        class Fake:
            def __init__(self):self.requests=[]
            def open(self,req,timeout):
                self.requests.append(req.full_url)
                class Response:
                    headers={"Content-Type":"application/json"}
                    def __enter__(self):return self
                    def __exit__(self,*a):return False
                    def read(self,n):return response
                return Response()
        opener=Fake()
        rows=WebSearch(opener=opener).search_wikipedia("Flughafen Frankfurt")
        self.assertEqual(len(rows),2)
        self.assertTrue(rows[0]["url"].startswith("https://de.wikipedia.org/wiki/"))
        self.assertNotIn("<",rows[0]["snippet"])
        self.assertNotIn("<",rows[1]["snippet"])
        parsed=urlsplit(opener.requests[0])
        self.assertEqual(parsed.hostname,"de.wikipedia.org")
        self.assertEqual(parse_qs(parsed.query)["srsearch"],["Flughafen Frankfurt"])

    def test_bing_rss_rejects_unsafe_external_entities(self):
        from neral_web_search import parse_bing_rss
        with self.assertRaises(WebSearchError):
            parse_bing_rss('<!DOCTYPE rss [<!ENTITY x SYSTEM "file:///etc/passwd">]><rss>&x;</rss>')

if __name__=="__main__":unittest.main()
