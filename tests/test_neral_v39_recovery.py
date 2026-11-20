"""RED-first source recovery and bounded sandbox regression contracts."""
import json
import unittest
from neral_evidence_agent import run_evidence_agent
from neral_page_reader import read_entity_pages
from neral_v32_verifier import _property_evidence, verified_python_question

class V39SourceRecovery(unittest.TestCase):
    def test_missing_search_results_can_use_fetched_exact_subject_reference(self):
        pages = {
            "https://ref.example.org/freebsd": "The FreeBSD kernel manages the operating system resources.",
            "https://en.wikipedia.org/wiki/NetBSD": "Unlike other systems, the NetBSD kernel requires an MMU.",
        }
        class Search:
            def search(self, query):
                if query.startswith("FreeBSD "):
                    return {"results": [{"url": "https://ref.example.org/freebsd"}]}
                return {"results": [{"url": "https://unrelated.example.org/other"}]}
        observed=[]
        def reader(hits, query):
            observed.extend(h.get("url") for h in hits)
            return [
                {"url": hit["url"], "page_read":True,"page_text": pages[hit["url"]]}
                for hit in hits if hit["url"] in pages
            ]
        result=run_evidence_agent(
            "FreeBSD vs NetBSD: Vergleiche Betriebssystem-Kernel anhand tatsächlich gelesener Webseiten.",
            lambda *a: json.dumps({"facts": [
                {"entity":"FreeBSD","fact":"unknown"},{"entity":"NetBSD","fact":"unknown"}]}),
            lambda *a: None, web_query="FreeBSD NetBSD Betriebssystem-Kernel",
            search=Search(),page_reader=reader)
        self.assertEqual(result["error_code"],None)
        self.assertTrue(result["tool_verified"])
        self.assertIn("NetBSD: NetBSD-Kernel",result["text"])
        self.assertIn("https://en.wikipedia.org/wiki/NetBSD",
                      [s["url"] for s in result["sources"]])
        self.assertIn("https://en.wikipedia.org/wiki/NetBSD",observed)
        self.assertIn("[web2]",result["text"])

    def test_recovery_refuses_unread_and_irrelevant_reference(self):
        pages={"https://en.wikipedia.org/wiki/NetBSD":
               "NetBSD is an operating system. The Linux kernel belongs to Linux."}
        class Search:
            def search(self,query):
                if query.startswith("FreeBSD "):
                    return {"results":[{"url":"https://ref.example.org/freebsd"}]}
                return {"results":[]}
        def reader(hits,query):
            out=[]
            for h in hits:
                if h.get("url") == "https://ref.example.org/freebsd":
                    out.append({"url":h["url"],"page_read":True,
                                "page_text":"The FreeBSD kernel manages memory."})
                elif h.get("url") in pages:
                    out.append({"url":h["url"],"page_read":True,
                                "page_text":pages[h["url"]]})
            return out
        result=run_evidence_agent(
            "FreeBSD vs NetBSD: Vergleiche Betriebssystem-Kernel anhand tatsächlich gelesener Webseiten.",
            lambda *a: "{}",lambda *a: None,
            web_query="FreeBSD NetBSD Betriebssystem-Kernel",search=Search(),page_reader=reader)
        self.assertFalse(result["tool_verified"])
        self.assertEqual(result["error_code"],"source_evidence_missing")
        self.assertNotIn("NetBSD-Kernel",result["text"])

    def test_tenth_reference_candidate_is_not_lost_to_download_ads(self):
        hits=[{"url":f"https://example.org/marketing-{i}"} for i in range(9)]
        hits.append({"url":"https://en.wikipedia.org/wiki/Microsoft_Edge"})
        def reader(batch,query):
            return [{"url":h["url"],"page_read":True,
                     "page_text":"Microsoft Edge is a browser with Blink and V8 engines."}
                    if "wikipedia" in h["url"] else
                    {"url":h["url"],"page_read":True,"page_text":"Edge download marketing."}
                    for h in batch]
        found=read_entity_pages(hits,"Edge browser rendering engine","Edge",
                                reader=reader,max_pages=1,
                                predicate=lambda p:_property_evidence(
                                    p["page_text"],"Edge","engine") is not None)
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]["url"],hits[-1]["url"])

    def test_kernel_cannot_be_inferred_from_unrelated_mentions(self):
        self.assertIsNone(_property_evidence(
            "NetBSD is a platform. The Linux kernel belongs to Linux.",
            "NetBSD","kernel"))

class V39SandboxRecovery(unittest.TestCase):
    def test_retries_one_transient_sandbox_failure_then_checks_real_stdout(self):
        calls=[]
        def execute(code):
            calls.append(code)
            return ({"status":"FAILED","error_code":"docker_unavailable"}
                    if len(calls)==1 else
                    {"status":"PASSED","exit_code":0,"output":"450\n"})
        q="Value?\n'''python\nn=23\nm=7\nk=15\nprint((n+m)*k)\n'''"
        r=verified_python_question(q,execute)
        self.assertEqual(len(calls),2)
        self.assertTrue(r["tool_verified"])
        self.assertEqual(r["text"],"Ergebnis: 450")

    def test_does_not_retry_invalid_program_or_forge_stdout(self):
        calls=[]
        def execute(code):
            calls.append(code)
            return {"status":"FAILED","error_code":"invalid_python_code"}
        r=verified_python_question("'' 'python invalid",execute)
        self.assertIsNone(r)
        q="'' 'python invalid"
        code="'''python\nprint(1)\n'''"
        fail=verified_python_question(code,execute)
        self.assertFalse(fail["tool_verified"])
        self.assertEqual(len(calls),1)
        self.assertEqual(fail["error_code"],"sandbox_error")

if __name__=="__main__":
    unittest.main()
