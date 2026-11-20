"""V3.7 red-first web source selection: no gold facts, no homepage-only proof."""
import json
import unittest

from neral_evidence_agent import run_evidence_agent


class SourceSelectionContracts(unittest.TestCase):
    @staticmethod
    def _run(pages, hits_for, *, fact_json=None):
        search_calls=[]
        model_calls=[]
        class Search:
            def search(self, query):
                search_calls.append(query)
                return {"results":[{"url":url} for url in hits_for(query)]}
        def reader(hits, query):
            return [{"url":hit["url"],"page_read":True,
                     "page_text":pages[hit["url"]]} for hit in hits]
        def model(system,payload,budget,schema):
            model_calls.append((system,payload))
            return fact_json or json.dumps({"facts":[
                {"entity":"Astra","fact":"unknown"},
                {"entity":"Boreal","fact":"unknown"},
            ]})
        response=run_evidence_agent(
            "Astra vs Boreal: Vergleiche Browser-Engine anhand tatsächlich gelesener Webseiten.",
            model,lambda x:None,search=Search(),
            web_query="Astra Boreal Browser-Engine",page_reader=reader)
        return response,search_calls,model_calls

    def test_skip_marketing_homepages_and_use_proven_deeper_pages(self):
        pages={
            "https://example.org/astra-home":"Astra is a fast browser. Download Astra today.",
            "https://example.org/astra-ref":"Astra is a browser. Astra currently uses the Aurora rendering engine for pages.",
            "https://example.org/boreal-home":"Boreal is a useful browser. Download Boreal today.",
            "https://example.org/boreal-ref":"Boreal is a browser. Boreal currently uses the Borealis rendering engine for pages.",
        }
        def hits(query):
            key="astra" if "Astra" in query else "boreal"
            return [f"https://example.org/{key}-home",
                    f"https://example.org/{key}-ref"]
        response,queries,models=self._run(pages,hits)
        self.assertTrue(response["tool_verified"])
        self.assertTrue(response["agent_checked"])
        self.assertIn("Astra: Aurora",response["text"])
        self.assertIn("Boreal: Borealis",response["text"])
        self.assertEqual([p["url"] for p in response["sources"]],
                         ["https://example.org/astra-ref",
                          "https://example.org/boreal-ref"])
        self.assertIn("[web1]",response["text"])
        self.assertIn("[web2]",response["text"])
        self.assertTrue(all("example.org/" in p["url"] and p["page_read"]
                            for p in response["sources"]))

    def test_retry_different_search_query_when_first_page_lacks_property(self):
        pages={
            "https://example.org/astra-home":"Astra is a fast browser. Download Astra today.",
            "https://example.org/astra-ref":"Astra currently uses the Aurora rendering engine. Astra is a browser.",
            "https://example.org/boreal-ref":"Boreal currently uses the Borealis rendering engine. Boreal is a browser.",
        }
        calls={"Astra":0,"Boreal":0}
        def hits(query):
            entity="Astra" if "Astra" in query else "Boreal"
            calls[entity]+=1
            if entity=="Astra" and calls[entity]==1:
                return ["https://example.org/astra-home"]
            return ["https://example.org/astra-ref" if entity=="Astra"
                    else "https://example.org/boreal-ref"]
        response,queries,_=self._run(pages,hits)
        self.assertTrue(response["tool_verified"])
        self.assertGreaterEqual(calls["Astra"],2)
        self.assertEqual(len(response["sources"]),2)

    def test_no_proof_from_one_entity_never_mints_a_second_citation(self):
        pages={
            "https://example.org/astra-ref":"Astra currently uses the Aurora rendering engine. Astra is a browser.",
            "https://example.org/boreal-home":"Boreal is a browser. Click to download Boreal today.",
        }
        def hits(q):
            return ["https://example.org/astra-ref"] if "Astra" in q else [
                "https://example.org/boreal-home"]
        response,queries,models=self._run(pages,hits,fact_json=json.dumps({
            "facts":[{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"Borealis"}]}))
        self.assertFalse(response["tool_verified"])
        self.assertFalse(response["agent_checked"])
        self.assertNotIn("Boreal: Borealis",response["text"])
        self.assertNotIn("[web2]",response["text"])
        self.assertNotEqual(response["error_code"],"no_tool_called")
        self.assertTrue(queries)

    def test_read_page_not_snippet_or_title_is_the_fact_source(self):
        pages={
            "https://example.org/astra":"Astra is a browser. Download Astra today.",
            "https://example.org/boreal":"Boreal is a browser. Download Boreal today.",
        }
        def hits(q):
            return ["https://example.org/astra"] if "Astra" in q else [
                "https://example.org/boreal"]
        response,_,_=self._run(pages,hits)
        self.assertFalse(response["tool_verified"])


if __name__=="__main__":
    unittest.main()
