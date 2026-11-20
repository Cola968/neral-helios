"""V3.6 regression: native HF JSON budget and source-bounded diagnostics."""
import unittest
from neral_evidence_agent import run_evidence_agent, normalize_web_synthesis_json, parse_web_synthesis_json
from neral_knowledge_verifier import normalize_german_capital_name

class JsonParser(unittest.TestCase):
    def test_fences_and_preamble_and_aliases(self):
        raw='Result:\n'+chr(96)*3+'json\n{"answers":[{"name":"Astra","value":"Aurora"},{"name":"Boreal","value":"Borealis"}]}\n'+chr(96)*3
        self.assertEqual(parse_web_synthesis_json(raw,("Astra","Boreal")),{
            "facts":[{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"Borealis"}]})
        raw='text { "Astra": "Aurora", "Boreal": "Borealis" } EOF'
        self.assertEqual(parse_web_synthesis_json(raw,("Astra","Boreal"))["facts"][1]["fact"],"Borealis")
        x=normalize_web_synthesis_json({"results":[
            {"subject":"Astra","property":"Aurora"},
            {"subject":"Boreal","property":"Borealis"}]},("Astra","Boreal"))
        self.assertEqual(x["facts"],[
            {"entity":"Astra","fact":"Aurora"},
            {"entity":"Boreal","fact":"Borealis"}])

    def test_incomplete_or_fabricated_values_fail_closed(self):
        for raw in ('{"facts":[{"entity":"Astra","fact":"Aurora"}',
                    '{"facts":[{"entity":"Astra","fact":"Aurora"}],"fact":"oops"}',
                    '{"facts":[{"entity":"Astra","fact":"same"},{"entity":"Boreal","fact":"Borealis"}]}',
                    'Astra: Aurora, Boreal: Borealis',
                    '{"facts":[{"entity":"Astra","fact":"Aurora"},{"entity":"Fake","fact":"Borealis"}]}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_web_synthesis_json(raw,("Astra","Boreal"))

class OneShotModel(unittest.TestCase):
    def run_response(self,model):
        sources={
            "Astra":{"url":"https://example.org/a","page_read":True,
                     "page_text":"Astra is a web browser. Astra currently uses the Aurora rendering engine."},
            "Boreal":{"url":"https://example.org/b","page_read":True,
                      "page_text":"Boreal is a web browser. Boreal uses the Borealis browser engine today."}
        }
        class Search:
            def search(inner,q):
                entity="Astra" if "Astra" in q else "Boreal"
                return {"results":[{"url":sources[entity]["url"]}]}
        def reader(hits,q):
            entity="Astra" if "/a" in hits[0]["url"] else "Boreal"
            return [sources[entity]]
        return run_evidence_agent(
            "Astra vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.",
            model,lambda x:None,
            web_query="Astra Boreal Browser-Engine",search=Search(),page_reader=reader)

    def test_native_hf_supported_768_budget_once(self):
        called=[]
        def model(system,payload,budget,schema):
            called.append((budget,schema))
            return '{"facts":[{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"Borealis"}]}'
        out=self.run_response(model)
        self.assertEqual(len(called),1)
        self.assertEqual(called[0][0],768)
        self.assertTrue(out["tool_verified"])
        self.assertFalse(out["verified"])

    def test_raw_diagnostic_truncates_and_logs_exception(self):
        calls=[]
        raw="Intro\n"+'{"facts":[{"entity":"Astra","fact":'+("x"*1200)
        def model(*args):
            calls.append(1)
            return raw
        out=self.run_response(model)
        self.assertEqual(len(calls),1)
        self.assertEqual(out["verification_status"],"model_invalid_json")
        diagnostic=out["web_model_diagnostic"]
        self.assertTrue(0<len(diagnostic["raw_output"])<=800)
        self.assertTrue(diagnostic["exception"].startswith(("ValueError:","AgentError:")))
        self.assertFalse(out["tool_verified"])

    def test_backend_exception_is_not_misreported_as_json_parse(self):
        def model(*args):
            raise ValueError("Unapproved NERAL inference budget")
        out=self.run_response(model)
        self.assertIn("Unapproved NERAL inference budget",out["web_model_diagnostic"]["exception"])
        self.assertEqual(out["web_model_diagnostic"]["raw_output"],"")

class GermanNames(unittest.TestCase):
    def test_explicit_german_capital_name(self):
        q="Wie heißt die Hauptstadt von Griechenland? Nur den deutschen Namen."
        self.assertEqual(normalize_german_capital_name(q,"Athens"),"Athen")
        self.assertEqual(normalize_german_capital_name(q,"Athen"),"Athen")
        self.assertEqual(normalize_german_capital_name(q,"Antwort: Athens"),"Antwort: Athen")
        self.assertEqual(normalize_german_capital_name("Explain Athens","Athens"),"Athens")
        self.assertEqual(normalize_german_capital_name(q,"Athens ist eine schöne Stadt."),"Athens ist eine schöne Stadt.")

if __name__=="__main__":
    unittest.main()
