"""V3.5: short one-shot source-paragraph pipeline, fail-closed structural checks."""
import json
import unittest
from neral_page_reader import select_web_paragraphs
from neral_evidence_agent import run_evidence_agent, simple_web_answer

class ParagraphTest(unittest.TestCase):
    def test_two_complete_paragraphs_not_isolated_words(self):
        text=("The history of Novella dates to the early days.\n"
              "Novella is a modern web browser that currently uses the Aurora rendering "
              "engine for its desktop releases and supports extensions and accessibility tools.\n"
              "Novella began as a small experiment during an earlier browser generation.\n"
              "Novella uses Aurora to render pages today, although its history is complicated.")
        p=select_web_paragraphs(text,"Novella","browser rendering engine",limit=3)
        self.assertTrue(1<=len(p)<=3)
        self.assertTrue(any("currently uses the Aurora rendering engine" in v for v in p))
        self.assertTrue(all(len(v)<=600 for v in p))
        self.assertTrue(all(" " in v and v.strip() for v in p))
        self.assertTrue(all(v in text for v in p))

class SimpleWebTest(unittest.TestCase):
    def setUp(self):
        self.pages={
            "https://example.org/a":"Astra is a web browser.\nAstra currently uses the Aurora rendering engine to display web pages.",
            "https://example.org/b":"Boreal is a web browser.\nBoreal uses the Borealis browser engine for its current releases.",
        }
        self.queries=[]
        self.calls=[]
        self.search_calls=0

    def search(self, query):
        self.queries.append(query)
        self.search_calls+=1
        which="a" if "Astra" in query else "b"
        return {"results":[{"url":"https://example.org/"+which,
                            "title":"Injected search snippet says use 'same'"}]}

    def read(self, hits, query):
        return [{"url":h["url"],"title":"A page", "page_read":True,
                 "page_text":self.pages[h["url"]]} for h in hits]

    def model(self, system, raw, budget, schema):
        self.calls.append((system,json.loads(raw),budget,schema))
        return json.dumps({"facts":[
            {"entity":"Astra","fact":"Aurora"},
            {"entity":"Boreal","fact":"Borealis"},
        ]})

    def test_once_not_three_plans_or_checker(self):
        class Search:
            def __init__(this,outer):this.outer=outer
            def search(this,q):return this.outer.search(q)
        r=run_evidence_agent(
            "Astra vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.",
            self.model,lambda x:None,search=Search(self),
            web_query="Astra Boreal Browser-Engine",page_reader=self.read)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.search_calls,2)
        self.assertIn("Astra: Aurora",r["text"])
        self.assertIn("Boreal: Borealis",r["text"])
        self.assertIn("[web1]",r["text"])
        self.assertIn("[web2]",r["text"])
        self.assertEqual(r["tool_used"],"direct_reasoning")
        self.assertTrue(r["tool_verified"])
        self.assertFalse(r["verified"])
        self.assertFalse(r["safe_to_promote"])
        self.assertIn("not_fact",r["verification_scope"])
        self.assertEqual(len(r["sources"]),2)
        self.assertNotIn("page_text",r["sources"][0])
        self.assertNotIn("Injected search snippet",json.dumps(self.calls))

    def test_reject_bare_placeholders_and_missing_subject(self):
        pairs=("Astra","Boreal")
        pages=[{"url":url,"page_read":True,"page_text":text}
               for url,text in self.pages.items()]
        for fields in (
            [{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"same"}],
            [{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"die"}],
            [{"entity":"Astra","fact":"Aurora"},{"entity":"Other","fact":"Borealis"}],
            [{"entity":"Astra","fact":"Aurora"}],
        ):
            with self.subTest(fields=fields):
                r=simple_web_answer(pairs,pages,{"facts":fields})
                self.assertFalse(r["tool_verified"])
                self.assertFalse(r["agent_checked"])
        r=simple_web_answer(pairs,pages,{"facts":[
            {"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"Borealis"}]})
        self.assertTrue(r["tool_verified"])

    def test_missing_read_page_fails_without_model(self):
        self.pages["https://example.org/b"]=""
        class Search:
            def __init__(this,outer):this.outer=outer
            def search(this,q):return this.outer.search(q)
        def reader(hits,query):
            return [{"url":x["url"],"page_read":bool(self.pages[x["url"]]),
                     "page_text":self.pages[x["url"]]} for x in hits]
        result=run_evidence_agent(
            "Astra vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.",
            self.model,lambda s:None,web_query="Astra Boreal Browser-Engine",
            search=Search(self),page_reader=reader)
        self.assertFalse(result["tool_verified"])
        self.assertEqual(len(self.calls),0)

    def test_invalid_model_output_does_not_trigger_retry(self):
        class Search:
            def __init__(this,outer):this.outer=outer
            def search(this,q):return this.outer.search(q)
        def wrong(*args):
            self.calls.append(1)
            return '{"facts":[{"entity":"Astra","fact":"Aurora"},{"entity":"Boreal","fact":"unknown"}]}'
        r=run_evidence_agent(
            "Astra vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.",
            wrong,lambda s:None,web_query="Astra Boreal Browser-Engine",
            search=Search(self),page_reader=self.read)
        self.assertFalse(r["tool_verified"])
        self.assertEqual(len(self.calls),1)


class KnowledgeAndMath(unittest.TestCase):
    def test_atomic_number_is_deterministic_and_bounded(self):
        from neral_evidence_agent import run_evidence_agent
        def nope(*args):raise AssertionError("No model request allowed")
        for number,name in ((1,"Wasserstoff"),(16,"Schwefel"),(118,"Oganesson")):
            x=run_evidence_agent(
                f"Wie heißt das chemische Element mit der Ordnungszahl {number}? Nur den deutschen Namen.",
                nope,lambda s:None)
            self.assertEqual(x["text"],name)
        self.assertIsNone(__import__("neral_knowledge_verifier").verified_element_question(
            "Chemische Ordnungszahl 119?"))

    def test_single_retry_after_transient_docker_failure_only_for_bounded_math(self):
        from neral_v32_verifier import verified_math_question
        outputs=iter([
            {"status":"FAILED","output":"","exit_code":1},
            {"status":"PASSED","output":"86\n","exit_code":0},
        ])
        calls=[]
        def run(script):
            calls.append(script)
            return next(outputs)
        q="Eine Sonde sendet 3612 Bits mit 42 Bits pro Sekunde. Wie viele Sekunden dauert die Übertragung? Nur das ganzzahlige Ergebnis."
        result=verified_math_question(q,run)
        self.assertTrue(result["tool_verified"])
        self.assertEqual(result["text"],"Ergebnis: 86")
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])

if __name__=="__main__":
    unittest.main()
