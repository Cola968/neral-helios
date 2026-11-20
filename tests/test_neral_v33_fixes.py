"""V3.3 red-first regression contracts from original V3.2 private JSONL failures."""
import unittest
from neral_v32_verifier import verified_math_question, property_grounded_web_answer


class ArithmeticRouting(unittest.TestCase):
    def test_arithmetic_series_is_verified_by_actual_docker_execution(self):
        q = "Berechne die Summe aller ganzen Zahlen von 38 bis einschließlich 78. Antworte mit der ganzen Zahl."
        code=[]
        def run(script):
            code.append(script)
            return {"status":"PASSED","output":"2378","exit_code":0}
        out=verified_math_question(q,run)
        self.assertEqual(out["text"],"Ergebnis: 2378")
        self.assertEqual(out["tool_used"],"python_sandbox")
        self.assertTrue(out["tool_verified"])
        self.assertEqual(len(code),1)
        self.assertIn("print(",code[0])
        self.assertFalse(verified_math_question(q,lambda s:{"status":"PASSED","output":"2377"})["tool_verified"])

    def test_rectangle_grid_counts_any_sizes_not_only_single_cells(self):
        q="Ein Rechteckgitter ist 7 Felder breit und 11 Felder hoch. Wie viele achsenparallele Rechtecke beliebiger Größe enthält das Gitter? Nur das ganzzahlige Ergebnis."
        r=verified_math_question(q,lambda s:{"status":"PASSED","output":"1848"})
        self.assertTrue(r["tool_verified"])
        self.assertIn("1848",r["text"])
        self.assertEqual(verified_math_question(q,lambda s:{"status":"PASSED","output":"7"})["error_code"],"checker_veto")

    def test_percentage_uses_decimal_cents_not_float(self):
        q="Ein Produkt kostet 120,00 Euro. Der Preis sinkt um 15 Prozent. Wie viel Euro kostet es danach?"
        calls=[]
        result=verified_math_question(q,lambda code:(calls.append(code) or {"status":"PASSED","output":"102.00"}))
        self.assertEqual(result["text"],"Ergebnis: 102,00")
        self.assertIn("Decimal",calls[0])

    def test_ambiguous_natural_language_never_guess(self):
        self.assertIsNone(verified_math_question("Rechne etwas mit 12 und 8",lambda s:None))
        self.assertIsNone(verified_math_question("Eine Zahl von 1 bis 10",lambda s:None))


class GroundedComparisons(unittest.TestCase):
    @staticmethod
    def sources():
        return [
          {"url":"https://example.org/a","page_read":True,
           "page_text":"Aster is a web browser. Aster uses the Aurora rendering engine to display pages."},
          {"url":"https://example.org/b","page_read":True,
           "page_text":"Boreal is a web browser. Boreal uses the Borealis browser engine to render websites."},
        ]

    def test_template_contains_grounded_fact_in_each_section(self):
        out=property_grounded_web_answer(("Aster","Boreal"),self.sources(),"Browser-Engine")
        self.assertIsNotNone(out)
        self.assertEqual(out["tool_used"],"direct_reasoning")
        self.assertTrue(out["tool_verified"])
        self.assertIn("Aster: Aurora",out["text"])
        self.assertIn("Boreal: Borealis",out["text"])
        self.assertIn("[web1]",out["text"])
        self.assertIn("[web2]",out["text"])
        self.assertEqual(len(out["sources"]),2)

    def test_rejects_generic_page_without_grounded_property(self):
        s=self.sources()
        s[1]["page_text"]="Boreal is a great browser. Download Boreal today."
        self.assertIsNone(property_grounded_web_answer(("Aster","Boreal"),s,"Browser-Engine"))
        self.assertIsNone(property_grounded_web_answer(("Aster","Boreal"),[
            {**self.sources()[0],"page_read":False},self.sources()[1]],"Browser-Engine"))

    def test_creator_extraction_only_when_literal_relation_proven(self):
        s=[
         {"url":"https://example.org/x","page_read":True,
          "page_text":"Quanta is a programming language. Quanta was created by Alice Smith in 2011."},
         {"url":"https://example.org/y","page_read":True,
          "page_text":"Vela is a programming language. Vela was created by Bruno Keller in 2017."},
        ]
        out=property_grounded_web_answer(("Quanta","Vela"),s,"Entwickler und Schöpfer")
        self.assertIn("Quanta: Alice Smith",out["text"])
        self.assertIn("Vela: Bruno Keller",out["text"])
        self.assertNotIn("2011",out["text"].split("Quanta:",1)[1].split("\n",1)[0])

    def test_page_reader_skips_generic_pages_until_property_is_found(self):
        from neral_page_reader import read_entity_pages
        from neral_v32_verifier import _property_evidence
        urls=[f"https://example.org/{i}" for i in range(5)]
        pages={url:("Aster is a fast browser." if i<4 else
                    "Aster uses the Aurora rendering engine to display web pages.")
               for i,url in enumerate(urls)}
        def reader(hits,query):
            return [{"url":hit["url"],"page_read":True,
                     "page_text":pages[hit["url"]]} for hit in hits]
        result=read_entity_pages([{"url":url} for url in urls],
            "Aster browser rendering engine","Aster",reader=reader,
            max_pages=1,predicate=lambda page: _property_evidence(
                page.get("page_text",""),"Aster","engine") is not None)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["url"],urls[4])

    def test_evidence_agent_short_circuit_uses_only_read_page_proofs(self):
        from neral_evidence_agent import run_evidence_agent
        class Search:
            calls=[]
            def search(self,term):
                self.calls.append(term)
                n="a" if "Aster" in term else "b"
                return {"results":[{"url":"https://example.org/"+n}]}
        def reader(hits,query):
            return [next(s for s in self.sources() if s["url"]==hits[0]["url"])]
        # Self not bound in function; use local fixture.
        sources=self.sources()
        def reader(hits,query):
            return [next(s for s in sources if s["url"]==hits[0]["url"])]
        def no_model(*args,**kwargs):
            raise AssertionError("Proof-bearing comparison must use literal page evidence")
        out=run_evidence_agent("Aster vs Boreal: Vergleiche Browser-Engine anhand gelesener Webseiten.",
                               no_model,lambda s:None,search=Search(),
                               web_query="Aster Boreal Browser-Engine",page_reader=reader)
        self.assertTrue(out["tool_verified"])
        self.assertIn("Aster: Aurora",out["text"])
        self.assertIn("Boreal: Borealis",out["text"])


if __name__=="__main__":
    unittest.main()
