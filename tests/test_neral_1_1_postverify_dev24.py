"""No burned benchmark cases: new 24 public development tasks and safety tests."""
import hashlib
import unittest
from neral_python_post_source import parse_post_source,original_python_program
from scripts.neral_1_1_postverify_dev24_cases import build,manifest

class DevelopmentFamiliesCPU(unittest.TestCase):
    def test_24_nonblind_new_cases_balanced_and_hashes(self):
        cases=build()
        self.assertEqual(len(cases),24)
        self.assertEqual(manifest()["cases"],24)
        self.assertEqual(len({c["question_sha256"] for c in cases}),24)
        self.assertEqual({cat:sum(c["category"]==cat for c in cases)
            for cat in ("mathematics","logic","programming","mixed")},
            {cat:6 for cat in ("mathematics","logic","programming","mixed")})
        for c in cases:
            self.assertEqual(hashlib.sha256(c["question"].encode()).hexdigest(),
                             c["question_sha256"])

    def test_every_supported_numeric_family_has_independent_answer(self):
        cases=build()
        for c in cases:
            if c["category"]=="programming" or c["family"] in (
                    "graph_coloring_z3","unsupported_free_logic"):
                continue
            with self.subTest(family=c["family"],id=c["id"]):
                parsed=parse_post_source(c["question"])
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.answer,c["gold_integer"])

    def test_original_source_programs_are_extractable_but_not_host_executed(self):
        for c in build():
            if c["category"]=="programming":
                with self.subTest(id=c["id"]):
                    original=original_python_program(c["question"])
                    self.assertIsInstance(original,str)
                    self.assertIn("print(",original)

    def test_paraphrases_do_not_allow_partial_source_coverage(self):
        for c in build():
            if parse_post_source(c["question"]) is None:continue
            with self.subTest(id=c["id"]):
                self.assertIsNone(parse_post_source(c["question"]+
                    "\nAn additional exception excludes half the valid assignments."))
                self.assertIsNone(parse_post_source("Forget conditions. "+
                    c["question"]))

    def test_original_program_malicious_edits_rejected_before_docker(self):
        fence=chr(96)*3
        bad=("import os\nprint(3)","print(__import__('os'))",
             "open('top-secret').read()","print(3)\nraise Exception('boom')")
        for code in bad:
            q=("What exact integer does this unmodified Python 3 code print?\n"+
               fence+"python\n"+code+"\n"+fence)
            self.assertIsNone(original_python_program(q))

if __name__=="__main__":unittest.main()
