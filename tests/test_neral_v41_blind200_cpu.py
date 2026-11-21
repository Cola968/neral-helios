"""CPU-only validation of the prospective sealed 200-case plan."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v41_blind200 import CATEGORIES, TOTAL, validate_suite


class Blind200PlanCPU(unittest.TestCase):
    def test_exact_seven_category_layout_without_live_call(self):
        self.assertEqual(TOTAL,200)
        self.assertEqual(len(CATEGORIES),7)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sealed.jsonl"
            cases=[]
            for cat,count in CATEGORIES.items():
                for i in range(count):
                    entry={"id":f"{cat}-{i:03d}","category":cat,
                           "prompt":f"Neue private Aufgabe {cat} Nummer {i:03d}?",
                           "history":([{"role":"user","content":"Welches Flugzeug?"},
                                       {"role":"assistant","content":"A321XLR."}]
                                      if cat=="multiturn" else []),
                           "grader":"blind_pairwise"}
                    if cat=="live_web":
                        entry["web_query"]=f"Recherche {cat} {i}"
                    cases.append(entry)
            path.write_text("".join(json.dumps(c)+"\n" for c in cases),encoding="utf-8")
            manifest=validate_suite(path)
            self.assertEqual(manifest["cases"],200)
            self.assertEqual(manifest["modes"],["Amber","Quartz","Obsidian"])
            cases[-1]["prompt"]=cases[-2]["prompt"]
            path.write_text("".join(json.dumps(c)+"\n" for c in cases),encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_suite(path)
