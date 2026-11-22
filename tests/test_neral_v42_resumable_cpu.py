"""CPU contracts for resumable, privacy-preserving bounded owner evaluation."""
import json
import tempfile
import unittest
from collections import Counter
from hashlib import sha256
from pathlib import Path

from scripts.neral_v42_resumable import (
    TIMEOUTS, balanced_sample, load_checkpoint, parse_vram_mb,
    make_record, select_batch,
)

class ResumableBlindCPU(unittest.TestCase):
    def setUp(self):
        self.cases = []
        counts = {"everyday":35, "multiturn":40, "live_web":35,
                  "math_logic":30, "programming":25,"writing":20,"uncertainty":15}
        for cat, n in counts.items():
            for k in range(n):
                self.cases.append({"id":f"{cat}-{k:03d}", "category":cat,
                                   "prompt":f"Unique private task {cat} {k}"})

    def test_balanced_50_covers_every_category_and_has_no_duplicates(self):
        chosen = balanced_sample(self.cases, 50)
        self.assertEqual(len(chosen), 50)
        self.assertEqual(len({c["id"] for c in chosen}), 50)
        self.assertEqual(set(Counter(c["category"] for c in chosen)), {
            "everyday", "multiturn", "live_web", "math_logic",
            "programming", "writing", "uncertainty"
        })
        self.assertEqual([c["id"] for c in chosen], [c["id"] for c in balanced_sample(self.cases,50)])

    def test_disjoint_batches_cover_all_200_cases(self):
        batches = [select_batch(self.cases, index=i, count=50) for i in range(4)]
        all_ids = [x["id"] for group in batches for x in group]
        self.assertEqual(len(all_ids),200)
        self.assertEqual(len(set(all_ids)),200)

    def test_resume_uses_pair_and_prompt_hash_not_just_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"details.jsonl"
            target=self.cases[0]
            digest=sha256(target["prompt"].encode()).hexdigest()
            row={"id":target["id"], "tier":"Quartz", "prompt_sha256":digest,
                 "grade":"fail", "latency_s":20.3}
            path.write_text(json.dumps(row)+"\n",encoding="utf-8")
            loaded=load_checkpoint(path,self.cases,("Quartz",))
            self.assertEqual(len(loaded),1)
            self.assertEqual(loaded[0]["id"],target["id"])
            self.assertEqual(load_checkpoint(path,self.cases,("Amber",)),[])  # other tier not reused
            row["prompt_sha256"]="bad"
            path.write_text(json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(path,self.cases,("Quartz",))

    def test_duplicate_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"details.jsonl"
            item=self.cases[0]
            row={"id":item["id"],"tier":"Amber",
                 "prompt_sha256":sha256(item["prompt"].encode()).hexdigest(),
                 "grade":"pass"}
            path.write_text((json.dumps(row)+"\n")*2,encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(path,self.cases,("Amber",))

    def test_deadlines_and_vram_parse(self):
        self.assertEqual(TIMEOUTS,{"Amber":30,"Quartz":45,"Obsidian":75})
        self.assertEqual(parse_vram_mb("4096 MiB\n"),4096)
        self.assertIsNone(parse_vram_mb("N/A"))
        self.assertIsNone(parse_vram_mb(""))

    def test_checkpoint_record_has_requested_metadata(self):
        case=self.cases[0]
        record=make_record(case,"Obsidian",{"text":"private","tool_used":"z3"},
                           "pass","verified",42.125,8192,None)
        for key in ("task_id","tier","score","latency_ms","vram_peak_mb","error"):
            self.assertIn(key,record)
        self.assertEqual(record["latency_ms"],42125)
        self.assertEqual(record["vram_peak_mb"],8192)
        self.assertEqual(record["task_id"],case["id"])

if __name__=="__main__":
    unittest.main()
