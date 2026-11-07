"""Novel-math 120 A/B contract: unseen prompt generators, local-only raw answers."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v5_ood_math_eval import (
    CASES, CATEGORIES, case_hash, make_case, report, combine, BASE_REVISION,
    verify_novel_prompts
)
from scripts.neral_v5_style_eval import CASES as OLD_CASES, grade
from scripts.neral_v5_transfer_curriculum import build


class NovelMath120Tests(unittest.TestCase):
    def test_six_new_families_and_all_labels_recompute(self):
        self.assertEqual(len(CASES),120)
        self.assertEqual(len(CATEGORIES),6)
        self.assertEqual(len({row["id"] for row in CASES}),120)
        self.assertEqual(len({row["prompt"] for row in CASES}),120)
        self.assertEqual(len(case_hash()),64)
        for cat in CATEGORIES:
            self.assertEqual(sum(c["category"]==cat for c in CASES),20)
        old={c["prompt"] for c in OLD_CASES}
        self.assertFalse(old & {c["prompt"] for c in CASES})
        for c in CASES:
            self.assertEqual(grade(c,c["answer"]),(True,True),c["id"])
            self.assertEqual(grade(c,"Antwort: "+c["answer"]),(False,False),c["id"])
        for category in CATEGORIES:
            for i in range(20):
                self.assertEqual(make_case(category,i),
                                 next(c for c in CASES if c["id"]==f"{category}-{i:02d}"))
        with self.assertRaises(ValueError):
            make_case("arbitrary",0)
        with self.assertRaises(ValueError):
            make_case("money",20)

    def test_isolated_from_all_v5_splits(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"training"
            build(source)
            self.assertTrue(verify_novel_prompts(source))
            for split in ("train","val","test"):
                old={json.loads(line)["prompt"][1]["content"]
                     for line in (source/(split+".jsonl")).read_text(encoding="utf8").splitlines()}
                self.assertFalse(old & {c["prompt"] for c in CASES})
            broken=list(CASES)
            original=next(iter(old))
            broken[0]={**broken[0],"prompt":original}
            with self.assertRaises(ValueError):
                verify_novel_prompts(source,broken)

    def test_paired_score_and_no_independence_overclaim(self):
        def rows(flags):
            return [{"id":c["id"],"correct":bool(v),"format_valid":True}
                    for c,v in zip(CASES,flags)]
        b=[True]*30+[False]*90
        n=[True]*29+[False]+[True]*2+[False]*88
        r=report(rows(b),rows(n),adapter_sha="a"*64,base_revision=BASE_REVISION)
        self.assertEqual(r["base"]["correct"],30)
        self.assertEqual(r["neral_v5"]["correct"],31)
        self.assertEqual(r["paired"]["improvements"],2)
        self.assertEqual(r["paired"]["regressions"],1)
        self.assertFalse(r["independent_external_benchmark"])
        self.assertFalse(r["general_improvement_proven"])
        self.assertFalse(r["release_validated"])

    def test_combine_requires_sealed_matching_ids_and_never_exports_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for variant in ("base","neral_v5"):
                folder=root/variant
                folder.mkdir()
                scores=[]
                for case in CASES:
                    raw=case["answer"]
                    scores.append({"id":case["id"],"correct":True,
                                   "format_valid":True,
                                   "answer_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                                   "local_raw":raw})
                data={"variant":variant,"adapter_sha256":"a"*64,
                      "base_revision":BASE_REVISION,
                      "case_hash":case_hash(),"scores":scores}
                (folder/"local.json").write_text(json.dumps(data),encoding="utf8")
            out=combine(root,adapter_sha="a"*64,base_revision=BASE_REVISION)
            self.assertEqual(out["base"]["correct"],120)
            payload=json.dumps(out)
            for secret in ('"local_raw"','"answer_sha256"','"prompt"'):
                self.assertNotIn(secret,payload)
            corrupted=root/"base"/"local.json"
            data=json.loads(corrupted.read_text())
            data["case_hash"]="0"*64
            corrupted.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                combine(root,adapter_sha="a"*64,base_revision=BASE_REVISION)

    def test_gpu_workflow_automatic_readonly_bounded(self):
        p=Path(".github/workflows/neral-v5-ood-math-120.yml")
        w=p.read_text(encoding="utf8")
        for value in ("push:","branches: [main]","self-hosted, windows, x64, neral-gpu",
                      "neral-qwen7b-r32-gpu-exclusive","42378021782",
                      "neral_v5_ood_math_eval","--variant $variant","--combine",
                      "upload-artifact", "if-no-files-found: error"):
            self.assertIn(value,w)
        for unsafe in ("--run-local-gpu","trainer.train","schedule:","git push","curl ","Invoke-WebRequest"):
            self.assertNotIn(unsafe,w)


if __name__=="__main__":
    unittest.main()
