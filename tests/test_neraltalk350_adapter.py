from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from frontier350.build_neraltalk350_from_screened import convert

class AdapterTests(unittest.TestCase):
    def test_convert(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/"old"; out=root/"new"; src.mkdir()
            source={"id":"oasst2-human-de","license":"Apache-2.0","provenance_url":"https://example.invalid","sha256":"a"*64}
            (src/"manifest.json").write_text(json.dumps({"version":1,"external_model_calls":False,"sources":[source],"prompt_group_overlap_audit":{"cross_split_prompt_group_overlap":0}}),encoding="utf8")
            for split,i in [("train","1"),("val","2"),("test","3")]:
                row={"id":i,"source_id":"oasst2-human-de","language":"de","domain":"chat","prompt":"Frage "+i,"answer":"Antwort "+i}
                (src/f"{split}.jsonl").write_text(json.dumps(row)+"\n",encoding="utf8")
            p=convert(src,out)
            self.assertFalse(p["holdout_imported"])
            self.assertEqual(p["output"]["train"]["rows"],1)
            self.assertTrue((out/"provenance.json").is_file())

    def test_unknown_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/"old"; out=root/"new"; src.mkdir()
            source={"id":"ok","license":"Apache-2.0","provenance_url":"x","sha256":"a"*64}
            (src/"manifest.json").write_text(json.dumps({"version":1,"external_model_calls":False,"sources":[source]}),encoding="utf8")
            for split in ("train","val","test"):
                sid="bad" if split=="train" else "ok"
                row={"id":split,"source_id":sid,"language":"de","domain":"chat","prompt":"P "+split,"answer":"A "+split}
                (src/f"{split}.jsonl").write_text(json.dumps(row)+"\n",encoding="utf8")
            with self.assertRaises(ValueError):
                convert(src,out)

if __name__=="__main__":
    unittest.main()
