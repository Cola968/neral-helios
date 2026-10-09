from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from frontier350.audit_neraltalk350_quality import audit as audit_chat,gates
from frontier350.audit_preferences350 import audit as audit_pref

class QualityAuditTests(unittest.TestCase):
    def test_chat_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val","test"):
                rows=[]
                n=12 if split=="train" else 2
                for i in range(n):
                    rows.append({"id":f"{split}-{i}","language":"de" if i%2==0 else "en","domain":["chat","math","code","reasoning","science"][i%5],
                                 "source":"oasst2-human-multiturn" if i<4 else "neral-generated","license":"x",
                                 "messages":[{"role":"user","content":f"frage {split} {i}"},{"role":"assistant","content":f"antwort {split} {i}"}] + ([{"role":"user","content":"weiter"},{"role":"assistant","content":f"mehr {i}"}] if i<2 else [])})
                (root/f"{split}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
            r=audit_chat(root)
            self.assertEqual(r["splits"]["train"]["rows"],12)
            self.assertGreater(r["splits"]["train"]["multiturn_rows"],0)
    def test_pref_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val","test"):
                n=100 if split=="train" else 2;rows=[]
                for i in range(n):
                    rows.append({"id":f"{split}-{i}","pair_sha256":f"{split}-{i}","language":"de" if i%2==0 else "en","context":[{"role":"user","content":"q"}],
                                 "chosen":"good answer","rejected":"bad answer","chosen_rank":0,"rejected_rank":1})
                (root/f"{split}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
            r=audit_pref(root);self.assertTrue(r["gates"]["pass"])
if __name__=="__main__":unittest.main()
