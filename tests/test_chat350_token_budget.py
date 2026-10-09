from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from frontier350.validate_chat350_token_budget import audit_chat,audit_preferences

class Enc:
    def __init__(self,ids): self.ids=ids

class FakeTokenizer:
    ids={"<|bos|>":0,"<|eos|>":1}
    def token_to_id(self,x): return self.ids.get(x)
    def encode(self,text): return Enc([10]*len(str(text)))

def chat_row(rid,prompt,answer):
    return {"id":rid,"language":"de","domain":"chat","source":"unit","license":"CC0","prompt":prompt,"answer":answer}

def pref_row(rid,n):
    return {"id":rid,"language":"de","source":"unit","license":"CC0","context":[{"role":"user","content":"q"*n}],"chosen":"c"*n,"rejected":"r"*n,"chosen_rank":0,"rejected_rank":1,"pair_sha256":rid}

class TokenBudgetTests(unittest.TestCase):
    def test_chat_detects_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val","test"):
                rows=[chat_row(split+"-a","q-"+split,"a-"+split)]
                if split=="train":
                    rows.append(chat_row("long","q"*20,"a"*20))
                (root/f"{split}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
            r=audit_chat(FakeTokenizer(),root,max_seq=60,hard_cap=500)
            self.assertFalse(r["pass"])
            self.assertEqual(r["splits"]["train"]["violations"],1)

    def test_preferences_detects_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val","test"):
                rows=[pref_row(split+"-a",1)]
                if split=="train":
                    rows.append(pref_row("long",20))
                (root/f"{split}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
            r=audit_preferences(FakeTokenizer(),root,max_seq=35,hard_cap=500)
            self.assertFalse(r["pass"])
            self.assertEqual(r["splits"]["train"]["violations"],1)

    def test_clean_passes_and_reports_supervised_mix(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val","test"):
                (root/f"{split}.jsonl").write_text(
                    json.dumps(chat_row(split,"q-"+split,"a-"+split))+"\n"
                )
            r=audit_chat(FakeTokenizer(),root,max_seq=100,hard_cap=500)
            self.assertTrue(r["pass"])
            self.assertGreater(r["splits"]["train"]["supervised_tokens"],0)
            self.assertIn("de",r["splits"]["train"]["supervised_tokens_by_language"])

if __name__=="__main__":
    unittest.main()
