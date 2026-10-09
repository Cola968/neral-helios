from __future__ import annotations
import unittest
from frontier350.chat_prompt350 import render_history,trim_history
from frontier350.release_check350 import check

class E:
    def __init__(self,ids): self.ids=ids

class T:
    def token_to_id(self,x): return {"<|bos|>":0}.get(x)
    def encode(self,x): return E(list(range(len(x))))

class RuntimeTests(unittest.TestCase):
    def test_render(self):
        self.assertEqual(
            render_history([("user","hi"),("assistant","hello"),("user","again")]),
            "User: hi\nAssistant: hello\nUser: again\nAssistant:"
        )

    def test_trim_preserves_latest_user(self):
        h=[("user","a"),("assistant","b"),("user","c")]
        t=trim_history(T(),h,32)
        self.assertEqual(t[-1],("user","c"))

    def test_release_never_auto(self):
        q={"gates":{"pass":True}}
        f={"decision":"manual_review_required"}
        cap={"comparison":{"capability_no_regression":True}}
        sl={"slice_no_regression":True}
        r=check(q,f,cap,sl)
        self.assertTrue(r["machine_gates_pass"])
        self.assertFalse(r["automatic_release"])

if __name__=="__main__":
    unittest.main()
