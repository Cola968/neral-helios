from __future__ import annotations
import tempfile,unittest
from pathlib import Path

from frontier350.oasst2_human350 import indexes,path_to_root,valid_human_path,split_for_root_prompt
from frontier350.chat350_preference_format import encode_final_response
from frontier350.build_oasst2_multiturn350 import extract as extract_multi
from frontier350.build_oasst2_preferences350 import extract_preferences

class E:
    def __init__(self,ids):self.ids=ids
class T:
    def token_to_id(self,x):return {"<|bos|>":0,"<|eos|>":1}.get(x)
    def encode(self,x):return E([10+(ord(c)%200) for c in x])

def row(mid,pid,role,text,rank=None):
    return {"message_id":mid,"parent_id":pid,"role":role,"text":text,"lang":"de","deleted":False,"synthetic":False,"review_result":True,"model_name":None,"rank":rank}

class HumanAlignmentTests(unittest.TestCase):
    def test_path_and_roles(self):
        rs=[row("u0",None,"prompter","Hallo"),row("a0","u0","assistant","Hi",0),row("u1","a0","prompter","Nochmal"),row("a1","u1","assistant","Gern",0)]
        by,_=indexes(rs);p=path_to_root("a1",by)
        self.assertTrue(valid_human_path(p,end_role="assistant"));self.assertEqual(len(p),4)

    def test_split_stable(self):
        self.assertEqual(split_for_root_prompt("  Hallo Welt "),split_for_root_prompt("hallo   welt"))

    def test_final_response_only_supervision(self):
        e=encode_final_response(T(),[{"role":"user","content":"A"},{"role":"assistant","content":"B"},{"role":"user","content":"C"}],"D",max_seq_len=128)
        self.assertEqual(e.targets[-1],1);self.assertGreater(e.response_tokens,1);self.assertIn(-100,e.targets)

    def test_synthetic_path_rejected(self):
        rs=[row("u0",None,"prompter","Hallo"),row("a0","u0","assistant","Hi",0)]
        rs[1]["synthetic"]=True
        by,_=indexes(rs)
        self.assertFalse(valid_human_path(path_to_root("a0",by),end_role="assistant"))

if __name__=="__main__":unittest.main()
