"""Read-only tokenizer inventory; never outputs prompt text or holdout answers."""
from __future__ import annotations
import json
from pathlib import Path
from scripts.neral_1_1_postverify_dev24_cases import build

def inspect_tokens(value):
    kind=type(value).__name__
    if isinstance(value,dict):
        value=value.get("input_ids")
    if not isinstance(value,list):
        return {"type":kind,"count":len(value) if hasattr(value,"__len__") else -1}
    return {"type":kind,"count":len(value),
            "first_type":type(value[0]).__name__ if value else None,
            "first_nested_length":len(value[0]) if value and isinstance(value[0],list) else None}

def main():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained("D:/neral-qlora/checkpoint",
                                  local_files_only=True,use_fast=True)
    cases=build()
    rows=[]
    for ix,c in enumerate(cases[:2]):
        for label,system in (("short","Answer with an integer."),
                             ("full","Answer the user's task with only the exact nonnegative integer. Do not guess; no explanations.")):
            turns=[{"role":"system","content":system},{"role":"user","content":c["question"]}]
            val=t.apply_chat_template(turns,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            raw=t.apply_chat_template(turns,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            direct=t(raw,add_special_tokens=False)["input_ids"]
            rows.append({"index":ix,"template":label,
                         "tokenized":inspect_tokens(val),
                         "formatted_chars":len(raw),
                         "retokenized":inspect_tokens(direct),
                         "user_direct_tokens":len(t(c["question"],add_special_tokens=False)["input_ids"])})
    print("NERAL_QWEN35_TOKEN_DIAGNOSTIC "+json.dumps(rows,sort_keys=True),flush=True)

if __name__=="__main__":main()
