from scripts.neral_v2_luna_encoding import assistant_only_4b
import pytest

class FourBTemplate:
    # Intentional mismatch between the generation prompt and real assistant:
    # 4B appends a /final suffix only when content is provided.
    def apply_chat_template(self,turns,*,tokenize,add_generation_prompt,enable_thinking):
        template="<|im_start|>system\\n"+turns[0]["content"]+"<|im_end|>"
        template+="<|im_start|>user\\n"+turns[1]["content"]+"<|im_end|>"
        template+="<|im_start|>assistant"
        template+=("/generation" if add_generation_prompt else "/final")
        if not add_generation_prompt:
            template+="\\n"+turns[-1]["content"]+"<|im_end|>"
        if tokenize:return self(template,add_special_tokens=False,return_offsets_mapping=False)["input_ids"]
        return template
    def __call__(self,text,*,add_special_tokens,return_offsets_mapping):
        data={"input_ids":[ord(c) for c in text]}
        if return_offsets_mapping:data["offset_mapping"]=[(i,i+1) for i in range(len(text))]
        return data

def turns(answer="Das ist 42"):
    return [{"messages":[{"role":"system","content":"Kurz."},{"role":"user","content":"Wie viel?"},{"role":"assistant","content":answer}]}]

def test_qwen4b_different_native_header_no_mismatch_and_no_prompt_loss():
    t=FourBTemplate()
    assert t.apply_chat_template(turns()[0]["messages"][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False).endswith("/generation")
    encoded=assistant_only_4b(turns(),t,max_seq_length=512)[0]
    actual="".join(chr(x) for x in encoded["input_ids"])
    first=next(i for i,v in enumerate(encoded["labels"]) if v!=-100)
    assert actual[first:]=="Das ist 42<|im_end|>"
    assert all(x==-100 for x in encoded["labels"][:first])

def test_luna_safety_rejects_oversize_and_wrong_roles():
    t=FourBTemplate()
    with pytest.raises(RuntimeError,match="context cap"):
        assistant_only_4b(turns("A"*500),t,max_seq_length=512)
    bad=turns();bad[0]["messages"][2]["role"]="user"
    with pytest.raises(ValueError):
        assistant_only_4b(bad,t,max_seq_length=512)
