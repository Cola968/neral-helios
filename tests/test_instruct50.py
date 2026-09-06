import math

from tokenizer import ByteTokenizer
from train_instruct50 import encode_sft, lr_at


def test_sft_masks_prompt_and_supervises_answer_eos():
    tok=ByteTokenizer()
    row={"id":"x","prompt":"What is gravity?","answer":"Gravity attracts masses.","language":"en","domain":"science"}
    x,y=encode_sft(row,256)
    supervised=(y!=-100).nonzero().flatten()
    assert len(supervised)>1
    first=int(supervised[0])
    assert (y[:first]==-100).all()
    assert int(y[supervised[-1]])==tok.eos_id
    assert int((y==tok.eos_id).sum())==1
    assert int((y!=-100).sum())==len(tok.encode(" "+row["answer"]+"\n",eos=True))


def test_sft_rejects_silent_truncation():
    row={"id":"long","prompt":"Q"*200,"answer":"A"*200,"language":"en","domain":"chat"}
    try:
        encode_sft(row,64)
    except ValueError as exc:
        assert "exceeds seq" in str(exc)
    else:
        raise AssertionError("overlong SFT row must fail instead of truncating")


def test_lr_warmup_then_cosine_decay():
    values=[lr_at(step,100,3e-5,3e-6,.05) for step in range(1,101)]
    assert math.isclose(values[4],3e-5,rel_tol=1e-8)
    assert values[0]<values[4]
    assert values[-1]>=3e-6
    assert math.isclose(values[-1],3e-6,rel_tol=1e-8)
    assert all(values[i]>=values[i+1] for i in range(4,len(values)-1))
