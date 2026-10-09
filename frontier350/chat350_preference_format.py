from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EncodedFinalResponse:
    input_ids:list[int]
    targets:list[int]
    response_tokens:int

def _id(tok,t):
    v=tok.token_to_id(t)
    if v is None: raise ValueError(f"missing required special token: {t}")
    return int(v)

def _enc(tok,text): return list(tok.encode(text).ids)

def validate_context(context):
    if not isinstance(context,list) or not context or context[0].get("role")!="user" or context[-1].get("role")!="user": raise ValueError("context must start/end with user")
    exp="user"
    for m in context:
        if m.get("role")!=exp or not str(m.get("content","")).strip(): raise ValueError(f"invalid {exp} context turn")
        exp="assistant" if exp=="user" else "user"

def encode_final_response(tok,context,response,*,max_seq_len):
    validate_context(context); response=str(response).strip()
    if not response: raise ValueError("empty preference response")
    ids=[_id(tok,"<|bos|>")]; sup=[False]
    for m in context:
        role=m["role"]; content=str(m["content"]).strip()
        text=f"User: {content}\nAssistant:" if role=="user" else f" {content}\n"
        seg=_enc(tok,text);ids.extend(seg);sup.extend([False]*len(seg))
    seg=_enc(tok,f" {response}\n");ids.extend(seg);sup.extend([True]*len(seg))
    eos=_id(tok,"<|eos|>");ids.append(eos);sup.append(True)
    if len(ids)-1>max_seq_len: raise ValueError(f"encoded preference exceeds max_seq_len={max_seq_len}: {len(ids)-1}")
    ts=[x if s else -100 for x,s in zip(ids,sup)]
    x,y=ids[:-1],ts[1:];n=sum(t!=-100 for t in y)
    if not n or y[-1]!=eos: raise AssertionError("response alignment failure")
    return EncodedFinalResponse(x,y,n)
