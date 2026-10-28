"""Auditable token-weighted answer-NLL decomposition for native Arctus350.

The tokenizer is BPE, so the first generated response token can contain a
word as well as whitespace: do not falsely call it a pure 'format token'.
The only unambiguous special target is terminal EOS. We report opening
assistant token, remaining response tokens and final EOS separately.

No torch, CUDA or checkpoint imports; testable in CPU-only CI.
"""
from __future__ import annotations

import hashlib
import math

try:
    from .chat350_sft_format import encode_messages, EncodedChatExample
except ImportError:  # Existing native scripts execute with PYTHONPATH=frontier350
    from chat350_sft_format import encode_messages, EncodedChatExample


NLL_GROUPS=("opening","body","eos")


def classify_supervised_tokens(tokenizer, messages:list[dict[str,str]],
                               encoded:EncodedChatExample, *, max_seq_len:int)->list[str]:
    """Return per-target partitions exactly aligned with original shifted SFT labels."""
    if not isinstance(encoded,EncodedChatExample):
        raise ValueError("invalid EncodedChatExample")
    check=encode_messages(tokenizer,messages,max_seq_len=max_seq_len)
    if check!=encoded:
        raise ValueError("encoded conversation differs from source transcript")
    tags=["masked"]*len(encoded.targets)
    cursor=1  # BOS is at index 0 of unshifted token stream
    for msg in messages:
        role=msg["role"]
        content=str(msg["content"]).strip()
        if role=="user":
            seg=tokenizer.encode(f"User: {content}\nAssistant:").ids
            cursor+=len(seg)
        else:
            seg=tokenizer.encode(f" {content}\n").ids
            if not seg:
                raise ValueError("empty assistant token segment")
            first=cursor-1  # target index shifted one left from token position
            tags[first]="opening"
            for ix in range(1,len(seg)):
                tags[first+ix]="body"
            cursor+=len(seg)
    if cursor!=len(encoded.targets):
        raise ValueError("EOS alignment mismatch with actual original SFT labels")
    tags[-1]="eos"
    for label,tag in zip(encoded.targets,tags):
        if (label==-100)!=(tag=="masked"):
            raise ValueError("supervision mask does not match partition")
    if sum(x!="masked" for x in tags)!=encoded.assistant_tokens:
        raise ValueError("assistant token count mismatch")
    return tags


def stable_subset(rows:list[dict],limit:int)->list[dict]:
    """Reproducible id-hash sample, order-independent and never split-dependent."""
    if type(limit) is not int or limit<1 or limit>len(rows):
        raise ValueError("invalid sample limit")
    if any(not isinstance(r,dict) or not isinstance(r.get("id"),str) or not r["id"] for r in rows):
        raise ValueError("rows need stable nonempty ids")
    keys=[r["id"] for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError("duplicate validation ids")
    return sorted(rows,key=lambda r:(hashlib.sha256(r["id"].encode("utf8")).hexdigest(),r["id"]))[:limit]


def aggregate_nll(samples:list[dict])->dict:
    """Combine category token sums, never mean-of-per-example or per-batch means."""
    sums={key:0.0 for key in NLL_GROUPS}
    counts={key:0 for key in NLL_GROUPS}
    for sample in samples:
        for key in NLL_GROUPS:
            pair=sample.get(key)
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError("missing loss sum/count for "+key)
            value,num=pair
            if (type(value) not in (float,int) or not math.isfinite(value) or value<0
                or type(num) is not int or num<0):
                raise ValueError("invalid loss sum or token count for "+key)
            sums[key]+=float(value)
            counts[key]+=num
    total=sum(counts.values())
    if total==0:
        raise ValueError("no observed supervised assistant tokens")
    out={
        "answer_tokens":total,
        "answer_nll":sum(sums.values())/total,
        "proxy_definition":"first_assistant_token_vs_remaining_non_eos_and_terminal_eos",
    }
    for group in NLL_GROUPS:
        out[group+"_tokens"]=counts[group]
        out[group+"_nll"]=sums[group]/counts[group] if counts[group] else None
    return out
