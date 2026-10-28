from __future__ import annotations

def render_history(history):
    parts=[]
    for role,text in history:
        text=str(text).strip()
        if role=="user":
            parts.append("User: "+text+"\nAssistant:")
        elif role=="assistant":
            parts.append(" "+text+"\n")
        else:
            raise ValueError("role must be user/assistant")
    return "".join(parts)

def prompt_ids(tokenizer,history):
    bos=tokenizer.token_to_id("<|bos|>")
    if bos is None:
        raise ValueError("tokenizer missing BOS")
    # SFT/DPO encode each role segment independently. Encoding the whole
    # concatenated transcript here can produce different BPE merges across
    # the Assistant:/answer and answer/User: boundaries.
    # Preserve exactly the same token boundaries at inference time.
    ids = [int(bos)]
    for role, text in history:
        content = str(text).strip()
        if role == "user":
            segment = f"User: {content}\nAssistant:"
        elif role == "assistant":
            segment = f" {content}\n"
        else:
            raise ValueError("role must be user/assistant")
        ids.extend(tokenizer.encode(segment).ids)
    return ids

def trim_history(tokenizer,history,max_prompt_tokens):
    if max_prompt_tokens<32:
        raise ValueError("max_prompt_tokens too small")
    h=list(history)
    while h and len(prompt_ids(tokenizer,h))>max_prompt_tokens:
        if len(h)>=3 and h[0][0]=="user" and h[1][0]=="assistant":
            h=h[2:]
        else:
            h=h[1:]
    if not h or h[-1][0]!="user":
        raise ValueError("latest user prompt cannot fit context")
    if len(prompt_ids(tokenizer,h))>max_prompt_tokens:
        raise ValueError("latest user prompt too long")
    return h

