"""Luna-only Qwen3.5-4B assistant content mask from one native rendered chat.

Unlike earlier Qwen3.5-9B templates, 4B may render a different assistant header
for generation prompts and full assistant messages. Only native full chat is
tokenized; offsets prove exactly where answer tokens start. No loss on prompts.
"""
from __future__ import annotations
from collections.abc import Mapping


def assistant_only_4b(rows,tokenizer,*,max_seq_length:int):
    if not 512<=max_seq_length<=4096:
        raise ValueError("Invalid context budget")
    output=[]
    for row in rows:
        turns=row["messages"]
        if (not isinstance(turns,list) or len(turns)!=3 or
            [m.get("role") for m in turns] != ["system","user","assistant"] or
            any(type(m.get("content")) is not str or not m["content"] for m in turns)):
            raise ValueError("Only complete three-message supervised conversations")
        answer=turns[-1]["content"]
        rendered=tokenizer.apply_chat_template(
            turns,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        if type(rendered) is not str:
            raise RuntimeError("4B native chat template must render UTF-8 text")
        start=rendered.rfind(answer)
        if start<0 or rendered.find("<|im_start|>assistant",0,start)<0:
            raise RuntimeError("Luna assistant content not found in own final assistant segment")
        header=rendered.rfind("<|im_start|>assistant",0,start)
        if "<|im_end|>" in rendered[header:start] or "<|im_start|>" in rendered[header+len("<|im_start|>assistant"):start]:
            raise RuntimeError("Luna target shifted outside assistant section")
        full=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        if not isinstance(full,Mapping) or "input_ids" not in full or "offset_mapping" not in full:
            raise RuntimeError("Native Qwen3.5-4B fast tokenizer offset mapping required")
        ids=full["input_ids"]
        offsets=full["offset_mapping"]
        if (not isinstance(ids,list) or not isinstance(offsets,list)
            or len(ids)!=len(offsets)
            or any(type(x) is not int or x<0 for x in ids)
            or any(type(a) is not int or type(b) is not int or a>b for a,b in offsets)):
            raise RuntimeError("Invalid fast tokenization offsets")
        # Verify tokenizer(rendered) is byte-for-byte token-equivalent to the
        # native template's tokenize=True path. Refuse silent re-serialization.
        native=tokenizer.apply_chat_template(
            turns,tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if isinstance(native,Mapping):native=native.get("input_ids")
        if ids!=native:
            raise RuntimeError("Native 4B chat template tokenization does not match offsets")
        if len(ids)>max_seq_length:
            raise RuntimeError("Luna chat target exceeds strict context cap")
        first=next((i for i,(a,b) in enumerate(offsets)
                    if a>=start and a < start+len(answer) and b>a),None)
        if first is None:
            raise RuntimeError("No assistant target tokens found")
        if offsets[first][0]!=start:
            raise RuntimeError("Luna target begins within a merged prompt token")
        if any(b>start for a,b in offsets[:first] if b>a):
            raise RuntimeError("Assistant prompt token accidentally reaches answer")
        # Once target starts, also supervise assistant's native EOS/end marker.
        labels=[-100]*first+ids[first:]
        if not any(x>=0 for x in labels) or any(x>=0 for x in labels[:first]):
            raise RuntimeError("No correct assistant-only token labels")
        output.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels})
    return output
