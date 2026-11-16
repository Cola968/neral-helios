"""Twenty fresh deterministic synthetic RAG / everyday probes, never used in training.

Scores are transparent string-level *proxies*, not human quality ratings.
No personal profiles, external data, benchmark 24/40 contents or model traces.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, re, statistics, time
from collections.abc import Mapping
from pathlib import Path

SCHEMA="NERAL_DIALOG_RAG_SAMPLE_20261116_V1"

def casepack():
    samples=[]
    for i in range(8):
        code=7701+i
        days=11+2*i
        cite=f"[probe-{code}#c0]"
        chunk=f"{cite} Aus dem synthetischen Wartungsdokument: Für Messgerät B-{code} beträgt das Prüfintervall genau {days} Tage. Der Vorgang hat Kennung R-{code}."
        question=f"Wie lang ist das Prüfintervall von Messgerät B-{code}? Belege es ausschließlich mit dem Dokument."
        samples.append({"kind":"rag_present","system":"Du hast ausschließlich den gegebenen Dokumentenauszug. Antworte kurz und zitiere die Quellenkennung. Erfinde keine Fakten.",
                        "prompt":chunk+"\nFrage: "+question,
                        "must":[str(days),cite],"forbid":["nicht in den vorliegenden quellen"]})
    for i in range(5):
        code=8811+i;cite=f"[missing-{code}#c0]"
        chunk=f"{cite} Testdokument: Das Projekt M-{code} umfasst zwei Sensoren. Angaben zur Lackfarbe fehlen."
        samples.append({"kind":"rag_missing",
           "system":"Beantworte nur belegte Fragen. Wenn die Antwort im gegebenen Text nicht steht, sage das ausdrücklich und erfinde nichts.",
           "prompt":chunk+f"\nWelche Lackfarbe hat das erste Gerät im Projekt M-{code}?",
           "must":["nicht","quelle"],"forbid":["blau","rot","grün","schwarz"]})
    for i in range(4):
        x=81+11*i;y=9+3*i;v=x+y
        samples.append({"kind":"simple_calc",
           "system":"Antworte auf Deutsch, direkt und kurz. Rechne korrekt ohne zusätzliche Zahlen zu erfinden.",
           "prompt":f"Was ist {x} + {y}?",
           "must":[str(v)],"forbid":[]})
    for topic in ["meine Prüfungsvorbereitung","meine Reiseplanung","ein persönliches Budget"]:
        samples.append({"kind":"clarification",
         "system":"Bitte stelle bei unklaren Aufträgen eine einzige sinnvolle, kurze Rückfrage. Vermeide lange Standardpläne ohne die nötigen Angaben.",
         "prompt":f"Erstelle einen genauen Plan für {topic}.",
         "must":["?"],"forbid":[]})
    if len(samples)!=20:raise RuntimeError("Sample must have 20 tasks")
    seen={hashlib.sha256(x["prompt"].encode()).hexdigest() for x in samples}
    if len(seen)!=20:raise RuntimeError("Sample duplicate")
    return samples

def score(row,text):
    if not isinstance(text,str) or not text.strip():return False
    low=text.casefold()
    return all(s.casefold() in low for s in row["must"]) and not any(s.casefold() in low for s in row["forbid"])

def evaluate(label,base_path,adapter_path,cases):
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(str(base_path),local_files_only=True,use_fast=True)
    base=Qwen3_5ForCausalLM.from_pretrained(str(base_path),
       quantization_config=BitsAndBytesConfig(load_in_4bit=True,
           bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=dtype),
       dtype=dtype,device_map={"":0},low_cpu_mem_usage=True,local_files_only=True)
    model=(PeftModel.from_pretrained(base,str(adapter_path),is_trainable=False)
           if adapter_path else base)
    model.eval();times=[];correct=0;by_kind={}
    for i,row in enumerate(cases):
        turns=[{"role":"system","content":row["system"]},{"role":"user","content":row["prompt"]}]
        tokens=tokenizer.apply_chat_template(turns,add_generation_prompt=True,
                          tokenize=True,enable_thinking=False)
        if isinstance(tokens,Mapping):tokens=tokens["input_ids"]
        if type(tokens) is not list or len(tokens)>3800:
            raise RuntimeError("Invalid short-dialogue tokenization")
        inp=torch.tensor([tokens],device="cuda",dtype=torch.long)
        torch.cuda.synchronize();t0=time.monotonic()
        with torch.inference_mode():
            out=model.generate(input_ids=inp,do_sample=False,max_new_tokens=256,
                  use_cache=True,pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        secs=time.monotonic()-t0
        reply=tokenizer.decode(out[0,len(tokens):],skip_special_tokens=True)
        good=score(row,reply)
        times.append(secs);correct+=int(good)
        st=by_kind.setdefault(row["kind"],{"correct":0,"total":0})
        st["correct"]+=int(good);st["total"]+=1
        print("NERAL_V2_DIALOGUE_PROGRESS "+label+" "+str(i+1)+"/20",flush=True)
        del inp,out
    result={"model":label,"n":20,"automated_proxy_passed":correct,
            "by_kind":by_kind,"median_latency_seconds":round(statistics.median(times),3),
            "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
            "scores_are_human_judgements":False}
    del model,base;gc.collect();torch.cuda.empty_cache()
    print("NERAL_V2_DIALOGUE_SCALAR "+json.dumps(result,sort_keys=True),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ("base9","v1","full","base4","luna","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Dialogue report already exists")
    from scripts.neral_v2_final_eval40 import validate_frozen
    validate_frozen(a.full,a.v1,a.out)
    lm=json.loads((a.luna/"manifest.json").read_text(encoding="utf8"))
    if (lm.get("status")!="frozen_adapter_ready_for_heldout_comparison"
        or lm.get("family")!="luna" or lm.get("train_rows")!=480
        or lm.get("val_rows")!=75 or lm.get("test_rows_opened")!=0
        or lm.get("docker_verified") is not True):
        raise RuntimeError("Luna 4B adapter was not proven frozen")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("Real RTX4070 required")
    cases=casepack()
    groups=[("Qwen3.5-9B-base",a.base9,None),
            ("QLoRA-v1-9B",a.base9,a.v1),
            ("Helios-Terra-v2-full-9B",a.base9,a.full/"adapter"),
            ("Qwen3.5-4B-base",a.base4,None),
            ("Luna-v1-4B",a.base4,a.luna/"adapter")]
    output={name:evaluate(name,base,adapter,cases) for name,base,adapter in groups}
    report={"status":"synthetic_20_dialogue_rag_proxy","casepack_id":SCHEMA,
            "heldout_from_training":True,"real_human_review":False,
            "no_model_weights_uploaded":True,"models":output}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_V2_DIALOG_RAG_FINAL "+json.dumps({
        k:{"passed":v["automated_proxy_passed"],"median_s":v["median_latency_seconds"],
           "peak_mib":v["peak_reserved_mib"]} for k,v in output.items()},sort_keys=True),flush=True)

if __name__=="__main__":main()
