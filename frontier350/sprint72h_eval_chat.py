"""Fast public-development chat A/B for the 72h Helios sprint.

Not a release benchmark. Compares the immutable 2.073B base and the sprint SFT candidate
on simple DE/EN instruction, arithmetic, factual and multi-turn cases. Writes raw outputs
locally so the operator can inspect whether the candidate actually behaves like a chatbot.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from nextgen_v1.helios_chat_core import generate_turn

BASE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"

CASES=[
 {"id":"de_exact","messages":[{"role":"user","content":"Antworte nur mit dem Wort blau."}],"need":["blau"]},
 {"id":"de_math","messages":[{"role":"user","content":"Was ist 17 mal 23? Antworte kurz."}],"need":["391"]},
 {"id":"de_fact","messages":[{"role":"user","content":"Was ist die Hauptstadt von Frankreich?"}],"need":["paris"]},
 {"id":"de_science","messages":[{"role":"user","content":"Welcher Planet wird der rote Planet genannt?"}],"need":["mars"]},
 {"id":"en_exact","messages":[{"role":"user","content":"Reply with only the word green."}],"need":["green"]},
 {"id":"en_math","messages":[{"role":"user","content":"What is 12 times 14? Answer briefly."}],"need":["168"]},
 {"id":"en_fact","messages":[{"role":"user","content":"What is the capital of Germany?"}],"need":["berlin"]},
 {"id":"en_science","messages":[{"role":"user","content":"At what Celsius temperature does pure water freeze at standard pressure?"}],"need":["0"]},
 {"id":"de_memory","messages":[{"role":"user","content":"Ich heiße Alex."},{"role":"assistant","content":"Hallo Alex!"},{"role":"user","content":"Wie heiße ich?"}],"need":["alex"]},
 {"id":"en_memory","messages":[{"role":"user","content":"My favorite fruit is mango."},{"role":"assistant","content":"Got it."},{"role":"user","content":"What is my favorite fruit?"}],"need":["mango"]},
]


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()


def repetition_bad(text:str)->bool:
 words=re.findall(r"\w+",text.casefold())
 if len(words)<12:return False
 grams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
 return len(grams)>0 and (len(grams)-len(set(grams)))/len(grams)>0.35


def main():
 p=argparse.ArgumentParser()
 p.add_argument("--execute",action="store_true")
 p.add_argument("--base",type=Path,required=True)
 p.add_argument("--candidate",type=Path,required=True)
 p.add_argument("--tokenizer",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if not a.execute:p.error("--execute required")
 if sha(a.base)!=BASE_SHA:raise ValueError("base SHA mismatch")
 if sha(a.tokenizer)!=TOKENIZER_SHA:raise ValueError("tokenizer SHA mismatch")
 if a.out.exists():raise FileExistsError(a.out)

 import torch
 from tokenizers import Tokenizer
 from model350 import FRONTIER_350,Neral350M
 tok=Tokenizer.from_file(str(a.tokenizer));bos=tok.token_to_id("<|bos|>");eos=tok.token_to_id("<|eos|>")
 if bos is None or eos is None:raise ValueError("special token IDs missing")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32

 def run(path:Path,kind:str):
  state=torch.load(path,map_location="cpu",weights_only=False)
  if state.get("config")!=FRONTIER_350.to_dict() or state.get("tokenizer_sha256")!=TOKENIZER_SHA:
   raise ValueError(f"{kind}: checkpoint contract mismatch")
  m=Neral350M().to(device=device,dtype=dtype);m.load_state_dict(state["model"],strict=True);m.eval()
  results=[];correct=0;nonempty=0;loops=0
  with torch.inference_mode():
   for case in CASES:
    def next_token(ids,emitted):
     x=torch.tensor([ids],device=device,dtype=torch.long)
     logits,_=m(x)
     scores=logits[0,-1].float().clone()
     for token in set(emitted[-24:]):
      if token!=eos:scores[token]-=0.35
     return int(scores.argmax().item())
    out=generate_turn(tok,bos,eos,case["messages"],48,next_token)
    answer=out["answer"].strip();low=answer.casefold()
    ok=all(x.casefold() in low for x in case["need"])
    bad=repetition_bad(answer)
    correct+=int(ok);nonempty+=int(bool(answer));loops+=int(bad)
    results.append({"id":case["id"],"answer":answer,"correct":ok,
                    "finish_reason":out["finish_reason"],"repetition_bad":bad})
  del m
  if device.type=="cuda":torch.cuda.empty_cache()
  return {"kind":kind,"correct":correct,"total":len(CASES),"nonempty":nonempty,
          "repetition_bad":loops,"results":results}

 base=run(a.base,"immutable_2.073B_base")
 cand=run(a.candidate,"sprint72h_sft")
 gate=(cand["correct"]>=max(4,base["correct"]+2) and cand["nonempty"]==len(CASES)
       and cand["repetition_bad"]<=1)
 report={"schema":1,"classification":"public_development_chat_ab_not_release_benchmark",
         "base":base,"candidate":cand,
         "delta_correct":cand["correct"]-base["correct"],
         "chat_mvp_gate_passed":gate,
         "quality_verified":False,
         "note":"Development gate only; passing means continue to human chat testing, not release-quality proof."}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
 print(json.dumps({k:v for k,v in report.items() if k not in ("base","candidate")},
                  indent=2,ensure_ascii=False))
 print(json.dumps({"base_correct":base["correct"],"candidate_correct":cand["correct"],
                   "candidate_nonempty":cand["nonempty"],"candidate_repetition_bad":cand["repetition_bad"],
                   "chat_mvp_gate_passed":gate},ensure_ascii=False))


if __name__=="__main__":main()
