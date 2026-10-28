# Arctus 350M — Claude Code Review · Teil B: Evaluation, Checkpointing, Laufparameter

**Source:** private `Cola968/neral-helios`, pinned commit `bf9f638f6c45311c175c3f154d7c62a7952b8465` (`arctus350-chat-cloud-prep`). Generated 2026-10-28. This is the **historical code at the time of the SFT research**, not a claim that current tests have been executed by Claude. Review source lines are prefixed `L### |` for reliable references; remove markers to run. **No checkpoint bytes, local corpora, access tokens, or frozen benchmark questions/answer keys included.**

## Key research context and limits

- Native-from-scratch **Arctus 350M**, user-reported **over 2B total training tokens**. The 1B SAFE and later 2B continuation checkpoints are **distinct**; no local checkpoint was made available in this review packet.
- The project research note reports a 1,200-step SFT pilot: validation answer NLL **3.163 → 1.736**, frozen capability probe **3/30 → 2/30**. Small n; one-point change can be noise, not definitive proof of regression or bug. Reviewer should check underlying format, code and statistical validity.
- Pinned 16K BPE tokenizer SHA-256 from `chat350_contract.py`: `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.
- **Today a separate draft change [PR #140](https://github.com/Cola968/neral-helios/pull/140) repaired evaluator/release vulnerabilities** (duplicate case IDs, missing values, category regressions and suite hash). The code below is the *source/base-branch version*, preserved to help inspect the old training experiment. Do not confuse evaluator defects with an actual measured new training improvement.
- Avoid leaking the frozen capability suite. If asking for concrete examples, use separately selected public/non-held-out synthetic cases, or redact scoring answers.

## Actual training invocation from the repository workflow

The trusted self-hosted Windows runner workflow `.github/workflows/arctus350-chat-pilot.yml` at the same pinned commit selects a **1B SAFE foundation checkpoint** and starts a bounded SFT pilot using: `--steps 1200 --seq 512 --batch-size 4 --replay-batch 2 --replay-seq 512 --replay-every 4 --sft-weight 0.90 --eos-weight 1.5 --peak-lr 0.000015 --min-lr 0.000001 --eval-every 200 --save-every 200 --gradient-checkpointing --sampling-mode human-balanced --human-per-batch 2`. Actual run logs and local checkpoint identities are **not** included. **No training is executed by preparing this packet.**

Source workflow: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/.github/workflows/arctus350-chat-pilot.yml#L83-L95

## What I want you to analyze

1. Test whether `eval_chat350.py` serializes prompts exactly like `chat350_sft_format.py` trained them, including BOS/EOS and User:/Assistant: markers.
2. Trace the masked labels, shifted CE loss, pad masking, EOS loss weighting and answer-only validation NLL across `train_chat350.py` → `eval_chat350_slices.py`.
3. Check `eval_chat350.py` decoding, EOS stop, max_new_tokens and the structure `score_chat350_capabilities.py` reads; distinguish preference/choice scoring from free generation.
4. Inspect whether `best_chat.pt` selection, replay loss checks, data split checks and release guards select a checkpoint for lower NLL rather than improved real capabilities.
5. For each finding: severity P0–P3, precise file and line, concrete reproducer/regression-test sketch, and whether it is confirmed by static code review or only a hypothesis requiring a GPU/model run.

---

## 1. `frontier350/eval_chat350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/eval_chat350.py  
Git blob SHA: `7a0e41c1a0c900a3d97e5e4bd9045c53be76e2d4` · 72 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse
L003 | import hashlib
L004 | import json
L005 | from pathlib import Path
L006 | 
L007 | import torch
L008 | from tokenizers import Tokenizer
L009 | 
L010 | from chat350_contract import EXPECTED_TOKENIZER_SHA256
L011 | from secure_checkpoint350 import load_checkpoint
L012 | from chat350_generation_metrics import summarize_generations
L013 | from model350 import FRONTIER_350, Neral350M
L014 | 
L015 | def sha256_file(path: Path):
L016 |     h=hashlib.sha256()
L017 |     with path.open("rb") as f:
L018 |         for c in iter(lambda:f.read(4*1024*1024),b""): h.update(c)
L019 |     return h.hexdigest()
L020 | 
L021 | def load_cases(path: Path):
L022 |     obj=json.loads(path.read_text(encoding="utf8"))
L023 |     if not isinstance(obj,list) or not obj: raise ValueError("cases must be a non-empty JSON list")
L024 |     ids=set()
L025 |     for row in obj:
L026 |         if row.get("id") in ids: raise ValueError("duplicate case id")
L027 |         ids.add(row.get("id"))
L028 |         if not row.get("prefix") or int(row.get("max_new_tokens",0))<=0: raise ValueError("invalid generation case")
L029 |     return obj
L030 | 
L031 | @torch.inference_mode()
L032 | def generate(model,tok,prefix,max_new,device):
L033 |     bos=int(tok.token_to_id("<|bos|>")); eos=int(tok.token_to_id("<|eos|>"))
L034 |     ids=[bos]+list(tok.encode(prefix).ids); generated=[]
L035 |     for _ in range(max_new):
L036 |         if len(ids)>=FRONTIER_350.max_seq_len: break
L037 |         x=torch.tensor([ids],device=device,dtype=torch.long)
L038 |         logits,_=model(x); nxt=int(logits[0,-1].argmax())
L039 |         generated.append(nxt); ids.append(nxt)
L040 |         if nxt==eos: break
L041 |     content=generated[:-1] if generated and generated[-1]==eos else generated
L042 |     return generated,tok.decode(content,skip_special_tokens=True)
L043 | 
L044 | def main():
L045 |     ap=argparse.ArgumentParser()
L046 |     ap.add_argument("--checkpoint",required=True); ap.add_argument("--tokenizer",required=True)
L047 |     ap.add_argument("--cases",required=True); ap.add_argument("--out",required=True)
L048 |     args=ap.parse_args()
L049 |     token_path=Path(args.tokenizer)
L050 |     if sha256_file(token_path)!=EXPECTED_TOKENIZER_SHA256: raise ValueError("tokenizer digest mismatch")
L051 |     tok=Tokenizer.from_file(str(token_path)); cases=load_cases(Path(args.cases))
L052 |     ck=load_checkpoint(args.checkpoint)
L053 |     if ck.get("config")!=FRONTIER_350.to_dict(): raise ValueError("checkpoint architecture mismatch")
L054 |     if ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256: raise ValueError("checkpoint/tokenizer mismatch")
L055 |     state=ck.get("model")
L056 |     if not isinstance(state,dict): raise ValueError("checkpoint has no model state")
L057 |     if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
L058 |     device=torch.device("cuda"); model=Neral350M().to(device=device,dtype=torch.bfloat16); model.load_state_dict(state,strict=True); model.eval()
L059 |     rows=[]
L060 |     for case in cases:
L061 |         ids,text=generate(model,tok,case["prefix"],int(case["max_new_tokens"]),device)
L062 |         row={"id":case["id"],"category":case.get("category","unknown"),"generated_ids":ids,"text":text}
L063 |         if "expected_exact" in case: row["expected_exact"]=case["expected_exact"]
L064 |         rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
L065 |     summary=summarize_generations(rows,int(tok.token_to_id("<|eos|>")))
L066 |     case_sha=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
L067 |     report={"checkpoint_sha256":sha256_file(Path(args.checkpoint)),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
L068 |             "case_sha256":case_sha,"summary":summary,"rows":rows}
L069 |     Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
L070 |     print(json.dumps({"summary":summary,"case_sha256":case_sha},ensure_ascii=False),flush=True)
L071 | 
L072 | if __name__=="__main__": main()
```

## 2. `frontier350/eval_chat350_slices.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/eval_chat350_slices.py  
Git blob SHA: `34009edb366c04ae21af69e39a0b4b49598ba22a` · 45 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse,collections,hashlib,json
L003 | from pathlib import Path
L004 | import torch
L005 | from chat350_contract import EXPECTED_TOKENIZER_SHA256
L006 | from chat350_data import validate_dataset_root
L007 | from chat350_sft_format import encode_messages
L008 | from model350 import FRONTIER_350,Neral350M
L009 | from train_chat350 import load_tokenizer,eval_sft
L010 | 
L011 | def sha256_file(path):
L012 |     h=hashlib.sha256()
L013 |     with Path(path).open("rb") as f:
L014 |         for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
L015 |     return h.hexdigest()
L016 | 
L017 | def group_rows(rows):
L018 |     groups={"overall":rows}
L019 |     def add(name,row):groups.setdefault(name,[]).append(row)
L020 |     for r in rows:
L021 |         add("language:"+r["language"],r)
L022 |         add("domain:"+r["domain"],r)
L023 |         add("source:"+r["source"],r)
L024 |         add("turns:"+("multi" if len(r["messages"])>=4 else "single"),r)
L025 |     return dict(sorted(groups.items()))
L026 | 
L027 | def main():
L028 |     ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--data",required=True);ap.add_argument("--split",choices=["val","test"],default="val");ap.add_argument("--out",required=True);ap.add_argument("--max-seq",type=int,default=512);ap.add_argument("--batch-size",type=int,default=4);a=ap.parse_args()
L029 |     manifest,splits=validate_dataset_root(Path(a.data));rows=splits[a.split]
L030 |     tok=load_tokenizer(Path(a.tokenizer));ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
L031 |     if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
L032 |     if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
L033 |     dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
L034 |     pad=int(tok.token_to_id("<|pad|>"));eos=int(tok.token_to_id("<|eos|>"));report={}
L035 |     for name,rs in group_rows(rows).items():
L036 |         encoded=[];skipped=0
L037 |         for r in rs:
L038 |             try:encoded.append(encode_messages(tok,r["messages"],max_seq_len=a.max_seq))
L039 |             except ValueError:skipped+=1
L040 |         if not encoded:continue
L041 |         ev=eval_sft(m,encoded,a.batch_size,pad,eos,dev)
L042 |         report[name]={"rows":len(encoded),"skipped_too_long":skipped,**ev}
L043 |     out={"schema":1,"checkpoint_sha256":sha256_file(a.checkpoint),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"dataset_manifest_sha256":manifest["manifest_sha256"],"split":a.split,"max_seq":a.max_seq,"slices":report}
L044 |     Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({"split":a.split,"slices":len(report),"overall":report.get("overall")},ensure_ascii=False))
L045 | if __name__=="__main__":main()
```

## 3. `frontier350/score_chat350_capabilities.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/score_chat350_capabilities.py  
Git blob SHA: `b7a41e0312418bbf536988f771186574d0ddffcb` · 53 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse,collections,json,re
L003 | from pathlib import Path
L004 | 
L005 | def norm(text:str)->str:
L006 |     return " ".join(str(text).casefold().strip().split())
L007 | 
L008 | def score_one(text:str,spec:dict)->bool:
L009 |     t=norm(text);typ=spec.get("type")
L010 |     if typ=="exact":
L011 |         return str(text).strip()==str(spec["value"]).strip()
L012 |     if typ=="integer":
L013 |         # Match signed integers even when followed by normal punctuation ("95.", "-7,").
L014 |         # Thousands/decimal separators are intentionally unsupported for this frozen benchmark.
L015 |         found=re.findall(r"(?<![\w.,])[+-]?\d+(?![\w]|[.,]\d)",str(text))
L016 |         return bool(found) and int(found[-1])==int(spec["value"])
L017 |     if typ=="contains_all":
L018 |         return all(norm(x) in t for x in spec.get("values",[]))
L019 |     if typ=="contains_any":
L020 |         ok=any(norm(x) in t for x in spec.get("values",[]))
L021 |         return ok and all(norm(x) in t for x in spec.get("and_values",[]))
L022 |     raise ValueError(f"unknown scoring type: {typ}")
L023 | 
L024 | def score(report:dict,cases:list[dict])->dict:
L025 |     cmap={c["id"]:c for c in cases}
L026 |     rows=report.get("rows")
L027 |     if not isinstance(rows,list):raise ValueError("report missing rows")
L028 |     got={r.get("id"):r for r in rows}
L029 |     if set(got)!=set(cmap):raise ValueError(f"case/report id mismatch: missing={sorted(set(cmap)-set(got))} extra={sorted(set(got)-set(cmap))}")
L030 |     cats=collections.defaultdict(list);detail=[]
L031 |     for cid,c in cmap.items():
L032 |         text=str(got[cid].get("text",""))
L033 |         ok=score_one(text,c["scoring"]);cats[c["category"]].append(ok)
L034 |         detail.append({"id":cid,"category":c["category"],"pass":ok,"text":text})
L035 |     by={k:{"passed":sum(v),"cases":len(v),"pct":100*sum(v)/len(v)} for k,v in sorted(cats.items())}
L036 |     total=sum(x["pass"] for x in detail)
L037 |     return {"cases":len(detail),"passed":total,"overall_pct":100*total/len(detail),"by_category":by,"details":detail}
L038 | 
L039 | def compare(base:dict,cand:dict)->dict:
L040 |     cats=sorted(set(base["by_category"])|set(cand["by_category"]))
L041 |     deltas={k:cand["by_category"].get(k,{"pct":0})["pct"]-base["by_category"].get(k,{"pct":0})["pct"] for k in cats}
L042 |     hard_reg=[k for k,d in deltas.items() if d < -20.0]
L043 |     return {"overall_delta_pp":cand["overall_pct"]-base["overall_pct"],"category_delta_pp":deltas,"hard_regressions":hard_reg,
L044 |             "no_hard_category_regression":not hard_reg,"nonnegative_overall":cand["overall_pct"]>=base["overall_pct"],
L045 |             "capability_no_regression":not hard_reg and cand["overall_pct"]>=base["overall_pct"]}
L046 | 
L047 | def main():
L048 |     ap=argparse.ArgumentParser();ap.add_argument("--report",required=True);ap.add_argument("--cases",required=True);ap.add_argument("--out",required=True);ap.add_argument("--baseline-score");a=ap.parse_args()
L049 |     report=json.loads(Path(a.report).read_text(encoding="utf8"));cases=json.loads(Path(a.cases).read_text(encoding="utf8"));res=score(report,cases)
L050 |     if a.baseline_score:
L051 |         base=json.loads(Path(a.baseline_score).read_text(encoding="utf8"));res["comparison"]=compare(base,res)
L052 |     Path(a.out).write_text(json.dumps(res,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({k:v for k,v in res.items() if k!="details"},ensure_ascii=False))
L053 | if __name__=="__main__":main()
```

## 4. `frontier350/chat350_generation_metrics.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat350_generation_metrics.py  
Git blob SHA: `7bd57f6f34dd739b4bb1d8c5ceed2f7231c92614` · 68 lines.

```text
L001 | from __future__ import annotations
L002 | import statistics
L003 | 
L004 | def repeated_ngram_fraction(ids: list[int], n: int = 3) -> float:
L005 |     if n <= 0:
L006 |         raise ValueError("n must be positive")
L007 |     if len(ids) < n:
L008 |         return 0.0
L009 |     grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
L010 |     seen = set()
L011 |     repeated = 0
L012 |     for g in grams:
L013 |         if g in seen:
L014 |             repeated += 1
L015 |         seen.add(g)
L016 |     return repeated / len(grams)
L017 | 
L018 | def distinct_ngram_fraction(ids: list[int], n: int = 2) -> float:
L019 |     if len(ids) < n:
L020 |         return 1.0
L021 |     grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
L022 |     return len(set(grams)) / len(grams)
L023 | 
L024 | def summarize_generations(rows: list[dict], eos_id: int) -> dict:
L025 |     if not rows:
L026 |         raise ValueError("generation rows are empty")
L027 |     rep, d2, lengths, eos, utf8 = [], [], [], [], []
L028 |     exact = []
L029 |     for row in rows:
L030 |         ids = [int(x) for x in row.get("generated_ids", [])]
L031 |         hit = eos_id in ids
L032 |         content_ids = ids[:ids.index(eos_id)] if hit else ids
L033 |         text = str(row.get("text", ""))
L034 |         try:
L035 |             text.encode("utf8", "strict")
L036 |             utf8.append(1.0)
L037 |         except UnicodeError:
L038 |             utf8.append(0.0)
L039 |         rep.append(repeated_ngram_fraction(content_ids, 3))
L040 |         d2.append(distinct_ngram_fraction(content_ids, 2))
L041 |         lengths.append(len(content_ids))
L042 |         eos.append(float(hit))
L043 |         if "expected_exact" in row:
L044 |             exact.append(float(text.strip() == str(row["expected_exact"]).strip()))
L045 |     return {
L046 |         "cases": len(rows),
L047 |         "eos_fraction": statistics.fmean(eos),
L048 |         "mean_repeated_trigram_fraction": statistics.fmean(rep),
L049 |         "mean_distinct2_fraction": statistics.fmean(d2),
L050 |         "mean_generated_tokens_before_eos": statistics.fmean(lengths),
L051 |         "utf8_fraction": statistics.fmean(utf8),
L052 |         "exact_instruction_pct": (100.0 * statistics.fmean(exact)) if exact else None,
L053 |     }
L054 | 
L055 | def compare_health(baseline: dict, candidate: dict) -> dict:
L056 |     eos_ok = candidate["eos_fraction"] >= baseline["eos_fraction"]
L057 |     repetition_ok = candidate["mean_repeated_trigram_fraction"] <= baseline["mean_repeated_trigram_fraction"] + 1e-12
L058 |     utf8_ok = candidate["utf8_fraction"] >= 0.999
L059 |     exact_base = baseline.get("exact_instruction_pct")
L060 |     exact_cand = candidate.get("exact_instruction_pct")
L061 |     exact_ok = exact_cand is None or exact_base is None or exact_cand >= exact_base
L062 |     return {
L063 |         "eos_no_regression": eos_ok,
L064 |         "repetition_no_regression": repetition_ok,
L065 |         "utf8_ok": utf8_ok,
L066 |         "exact_instruction_no_regression": exact_ok,
L067 |         "health_no_regression": eos_ok and repetition_ok and utf8_ok and exact_ok,
L068 |     }
```

## 5. `frontier350/compare_chat350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/compare_chat350.py  
Git blob SHA: `24340530ea99e46719187d4c6be6667ffa286cdb` · 19 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse,json
L003 | from pathlib import Path
L004 | from chat350_generation_metrics import compare_health
L005 | 
L006 | def main():
L007 |     ap=argparse.ArgumentParser(); ap.add_argument("--baseline",required=True); ap.add_argument("--candidate",required=True); ap.add_argument("--out",required=True); ap.add_argument("--training-summary",default="")
L008 |     a=ap.parse_args(); b=json.loads(Path(a.baseline).read_text(encoding="utf8")); c=json.loads(Path(a.candidate).read_text(encoding="utf8"))
L009 |     for k in ("tokenizer_sha256","case_sha256"):
L010 |         if b.get(k)!=c.get(k): raise ValueError(f"report identity mismatch: {k}")
L011 |     gate=compare_health(b["summary"],c["summary"])
L012 |     training=None
L013 |     if a.training_summary: training=json.loads(Path(a.training_summary).read_text(encoding="utf8"))
L014 |     trained_candidate=training is None or bool(training.get("best_checkpoint_exists"))
L015 |     decision="manual_review_required" if gate["health_no_regression"] and trained_candidate else ("reject_no_sft_improvement" if not trained_candidate else "reject_generation_regression")
L016 |     result={"baseline":b["summary"],"candidate":c["summary"],"gate":gate,"training":training,"automatic_promotion":False,"decision":decision}
L017 |     Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8"); print(json.dumps(result),flush=True)
L018 | 
L019 | if __name__=="__main__": main()
```

## 6. `frontier350/compare_chat350_slices.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/compare_chat350_slices.py  
Git blob SHA: `04f79c92c3e4e14142fc79d847d81272058c2bb1` · 27 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse,json
L003 | from pathlib import Path
L004 | 
L005 | def compare(base:dict,cand:dict,*,max_slice_regression=.10,min_tokens=128,min_rows=4):
L006 |     for k in ("tokenizer_sha256","dataset_manifest_sha256","split","max_seq"):
L007 |         if base.get(k)!=cand.get(k):raise ValueError(f"slice report identity mismatch: {k}")
L008 |     bs=base["slices"];cs=cand["slices"]
L009 |     if "overall" not in bs or "overall" not in cs:raise ValueError("missing overall slice")
L010 |     b0=float(bs["overall"]["answer_nll"]);c0=float(cs["overall"]["answer_nll"])
L011 |     eligible={};hard=[]
L012 |     for name in sorted(set(bs)&set(cs)):
L013 |         b=bs[name];c=cs[name]
L014 |         if min(int(b.get("answer_tokens",0)),int(c.get("answer_tokens",0)))<min_tokens or min(int(b.get("rows",0)),int(c.get("rows",0)))<min_rows:continue
L015 |         bn=float(b["answer_nll"]);cn=float(c["answer_nll"]);reg=cn/max(bn,1e-12)-1.0
L016 |         eligible[name]={"baseline_nll":bn,"candidate_nll":cn,"relative_change":reg,"rows":c["rows"],"answer_tokens":c["answer_tokens"]}
L017 |         if name!="overall" and reg>max_slice_regression:hard.append(name)
L018 |     result={"overall_baseline_nll":b0,"overall_candidate_nll":c0,"overall_relative_change":c0/max(b0,1e-12)-1.0,
L019 |             "overall_improved":c0<b0,"eligible_slices":eligible,"hard_regressions":hard,
L020 |             "slice_no_regression":c0<b0 and not hard,"thresholds":{"max_slice_regression":max_slice_regression,"min_tokens":min_tokens,"min_rows":min_rows}}
L021 |     return result
L022 | 
L023 | def main():
L024 |     ap=argparse.ArgumentParser();ap.add_argument("--baseline",required=True);ap.add_argument("--candidate",required=True);ap.add_argument("--out",required=True);ap.add_argument("--max-slice-regression",type=float,default=.10);a=ap.parse_args()
L025 |     b=json.loads(Path(a.baseline).read_text(encoding="utf8"));c=json.loads(Path(a.candidate).read_text(encoding="utf8"));r=compare(b,c,max_slice_regression=a.max_slice_regression)
L026 |     Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8");print(json.dumps(r))
L027 | if __name__=="__main__":main()
```

## 7. `frontier350/secure_checkpoint350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/secure_checkpoint350.py  
Git blob SHA: `b86f90a4efaedb757779b64b865997839b4145d9` · 39 lines.

```text
L001 | """Restricted deserialization of Neral-owned legacy Torch checkpoints.
L002 | 
L003 | Older 350M foundation resumes store NumPy's MT19937 RNG array. A plain
L004 | weights_only=True load rejects that array; unrestricted pickle isn't required.
L005 | Only NumPy array/dtype primitives are allowed. Architecture/tokenizer identity
L006 | is validated separately by the existing 350M contract.
L007 | """
L008 | from __future__ import annotations
L009 | 
L010 | from pathlib import Path
L011 | 
L012 | import numpy as np
L013 | import torch
L014 | 
L015 | 
L016 | def load_checkpoint(path: str | Path) -> dict:
L017 |     try:
L018 |         from numpy._core.multiarray import _reconstruct, scalar
L019 |     except ImportError:  # NumPy 1.x
L020 |         from numpy.core.multiarray import _reconstruct, scalar
L021 | 
L022 |     allowed = [
L023 |         (_reconstruct, "numpy._core.multiarray._reconstruct"),
L024 |         (_reconstruct, "numpy.core.multiarray._reconstruct"),
L025 |         (scalar, "numpy._core.multiarray.scalar"),
L026 |         (scalar, "numpy.core.multiarray.scalar"),
L027 |         np.ndarray,
L028 |         np.dtype,
L029 |         type(np.dtype("uint32")),
L030 |         type(np.dtype("int64")),
L031 |         type(np.dtype("float64")),
L032 |         type(np.dtype("bool")),
L033 |     ]
L034 |     # An explicitly scoped allowlist does not change future unpickling policy.
L035 |     with torch.serialization.safe_globals(allowed):
L036 |         payload = torch.load(str(path), map_location="cpu", weights_only=True)
L037 |     if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
L038 |         raise ValueError("Neral checkpoint must contain a model tensor dictionary")
L039 |     return payload
```

## 8. `frontier350/chat_cli350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat_cli350.py  
Git blob SHA: `aaa375e2d11930f206ed044501558e88e4374494` · 102 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse,hashlib
L003 | from pathlib import Path
L004 | import torch
L005 | from tokenizers import Tokenizer
L006 | from chat350_contract import EXPECTED_TOKENIZER_SHA256
L007 | from secure_checkpoint350 import load_checkpoint
L008 | from model350 import FRONTIER_350,Neral350M
L009 | from chat_prompt350 import prompt_ids,trim_history
L010 | from chat350_tools import tool_answer
L011 | from chat350_knowledge import KnowledgeIndex,render_knowledge_hits
L012 | 
L013 | def sha256_file(path):
L014 |     h=hashlib.sha256()
L015 |     with Path(path).open("rb") as f:
L016 |         for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
L017 |     return h.hexdigest()
L018 | 
L019 | def sample_next(logits,generated,temperature,top_p,repetition_penalty):
L020 |     z=logits.float().clone()
L021 |     if repetition_penalty<1.0:raise ValueError("repetition_penalty must be >=1")
L022 |     if repetition_penalty>1.0:
L023 |         for tid in set(generated):
L024 |             v=z[tid];z[tid]=torch.where(v<0,v*repetition_penalty,v/repetition_penalty)
L025 |     if temperature<=0:return int(torch.argmax(z))
L026 |     z=z/temperature
L027 |     probs=torch.softmax(z,dim=-1)
L028 |     if not 0<top_p<=1:raise ValueError("top_p must be in (0,1]")
L029 |     if top_p<1:
L030 |         sp,si=torch.sort(probs,descending=True);cum=torch.cumsum(sp,dim=-1)
L031 |         remove=cum-sp>top_p;sp[remove]=0;sp=sp/sp.sum()
L032 |         return int(si[torch.multinomial(sp,1)])
L033 |     return int(torch.multinomial(probs,1))
L034 | 
L035 | @torch.inference_mode()
L036 | def generate(model,tok,history,*,max_new=160,max_context=1024,temperature=.7,top_p=.9,repetition_penalty=1.08,device=None):
L037 |     if max_new<=0 or max_new>=max_context:raise ValueError("invalid max_new/max_context")
L038 |     h=trim_history(tok,history,max_context-max_new)
L039 |     ids=prompt_ids(tok,h);generated=[]
L040 |     eos=int(tok.token_to_id("<|eos|>"));bos=int(tok.token_to_id("<|bos|>"));pad=int(tok.token_to_id("<|pad|>"));sep=int(tok.token_to_id("<|sep|>"))
L041 |     dev=device or next(model.parameters()).device
L042 |     for _ in range(max_new):
L043 |         x=torch.tensor([ids],device=dev,dtype=torch.long)
L044 |         logits,_=model(x)
L045 |         z=logits[0,-1].clone()
L046 |         for bad in (bos,pad,sep):z[bad]=-float("inf")
L047 |         nxt=sample_next(z,generated,temperature,top_p,repetition_penalty)
L048 |         if nxt==eos:break
L049 |         generated.append(nxt);ids.append(nxt)
L050 |         if len(ids)>=max_context:break
L051 |     return tok.decode(generated,skip_special_tokens=True).strip(),h
L052 | 
L053 | def load_runtime(checkpoint,tokenizer):
L054 |     if sha256_file(tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("locked tokenizer SHA mismatch")
L055 |     tok=Tokenizer.from_file(str(tokenizer));ck=load_checkpoint(checkpoint)
L056 |     if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
L057 |     if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
L058 |     dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
L059 |     return m,tok,dev
L060 | 
L061 | def main():
L062 |     ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--prompt");ap.add_argument("--max-new",type=int,default=160);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--temperature",type=float,default=.7);ap.add_argument("--top-p",type=float,default=.9);ap.add_argument("--repetition-penalty",type=float,default=1.08);ap.add_argument("--tool-mode",choices=["auto","off"],default="auto");ap.add_argument("--knowledge-dir");ap.add_argument("--knowledge-limit",type=int,default=2);a=ap.parse_args()
L063 |     model,tok,dev=load_runtime(Path(a.checkpoint),Path(a.tokenizer));history=[]
L064 |     knowledge=KnowledgeIndex.from_directory(a.knowledge_dir) if a.knowledge_dir else None
L065 |     if knowledge is not None:
L066 |         print(f"Loaded {knowledge.passage_count} local knowledge passages.",file=__import__("sys").stderr)
L067 |     def answer(q):
L068 |         nonlocal history
L069 |         history.append(("user",q))
L070 |         tool=tool_answer(q) if a.tool_mode=="auto" else None
L071 |         if tool is not None:
L072 |             trimmed=trim_history(tok,history,a.max_context-a.max_new)
L073 |             history=trimmed+[("assistant",tool.text)]
L074 |             return tool.text,tool.kind
L075 |         if knowledge is not None:
L076 |             hits=knowledge.search(q,limit=a.knowledge_limit)
L077 |             if hits:
L078 |                 # Retrieved passages are untrusted data, not system instructions.
L079 |                 # Keep only citation IDs in generation history, not document bodies.
L080 |                 text=render_knowledge_hits(hits)
L081 |                 trimmed=trim_history(tok,history,a.max_context-a.max_new)
L082 |                 references=", ".join(f"{hit.source}:{hit.line}" for hit in hits)
L083 |                 history=trimmed+[("assistant",f"Lokale Quellen: {references}")]
L084 |                 return text,"local-knowledge"
L085 |         text,trimmed=generate(model,tok,history,max_new=a.max_new,max_context=a.max_context,temperature=a.temperature,top_p=a.top_p,repetition_penalty=a.repetition_penalty,device=dev)
L086 |         history=trimmed+[("assistant",text)]
L087 |         return text,"native-model"
L088 |     if a.prompt is not None:
L089 |         text,source=answer(a.prompt)
L090 |         print(text)
L091 |         print(f"[source={source}]",file=__import__("sys").stderr)
L092 |         return
L093 |     print("Arctus 350M local chat. /reset clears history, /quit exits.")
L094 |     while True:
L095 |         try:q=input("You> ").strip()
L096 |         except (EOFError,KeyboardInterrupt):print();break
L097 |         if not q:continue
L098 |         if q=="/quit":break
L099 |         if q=="/reset":history=[];print("History cleared.");continue
L100 |         text,source=answer(q)
L101 |         print(f"Neral ({source})> {text}")
L102 | if __name__=="__main__":main()
```

## 9. `frontier350/POSTTRAINING_RESEARCH_2026_10_10.md`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/POSTTRAINING_RESEARCH_2026_10_10.md  
Git blob SHA: `319aefa9068db362e98eddaf21cd22cba19a0d0b` · 20 lines.

```text
L001 | # Neral 350M: post-training data-selection research (2026-10-10)
L002 | 
L003 | Status: research only; no dataset has been admitted and no training run is authorized by this note.
L004 | 
L005 | The 1200-step SFT pilot lowered validation answer NLL (3.163 to 1.736) but decreased the frozen 30-case capability score (3 to 2). Keep this candidate rejected. Lower NLL is not evidence of better instruction following.
L006 | 
L007 | Potential evaluation hypotheses:
L008 | - Review prompt/response format parity and sampling mix before more updates.
L009 | - Compare category-level performance on held-out instruction following, German/English, code, context and calibration.
L010 | - Consider short, high-quality dialogues suitable for sub-1B models; do not blindly scale examples.
L011 | - Reject sources without independently checked license, provenance and contamination audit.
L012 | 
L013 | References (methodology only, not an approval to import data):
L014 | - https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk (small-model-oriented SFT subset; inspect component-level provenance)
L015 | - https://huggingface.co/datasets/OpenAssistant/oasst2 (human conversation trees; inspect role, quality and language filters)
L016 | - https://arxiv.org/abs/2503.11441 (diversity, difficulty and dependability in instruction data selection)
L017 | 
L018 | Known implementation issue to resolve: frontier350/chat350_knowledge.py currently tokenizes the full passage and only afterwards truncates the displayed excerpt to 2400 characters. Tokenization must use the displayed excerpt to avoid unsupported matches. Regression test: a query that only matches text beyond character 2400 must not return a hit.
L019 | 
L020 | No weights, checkpoints, or training artifacts should be promoted without a non-regression result on frozen capability tests and replay gates.
```

## 10. `frontier350/CHAT350_RELEASE_GATES.md`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/CHAT350_RELEASE_GATES.md  
Git blob SHA: `9d23e5281f996df5cf41f3e0f2251e5bc06a2d6f` · 29 lines.

```text
L001 | # Arctus 350M release gates
L002 | 
L003 | A model is never promoted because a single loss or chat sample looks good.
L004 | 
L005 | Required machine evidence before manual release review:
L006 | 1. Data contract and provenance pass.
L007 | 2. NeralTalk quality gate passes.
L008 | 3. SFT produced a selected validation improvement, not merely a latest checkpoint.
L009 | 4. Frozen 30-case capability score has no overall regression or hard category regression.
L010 | 5. SFT val NLL improves overall and no eligible language/domain/source/turn slice regresses by more than 10%.
L011 | 6. Generation health does not regress: EOS, repetition, UTF-8 and exact instruction contracts.
L012 | 7. Foundation replay loss stays inside the post-training guard.
L013 | 8. If DPO is used, human-ranked preference margin improves and DPO replay regression stays <=3%.
L014 | 9. External held-out benchmarks remain separate from training. GSM8K and MBPP are planned evaluation-only sources.
L015 | 10. Manual qualitative review is mandatory.
L016 | 
L017 | The release checker can only emit manual_release_review_required or reject. It cannot auto-release a model.
L018 | 
L019 | Local chat runtime:
L020 | - frontier350/chat_cli350.py
L021 | - exact locked tokenizer and checkpoint identity checks
L022 | - same User:/Assistant: serialization used by SFT
L023 | - token-budgeted recent-history retention
L024 | - EOS stopping
L025 | - greedy or temperature/top-p decoding
L026 | - repetition penalty
L027 | - no network calls and no conversation logging by default
L028 | 
L029 | The current generator intentionally does not claim KV-cache acceleration. Correctness and format consistency come before inference optimization.
```

**End of Part B.** To inspect real behavior, ask the owner for **selected redacted training logs and a separately agreed non-held-out sample of generation outputs**. It is not valid to infer 2B model quality from the historical 1B SFT run or mock tests. All private checkpoints and frozen test answers are intentionally omitted.
