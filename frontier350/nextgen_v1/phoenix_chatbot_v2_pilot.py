"""Conservative Chatbot-v2 behavior sweep from the immutable Arctus 350M anchor.

Purpose: fix Chatbot-v1 overfitting observed on the fresh 24-case challenge. Only the top
2 transformer blocks are trainable, learning rates are much lower, dialogue/concise examples
are sampled more heavily, and the fresh challenge is a mandatory candidate gate.
"""
from __future__ import annotations

import argparse,json,math,os,random,sys,time
from collections import defaultdict
from pathlib import Path

from chatbot_v1_data import build_encoded
from chatbot_v1_fresh_challenge import evaluate as fresh_evaluate
from helios350_preview import TOKENIZER_SHA,trusted_asset
from phoenix_auto_sft_public_dev_eval import run_model,sha256_file
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_master_precision import require_fp32_trainable_subset
from sft_eval_contract import dynamic_batch_width,token_weighted_loss,validation_indices

HISTORICAL_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
SCHEMA="phoenix_chatbot_v2_conservative_behavior_v1"
CLASSIFICATION="project_owned_chatbot_v2_conservative_behavior_development_only"
PARAMETERS=349_825_152
MAX_RESERVED=9*1024**3
CONFIGS=(
    {"name":"top2_lr5e7_s160","lr":5e-7,"steps":160},
    {"name":"top2_lr1e6_s140","lr":1e-6,"steps":140},
    {"name":"top2_lr2e6_s100","lr":2e-6,"steps":100},
)
CATEGORY_WEIGHTS={
    "multi_turn_state":8,
    "multi_turn_correction":8,
    "short_explanation":7,
    "uncertainty":5,
    "format_control":4,
    "clarification":3,
    "exact_instruction":2,
    "bilingual":2,
    "identity":1,
    "arithmetic":1,
}
PROTECTED_PUBLIC=("nis2_dialogue_state_pct","nis2_truthfulness_pct","nis2_robustness_pct",
                  "nis2_compositional_reasoning_pct")
PROTECTED_FRESH=("dialogue","concise","truthfulness","knowledge","math","bilingual")


def atomic_save(torch,state,path:Path):
    if path.exists():raise FileExistsError("candidate overwrite refused")
    temp=path.with_suffix(path.suffix+".tmp")
    try:
        torch.save(state,temp)
        with temp.open("rb") as f:os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def fresh_summary(result:dict)->dict:
    return {"correct":result["correct"],"total":result["total"],"accuracy_pct":result["accuracy_pct"],
            "categories":{k:v["accuracy_pct"] for k,v in result["categories"].items()}}


def gate(baseline_public,candidate_public,baseline_fresh,candidate_fresh,val_before,val_after):
    public_delta={k:candidate_public[k]-baseline_public[k] for k in baseline_public
                  if isinstance(baseline_public[k],(int,float)) and isinstance(candidate_public.get(k),(int,float))}
    fresh_delta={k:candidate_fresh["categories"][k]-baseline_fresh["categories"][k]
                 for k in baseline_fresh["categories"]}
    public_protected=all(public_delta.get(k,0)>=0 for k in PROTECTED_PUBLIC)
    free_protected=(candidate_public["free_answer_nonempty"]>=baseline_public["free_answer_nonempty"]-1 and
                    candidate_public["free_answer_digit_prefix"]>=baseline_public["free_answer_digit_prefix"])
    fresh_protected=all(fresh_delta.get(k,0)>=0 for k in PROTECTED_FRESH)
    fresh_total_nonregression=candidate_fresh["correct"]>=baseline_fresh["correct"]
    fresh_behavior_gain=(candidate_fresh["correct"]>baseline_fresh["correct"] or
                         any(fresh_delta.get(k,0)>0 for k in ("exact","format","dialogue","concise")))
    public_behavior_gain=(public_delta.get("nis2_exact_generation_pct",0)>0 or
                          public_delta.get("nis2_pair_overall_pct",0)>0)
    validation_improved=val_after<val_before
    passed=all((public_protected,free_protected,fresh_protected,fresh_total_nonregression,
                fresh_behavior_gain,public_behavior_gain,validation_improved))
    return {"passed":passed,"public_protected":public_protected,"free_answer_protected":free_protected,
            "fresh_categories_protected":fresh_protected,"fresh_total_nonregression":fresh_total_nonregression,
            "fresh_behavior_gain":fresh_behavior_gain,"public_behavior_gain":public_behavior_gain,
            "validation_improved":validation_improved,"public_delta":public_delta,"fresh_delta":fresh_delta}


def selection_key(row):
    g=row["gate"]
    return (row["fresh"]["correct"],
            g["public_delta"].get("nis2_exact_generation_pct",0),
            g["public_delta"].get("nis2_pair_overall_pct",0),
            -row["validation_loss_after"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:p.error("explicit own-checkpoint execution required")
    if a.out.exists() or a.out.is_symlink() or not a.out.is_absolute():p.error("new absolute output required")
    source=require_pinned_checkpoint(a.checkpoint)
    if sha256_file(source)!=HISTORICAL_SHA:raise ValueError("historical source SHA changed")

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np,torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from protection_v9 import configure_trainable
    guard.require_cuda_ready(torch)

    assets=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(assets,assets/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    tok=Tokenizer.from_file(str(tok_path));bos=tok.token_to_id("<|bos|>");eos=tok.token_to_id("<|eos|>")
    if tok.get_vocab_size()!=FRONTIER_350.vocab_size or bos is None or eos is None:raise ValueError("tokenizer mismatch")
    splits,data_report=build_encoded(tok,int(bos),int(eos),512)
    device=torch.device("cuda");val_indices=validation_indices(len(splits["validation"]))
    historical=_checked_historical_weights(source,torch,Neral350M,FRONTIER_350.to_dict());base_weights=historical["model"]

    category_rows=defaultdict(list)
    for i,row in enumerate(splits["train"]):category_rows[row["category"]].append(i)
    if set(CATEGORY_WEIGHTS)-set(category_rows):raise RuntimeError("missing weighted behavior category")
    weighted_categories=list(CATEGORY_WEIGHTS)
    weights=[CATEGORY_WEIGHTS[x] for x in weighted_categories]

    def make_model(dtype):
        m=Neral350M().to(device=device,dtype=dtype);m.load_state_dict(base_weights,strict=True)
        if m.parameter_report()["total"]!=PARAMETERS:raise RuntimeError("architecture changed")
        return m

    def batch(rows,picks):
        selected=[rows[i] for i in picks];width=dynamic_batch_width(selected,512)
        x=torch.full((len(selected),width),int(eos),dtype=torch.long)
        y=torch.full((len(selected),width),-100,dtype=torch.long)
        for n,row in enumerate(selected):
            length=len(row["input_ids"]);x[n,:length]=torch.tensor(row["input_ids"],dtype=torch.long)
            y[n,:length]=torch.tensor(row["labels"],dtype=torch.long)
        return x.to(device),y.to(device)

    def validation(model):
        model.eval();pairs=[]
        with torch.no_grad():
            for idx in val_indices:
                x,y=batch(splits["validation"],[idx])
                with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=model(x,y)
                if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite validation")
                pairs.append((float(loss.detach()),int((y!=-100).sum().item())))
        model.train();return token_weighted_loss(pairs)

    def public_eval(weights):
        m=Neral350M().to(device=device,dtype=torch.bfloat16);m.load_state_dict(weights,strict=True);m.eval()
        result=run_model(torch,m,tok,int(bos),int(eos))["summary"];del m;torch.cuda.empty_cache();return result

    def fresh_eval(weights):
        m=Neral350M().to(device=device,dtype=torch.bfloat16);m.load_state_dict(weights,strict=True);m.eval()
        result=fresh_summary(fresh_evaluate(m,tok,int(bos),int(eos),torch));del m;torch.cuda.empty_cache();return result

    baseline_public=public_eval(base_weights);baseline_fresh=fresh_eval(base_weights)
    vm=make_model(torch.float32);baseline_val=validation(vm);del vm;torch.cuda.empty_cache()

    a.out.mkdir(parents=True,exist_ok=False)
    results=[];best=None
    for cfg in CONFIGS:
        guard.require_safe_runtime_temperature()
        seed=20260924+22
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=random.Random(seed);torch.cuda.reset_peak_memory_stats()
        model=make_model(torch.float32);named=configure_trainable(model,2);params=[p for _,p in named]
        opt=torch.optim.AdamW(params,lr=cfg["lr"],betas=(0.9,0.95),eps=1e-8,weight_decay=0.01,fused=True)
        require_fp32_trainable_subset(model,opt,torch,require_moments=False)
        supervised=0;losses=[];started=time.monotonic();model.train()
        for step in range(1,cfg["steps"]+1):
            cats=rng.choices(weighted_categories,weights=weights,k=4)
            picks=[rng.choice(category_rows[cat]) for cat in cats]
            x,y=batch(splits["train"],picks);opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=model(x,y)
            if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite train loss")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            if not math.isfinite(float(grad)):raise FloatingPointError("nonfinite gradient")
            opt.step()
            if step==1:require_fp32_trainable_subset(model,opt,torch,require_moments=True)
            supervised+=int((y!=-100).sum().item());losses.append(float(loss.detach()))
            if torch.cuda.memory_reserved()>MAX_RESERVED:raise RuntimeError("Chatbot-v2 exceeds 9 GiB reserved VRAM")
            if step==1 or step%40==0 or step==cfg["steps"]:
                print(json.dumps({"event":"chatbot_v2_step","config":cfg["name"],"step":step,
                                  "loss":float(loss.detach()),"supervised_tokens":supervised},sort_keys=True),flush=True)
        elapsed=time.monotonic()-started;after_val=validation(model)
        snapshot={n:t.detach().cpu().float() for n,t in model.state_dict().items()}
        del model,opt;torch.cuda.empty_cache()
        candidate_public=public_eval(snapshot);candidate_fresh=fresh_eval(snapshot)
        g=gate(baseline_public,candidate_public,baseline_fresh,candidate_fresh,baseline_val,after_val)
        row={"config":cfg,"train_blocks":2,"supervised_tokens":supervised,
             "training_wall_seconds":elapsed,"mean_train_loss":sum(losses)/len(losses),
             "validation_loss_before":baseline_val,"validation_loss_after":after_val,
             "public":candidate_public,"fresh":candidate_fresh,"gate":g,
             "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
             "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
             "candidate_file":None,"candidate_sha256":None}
        if g["passed"]:
            path=a.out/f"candidate_{cfg['name']}.pt"
            state={"checkpoint_schema":SCHEMA,"classification":CLASSIFICATION,"model":snapshot,
                   "config":FRONTIER_350.to_dict(),"tokenizer_sha256":TOKENIZER_SHA,
                   "base_checkpoint_sha256":HISTORICAL_SHA,"learning_rate":cfg["lr"],
                   "sft_steps":cfg["steps"],"train_blocks":2,"sft_supervised_tokens":supervised,
                   "dataset_report":data_report,"development_gate":g,
                   "public_metrics":candidate_public,"fresh_summary":candidate_fresh,
                   "production_eligible":False,"independent_holdout_checked":False,
                   "quality_gain_verified":False}
            atomic_save(torch,state,path);row["candidate_file"]=path.name;row["candidate_sha256"]=sha256_file(path)
        results.append(row)
        if g["passed"] and (best is None or selection_key(row)>selection_key(best)):best=row
        del snapshot;torch.cuda.empty_cache()

    unchanged=sha256_file(source)==HISTORICAL_SHA
    report={"schema":SCHEMA,"classification":CLASSIFICATION,"dataset":data_report,
            "historical_source_unchanged":unchanged,"baseline":{"validation_loss":baseline_val,
            "public":baseline_public,"fresh":baseline_fresh},"results":results,
            "development_gate_passed":best is not None,
            "selected_candidate":None if best is None else {"config":best["config"],
              "candidate_file":best["candidate_file"],"candidate_sha256":best["candidate_sha256"],
              "public":best["public"],"fresh":best["fresh"],"gate":best["gate"]},
            "external_sft_text_used":False,"private_holdout_opened":False,
            "production_eligible":False,"quality_gain_verified":False,
            "next_action":"iterate_behavior_v2" if best is None else "truthfulness_then_knowledge_lane"}
    (a.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"chatbot_v2_complete","development_gate_passed":best is not None,
      "selected_candidate":None if best is None else {"config":best["config"],"sha256":best["candidate_sha256"],
      "fresh_correct":best["fresh"]["correct"],"public_delta":best["gate"]["public_delta"],
      "fresh_delta":best["gate"]["fresh_delta"]},"historical_source_unchanged":unchanged,
      "production_eligible":False},sort_keys=True),flush=True)
    if not unchanged:raise RuntimeError("historical source changed")
    return 0 if best is not None else 2

if __name__=="__main__":raise SystemExit(main())
