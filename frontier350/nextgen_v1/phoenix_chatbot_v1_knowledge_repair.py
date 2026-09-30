"""Chatbot-v1.1 knowledge/truthfulness repair sweep from the selected behavior candidate.

Development-only. Starts from the SHA-pinned Chatbot-v1 winner, trains only the top two
transformer blocks on project-owned knowledge/truthfulness/dialogue data, and requires
truthfulness recovery plus behavior preservation before writing a local candidate.
"""
from __future__ import annotations

import argparse,json,math,os,random,sys,time
from pathlib import Path

from chatbot_v1_knowledge_data import build_encoded,exact_screen_fingerprints
from phoenix_chatbot_v1_fresh_eval import (CASES as FRESH_CASES,evaluate as fresh_evaluate,
    load_candidate,normalized,sha_file)
from phoenix_chatbot_v1_pilot import development_eval_hashes
from phoenix_auto_sft_public_dev_eval import run_model
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_master_precision import require_fp32_trainable_subset
from helios350_preview import TOKENIZER_SHA,trusted_asset
from sft_eval_contract import dynamic_batch_width,token_weighted_loss,validation_indices

BASE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
BEHAVIOR_SHA="2221a57dadb4fba137c6b1da9b07dd4e00a3f34361219fc2a9b1fbee5c3b2674"
SCHEMA="phoenix_chatbot_v1_knowledge_repair_v1"
CLASSIFICATION="project_owned_chatbot_v1_knowledge_truthfulness_repair_development_only"
PARAMETERS=349_825_152
MAX_RESERVED=9*1024**3
CONFIGS=(
    {"name":"lr2e6_s120","lr":2e-6,"steps":120},
    {"name":"lr5e6_s100","lr":5e-6,"steps":100},
    {"name":"lr8e6_s80","lr":8e-6,"steps":80},
)
TRUTHFULNESS_ONLY_CONFIGS=(
    {"name":"lr2e6_s120_truthfulness","lr":2e-6,"steps":120},
)
MAX_RESERVED=9*1024**3
PROTECT_PUBLIC=("nis2_exact_generation_pct","nis2_instruction_following_pct",
                "nis2_dialogue_state_pct","nis2_robustness_pct",
                "nis2_compositional_reasoning_pct")
PROTECT_FRESH=("exact-de-2","correct-de-1")
KNOWLEDGE_FRESH_PREFIXES=("fact","unknown","math","translate")


def atomic_save(torch,state,path:Path):
    if path.exists():raise FileExistsError("candidate overwrite refused")
    temp=path.with_suffix(path.suffix+".tmp")
    if temp.exists():raise FileExistsError("stale temp candidate")
    try:
        torch.save(state,temp)
        with temp.open("rb") as f:os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def fresh_summary(result:dict)->dict:
    rows=result["rows"]
    correctness={row["id"]:bool(row["correct"]) for row in rows}
    return {"correct":result["correct"],"total":result["total"],
            "accuracy_pct":result["accuracy_pct"],"correctness":correctness}


def exact_eval_screen()->dict:
    train_hashes=exact_screen_fingerprints()
    public=development_eval_hashes()
    if train_hashes&public:raise RuntimeError("knowledge curriculum overlaps public development prompt bytes")
    fresh_hashes=set()
    from chatbot_v1_data import norm,text_sha
    for _ident,_kind,prompt,_spec in FRESH_CASES:
        fresh_hashes.add(text_sha(norm(prompt)))
        fresh_hashes.add(text_sha(norm("User: "+prompt+"\nAssistant:")))
    if train_hashes&fresh_hashes:raise RuntimeError("knowledge curriculum overlaps fresh audit prompt bytes")
    return {"train_eval_fingerprints":len(train_hashes),"public_hashes":len(public),
            "fresh_hashes":len(fresh_hashes),"exact_overlap":0}


def repair_gate(historical_public,candidate_public,repaired_public,
                candidate_fresh,repaired_fresh,val_before,val_after,
                require_knowledge_gain:bool=True)->dict:
    public_delta={k:repaired_public[k]-candidate_public[k] for k in repaired_public
                  if isinstance(repaired_public[k],(int,float)) and isinstance(candidate_public.get(k),(int,float))}
    truthfulness_repaired=(repaired_public["nis2_truthfulness_pct"]>=historical_public["nis2_truthfulness_pct"])
    public_protected=all(public_delta.get(k,0)>=0 for k in PROTECT_PUBLIC)
    free_protected=(repaired_public["free_answer_nonempty"]>=candidate_public["free_answer_nonempty"] and
                    repaired_public["free_answer_digit_prefix"]>=candidate_public["free_answer_digit_prefix"])
    fresh_map=repaired_fresh["correctness"];candidate_map=candidate_fresh["correctness"]
    gained=[ident for ident in fresh_map if fresh_map[ident] and not candidate_map.get(ident,False)]
    lost=[ident for ident in fresh_map if candidate_map.get(ident,False) and not fresh_map[ident]]
    core_preserved=all(fresh_map.get(ident,False) for ident in PROTECT_FRESH)
    knowledge_gain=any(ident.startswith(KNOWLEDGE_FRESH_PREFIXES) for ident in gained)
    fresh_nonregression=repaired_fresh["correct"]>=candidate_fresh["correct"] and not lost
    validation_improved=val_after<val_before
    passed=all((truthfulness_repaired,public_protected,free_protected,core_preserved,
                (knowledge_gain or not require_knowledge_gain),fresh_nonregression,validation_improved))
    return {"passed":passed,"truthfulness_repaired_to_historical":truthfulness_repaired,
            "public_behavior_protected":public_protected,"free_answer_protected":free_protected,
            "fresh_core_behavior_preserved":core_preserved,"fresh_knowledge_gain":knowledge_gain,
            "fresh_nonregression":fresh_nonregression,"validation_improved":validation_improved,
            "knowledge_gain_required":require_knowledge_gain,
            "public_delta_vs_behavior_candidate":public_delta,"fresh_gained":gained,"fresh_lost":lost}


def selection_key(result):
    gate=result["gate"];d=gate["public_delta_vs_behavior_candidate"]
    return (result["fresh"]["correct"],
            d.get("nis2_truthfulness_pct",0),
            sum(d.get(k,0) for k in PROTECT_PUBLIC),
            -result["validation_loss_after"])


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoints",action="store_true")
    p.add_argument("--historical",type=Path,required=True)
    p.add_argument("--behavior-candidate",type=Path,required=True)
    p.add_argument("--behavior-sha256",default=BEHAVIOR_SHA)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--truthfulness-only",action="store_true")
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoints:p.error("explicit execution/trust required")
    if a.out.exists() or a.out.is_symlink() or not a.out.is_absolute():p.error("new absolute output required")
    historical=require_pinned_checkpoint(a.historical)
    if sha_file(historical)!=BASE_SHA:raise ValueError("historical SHA mismatch")
    if sha_file(a.behavior_candidate)!=a.behavior_sha256 or a.behavior_sha256!=BEHAVIOR_SHA:
        raise ValueError("behavior candidate SHA mismatch")
    screen=exact_eval_screen()

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
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
    splits,data_report=build_encoded(tok,int(bos),int(eos),512);data_report.update(screen)
    val_indices=validation_indices(len(splits["validation"]));device=torch.device("cuda")

    historical_state=_checked_historical_weights(historical,torch,Neral350M,FRONTIER_350.to_dict())
    historical_weights=historical_state["model"]
    behavior_state=load_candidate(torch,a.behavior_candidate,BEHAVIOR_SHA,FRONTIER_350.to_dict())
    behavior_weights=behavior_state["model"]

    def make_model(weights,dtype):
        model=Neral350M().to(device=device,dtype=dtype);model.load_state_dict(weights,strict=True)
        if model.parameter_report()["total"]!=PARAMETERS:raise RuntimeError("architecture changed")
        return model

    def public_metrics(weights):
        model=make_model(weights,torch.bfloat16).eval()
        result=run_model(torch,model,tok,int(bos),int(eos))["summary"]
        del model;torch.cuda.empty_cache();return result

    def fresh_metrics(weights):
        model=make_model(weights,torch.bfloat16).eval()
        result=fresh_summary(fresh_evaluate(model,tok,int(bos),int(eos),torch))
        del model;torch.cuda.empty_cache();return result

    def batch(rows,picks):
        selected=[rows[i] for i in picks];width=dynamic_batch_width(selected,512)
        x=torch.full((len(selected),width),int(eos),dtype=torch.long)
        y=torch.full((len(selected),width),-100,dtype=torch.long)
        for n,row in enumerate(selected):
            length=len(row["input_ids"])
            x[n,:length]=torch.tensor(row["input_ids"],dtype=torch.long)
            y[n,:length]=torch.tensor(row["labels"],dtype=torch.long)
        return x.to(device),y.to(device)

    def validation(model):
        model.eval();pairs=[]
        with torch.no_grad():
            for index in val_indices:
                x,y=batch(splits["validation"],[index])
                with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=model(x,y)
                if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite validation")
                pairs.append((float(loss.detach()),int((y!=-100).sum().item())))
        model.train();return token_weighted_loss(pairs)

    historical_public=public_metrics(historical_weights)
    candidate_public=public_metrics(behavior_weights)
    candidate_fresh=fresh_metrics(behavior_weights)
    base_val_model=make_model(behavior_weights,torch.float32);baseline_val=validation(base_val_model)
    del base_val_model;torch.cuda.empty_cache()

    a.out.mkdir(parents=True,exist_ok=False)
    results=[];best=None
    configs=TRUTHFULNESS_ONLY_CONFIGS if a.truthfulness_only else CONFIGS
    for cfg in configs:
        guard.require_safe_runtime_temperature()
        seed=20260924+11
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        model=make_model(behavior_weights,torch.float32)
        named=configure_trainable(model,2);params=[param for _name,param in named]
        optimizer=torch.optim.AdamW(params,lr=cfg["lr"],betas=(0.9,0.95),eps=1e-8,
                                    weight_decay=0.01,fused=True)
        require_fp32_trainable_subset(model,optimizer,torch,require_moments=False)
        rng=random.Random(seed);losses=[];supervised=0;started=time.monotonic();model.train()
        for step in range(1,cfg["steps"]+1):
            guard.require_safe_runtime_temperature()
            picks=[rng.randrange(len(splits["train"])) for _ in range(4)]
            x,y=batch(splits["train"],picks);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=model(x,y)
            if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite train loss")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            if not math.isfinite(float(grad)):raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            if step==1:require_fp32_trainable_subset(model,optimizer,torch,require_moments=True)
            supervised+=int((y!=-100).sum().item());losses.append(float(loss.detach()))
            if torch.cuda.memory_reserved()>MAX_RESERVED:raise RuntimeError("repair exceeds 9 GiB reserved VRAM")
            if step==1 or step%20==0 or step==cfg["steps"]:
                print(json.dumps({"event":"chatbot_v1_knowledge_step","config":cfg["name"],"step":step,
                                  "train_loss":float(loss.detach()),"supervised_tokens":supervised},sort_keys=True),flush=True)
        elapsed=time.monotonic()-started
        after_val=validation(model)
        snapshot={name:t.detach().cpu().float() for name,t in model.state_dict().items()}
        del model,optimizer;torch.cuda.empty_cache()
        repaired_public=public_metrics(snapshot);repaired_fresh=fresh_metrics(snapshot)
        gate=repair_gate(historical_public,candidate_public,repaired_public,
                         candidate_fresh,repaired_fresh,baseline_val,after_val,
                         require_knowledge_gain=not a.truthfulness_only)
        result={"config":cfg,"supervised_tokens":supervised,"trainable_blocks":2,
                "training_wall_seconds":elapsed,"mean_train_loss":sum(losses)/len(losses),
                "validation_loss_before":baseline_val,"validation_loss_after":after_val,
                "public_metrics":repaired_public,"fresh":repaired_fresh,"gate":gate,
                "candidate_file":None,"candidate_sha256":None,
                "peak_cuda_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
                "peak_cuda_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1)}
        if gate["passed"]:
            path=a.out/f"candidate_{cfg['name']}.pt"
            state={"checkpoint_schema":SCHEMA,"classification":CLASSIFICATION,"model":snapshot,
                   "config":FRONTIER_350.to_dict(),"tokenizer_sha256":TOKENIZER_SHA,
                   "historical_base_sha256":BASE_SHA,"behavior_parent_sha256":BEHAVIOR_SHA,
                   "learning_rate":cfg["lr"],"sft_steps":cfg["steps"],"sft_supervised_tokens":supervised,
                   "train_blocks":2,"dataset_report":data_report,"development_gate":gate,
                   "development_public_metrics":repaired_public,
                   "development_fresh_summary":{"correct":repaired_fresh["correct"],"total":repaired_fresh["total"]},
                   "repair_mode":"truthfulness_only" if a.truthfulness_only else "knowledge_truthfulness",
                   "production_eligible":False,"independent_holdout_checked":False,
                   "quality_gain_verified":False}
            atomic_save(torch,state,path);result["candidate_file"]=path.name;result["candidate_sha256"]=sha_file(path)
        results.append(result)
        if gate["passed"] and (best is None or selection_key(result)>selection_key(best)):best=result
        del snapshot;torch.cuda.empty_cache()

    historical_unchanged=sha_file(historical)==BASE_SHA
    behavior_unchanged=sha_file(a.behavior_candidate)==BEHAVIOR_SHA
    report={"schema":SCHEMA,"classification":CLASSIFICATION,"dataset":data_report,
            "historical_source_unchanged":historical_unchanged,"behavior_parent_unchanged":behavior_unchanged,
            "historical_public_metrics":historical_public,"behavior_public_metrics":candidate_public,
            "behavior_fresh_summary":{"correct":candidate_fresh["correct"],"total":candidate_fresh["total"]},
            "validation_loss_before":baseline_val,"results":results,
            "repair_mode":"truthfulness_only" if a.truthfulness_only else "knowledge_truthfulness",
            "development_gate_passed":best is not None,
            "selected_candidate":None if best is None else {
                "config":best["config"],"candidate_file":best["candidate_file"],
                "candidate_sha256":best["candidate_sha256"],"public_metrics":best["public_metrics"],
                "fresh":{"correct":best["fresh"]["correct"],"total":best["fresh"]["total"]},
                "gate":best["gate"]},
            "external_sft_text_used":False,"private_holdout_opened":False,
            "production_eligible":False,"quality_gain_verified":False,
            "next_action":"iterate_repair_recipe" if best is None else ("run_separate_knowledge_lane" if a.truthfulness_only else "expand_dialogue_with_reviewed_human_data_or_add_second_fresh_audit")}
    (a.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"chatbot_v1_knowledge_repair_complete","development_gate_passed":best is not None,
      "selected_candidate":None if best is None else {"config":best["config"],"sha256":best["candidate_sha256"],
        "truthfulness":best["public_metrics"]["nis2_truthfulness_pct"],"fresh_correct":best["fresh"]["correct"],
        "fresh_gained":best["gate"]["fresh_gained"],"fresh_lost":best["gate"]["fresh_lost"]},
      "historical_source_unchanged":historical_unchanged,"behavior_parent_unchanged":behavior_unchanged,
      "production_eligible":False},sort_keys=True),flush=True)
    if not historical_unchanged or not behavior_unchanged:raise RuntimeError("source checkpoint integrity changed")
    return 0 if best is not None else 2


if __name__=="__main__":
    raise SystemExit(main())
