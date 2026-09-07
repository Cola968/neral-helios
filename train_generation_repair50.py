"""Neral-50M G1: bounded generation repair from the completed B2 100M checkpoint.

G1 targets the measured B2 failure mode: raw chat generations rarely terminate and repetition
rose despite better context/code scores. The stage uses assistant-only human SFT with real EOS
supervision plus substantial B2 base-LM replay. Checkpoint selection uses a group-disjoint SFT
validation set and a small deterministic held-out free-generation dev set. No decoding penalty,
forced EOS, logit bias, SOLVE/VERIFY/Helios tool, or external pretrained model is used.

A complete frozen benchmark is mandatory after this stage. Dev metrics here are only checkpoint
selection signals and must never be reported as the final capability benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from architectures import FIFTY
from experiment import atomic_save, load_model, sha, amp
from free_generation50 import diagnostics
from generation_repair_data50 import eligible_rows, select_language_mix, selection_summary
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from train_instruct50 import (
    EXPECTED_PARAMS,
    TOKENIZER,
    assistant_loss,
    data_hashes,
    evaluate_sft,
    lr_at,
    make_sft_batch,
    next_rows,
    read_jsonl,
)
from train_raw_math50 import dev_math_nll

STAGE_VERSION=1
SEED=968_601


@torch.inference_mode()
def evaluate_free_dev(model, rows, device, max_new_tokens=64):
    was=model.training;model.eval();items=[]
    for row in rows:
        ids=TOKENIZER.encode("User: "+row["prompt"].strip()+"\nAssistant:",bos=True)
        x=torch.tensor([ids],device=device)
        with amp(device):out=model.generate(x,max_new_tokens=max_new_tokens,temperature=0,eos_id=TOKENIZER.eos_id)
        generated=out[0,len(ids):].tolist();text=TOKENIZER.decode(generated);d=diagnostics(generated,text)
        items.append({"id":row.get("id"),"language":row.get("language"),"domain":row.get("domain"),"tokens":len(generated),"diagnostics":d})
    model.train(was)
    n=max(1,len(items))
    eos=100*sum(x["diagnostics"]["eos"] for x in items)/n
    rep=100*sum(x["diagnostics"]["repeated_word_ngrams"]["3"] for x in items)/n
    return {
        "examples":len(items),
        "utf8_valid_pct":100*sum(x["diagnostics"]["valid_utf8"] for x in items)/n,
        "eos_pct":eos,
        "repeated_trigram_pct":rep,
        "mean_generated_tokens":sum(x["tokens"] for x in items)/n,
        "health_score":eos-rep,
        "note":"Held-out G1 dev only; not a frozen benchmark or promotion score.",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--init-from",required=True)
    ap.add_argument("--expected-source-sha")
    ap.add_argument("--data",default="sft50_data")
    ap.add_argument("--replay-data",default="pretrain50_b2_data")
    ap.add_argument("--out",default="runs/neral50m_g1_generation_repair")
    ap.add_argument("--steps",type=int,default=1200)
    ap.add_argument("--seq",type=int,default=768)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--replay-batch",type=int,default=6)
    ap.add_argument("--sft-weight",type=float,default=.70)
    ap.add_argument("--eos-weight",type=float,default=1.5)
    ap.add_argument("--peak-lr",type=float,default=1e-5)
    ap.add_argument("--min-lr",type=float,default=1e-6)
    ap.add_argument("--warmup-fraction",type=float,default=.05)
    ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--patience",type=int,default=5)
    ap.add_argument("--max-train-examples",type=int,default=5000)
    ap.add_argument("--max-val-examples",type=int,default=256)
    ap.add_argument("--dev-generation-examples",type=int,default=64)
    ap.add_argument("--desired-de-fraction",type=float,default=.20)
    ap.add_argument("--min-answer-bytes",type=int,default=12)
    ap.add_argument("--max-answer-bytes",type=int,default=384)
    ap.add_argument("--max-lm-regression",type=float,default=.03)
    ap.add_argument("--max-math-regression",type=float,default=.05)
    ap.add_argument("--min-health-gain",type=float,default=2.0)
    ap.add_argument("--min-sft-nll-gain",type=float,default=.005)
    args=ap.parse_args()

    positive=(args.steps,args.seq,args.batch_size,args.replay_batch,args.eval_every,args.patience,args.max_train_examples,args.max_val_examples,args.dev_generation_examples)
    if min(positive)<=0:raise ValueError("positive G1 budgets required")
    if args.seq>FIFTY.max_seq_len:raise ValueError("sequence exceeds model context")
    if not .60<=args.sft_weight<=.80:raise ValueError("G1 sft-weight must be 0.60..0.80")
    if not 1.0<=args.eos_weight<=2.0:raise ValueError("G1 eos-weight must be 1..2")
    if not 0<args.min_lr<=args.peak_lr<=2e-5:raise ValueError("G1 uses a bounded low learning rate <=2e-5")
    if not 0<=args.warmup_fraction<=.2:raise ValueError("warmup-fraction must be 0..0.2")
    if not 0<=args.desired_de_fraction<=1:raise ValueError("desired-de-fraction must be 0..1")
    if not 0<=args.max_lm_regression<=.10 or not 0<=args.max_math_regression<=.20:raise ValueError("invalid regression guards")
    if args.expected_source_sha and (len(args.expected_source_sha)!=64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_source_sha)):
        raise ValueError("expected-source-sha must be a 64-character SHA-256")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("G1 requires BF16 CUDA")

    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_float32_matmul_precision("high")
    rng=random.Random(SEED);device=torch.device("cuda");root=Path(args.data);replay_root=Path(args.replay_data)
    hashes=data_hashes(root)
    raw_train=read_jsonl(root/"train.jsonl");raw_val=read_jsonl(root/"val.jsonl")
    train_eligible=eligible_rows(raw_train,seq=args.seq,min_answer_bytes=args.min_answer_bytes,max_answer_bytes=args.max_answer_bytes)
    val_eligible=eligible_rows(raw_val,seq=args.seq,min_answer_bytes=args.min_answer_bytes,max_answer_bytes=args.max_answer_bytes)
    train=select_language_mix(train_eligible,max_examples=args.max_train_examples,desired_de_fraction=args.desired_de_fraction,min_examples=min(2500,args.max_train_examples))
    val=select_language_mix(val_eligible,max_examples=args.max_val_examples,desired_de_fraction=args.desired_de_fraction,min_examples=min(64,args.max_val_examples))
    dev_rows=select_language_mix(val,max_examples=min(args.dev_generation_examples,len(val)),desired_de_fraction=args.desired_de_fraction,min_examples=min(32,len(val)))
    train_groups={r["prompt_group_sha256"] for r in train};val_groups={r["prompt_group_sha256"] for r in val}
    if train_groups&val_groups:raise ValueError("G1 train/val normalized prompt groups overlap")
    for row in train+val:make_sft_batch([row],args.seq,torch.device("cpu"))

    replay_streams=load_b2_streams(replay_root,"train");replay_val=load_b2_streams(replay_root,"val");replay_docs=load_b2_documents(replay_root)
    replay_blocks=document_blocks(replay_docs,"de",args.seq,"train")+document_blocks(replay_docs,"en",args.seq,"train")
    if len(replay_blocks)<args.replay_batch:raise ValueError("insufficient document-bounded replay blocks")
    rng.shuffle(replay_blocks);replay_cursor=0
    replay_val_schedule,replay_val_meta=build_validation_schedule(replay_docs,args.seq,96,SEED+17)

    model,payload=load_model(args.init_from,device)
    if model.parameter_report()["trainable"]!=EXPECTED_PARAMS:raise ValueError("G1 accepts exact Neral-50M only")
    source_tokens=int(payload.get("tokens",0));source_sha=sha(args.init_from)
    if args.expected_source_sha and source_sha.lower()!=args.expected_source_sha.lower():
        raise ValueError("G1 source checkpoint SHA-256 differs from the verified B2 handoff")
    model.train()
    baseline_sft=evaluate_sft(model,val,device,args.seq)
    baseline_gen=evaluate_free_dev(model,dev_rows,device)
    baseline_lm=validate_document_bounded(model,replay_val,replay_val_schedule,args.seq,device)
    baseline_math=dev_math_nll(model,device,min(args.seq,256))

    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Use a fresh G1 output directory")
    out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),weight_decay=.03,fused=True)
    order=list(range(len(train)));rng.shuffle(order);cursor=0;step=0;selected_step=0
    best_health=baseline_gen["health_score"];best_sft=baseline_sft["macro_answer_nll"];best_gen=baseline_gen
    bad=0
    run={
        "stage":"G1 generation repair","stage_version":STAGE_VERSION,"goal_issue":36,"parameters":EXPECTED_PARAMS,
        "source_checkpoint":str(Path(args.init_from).resolve()),"source_sha256":source_sha,"source_payload_tokens":source_tokens,
        "source_contract":"Workflow independently verified completed B2 100M summary + exact selected-checkpoint SHA-256.",
        "data_hashes":hashes,"train_selection":selection_summary(train),"val_selection":selection_summary(val),"dev_selection":selection_summary(dev_rows),
        "replay_root":str(replay_root.resolve()),"replay_validation":replay_val_meta,
        "objective":{"assistant_only":True,"real_eos_target":True,"eos_weight":args.eos_weight,"sft_weight":args.sft_weight,"base_lm_replay_weight":1-args.sft_weight},
        "selection":"highest held-out free-generation health (EOS%-repeated-trigram%) with SFT improvement, <= configured LM/math regression guards",
        "decoding_hacks":{"repetition_penalty":False,"no_repeat_ngram":False,"forced_eos":False,"eos_logit_bias":False},
        "solve_used":False,"tools_used":False,
        "benchmark_policy":"Full frozen free/v3/v4/v5/RawMath/NIS benchmark immediately after G1; no further weight stage before that benchmark.",
        "baseline_sft":baseline_sft,"baseline_free_dev":baseline_gen,"baseline_lm":baseline_lm,"baseline_math":baseline_math,"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"args":vars(args),
    }
    (out/"run.json").write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding="utf8")

    train_summary=selection_summary(train)
    def checkpoint():
        return {"model":model.state_dict(),"optimizer":opt.state_dict(),"config":FIFTY.to_dict(),"tokens":source_tokens,
                "source_payload_tokens":source_tokens,"g1_version":STAGE_VERSION,"g1_step":step,"g1_selected_step":selected_step,
                "best_health_score":best_health,"best_sft_macro_answer_nll":best_sft,"source_sha256":source_sha,"sft_data_hashes":hashes,
                "train_selection_sha256":train_summary["selection_sha256"]}

    atomic_save(checkpoint(),out/"best_repair.pt");atomic_save(checkpoint(),out/"latest.pt")
    start=time.perf_counter();update_seconds=0.0;input_tokens=0;answer_tokens=0;replay_tokens=0;stop="max_steps";torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1,args.steps+1):
            rows,cursor=next_rows(train,order,cursor,args.batch_size,rng);x,y=make_sft_batch(rows,args.seq,device)
            replay_items=[]
            for _ in range(args.replay_batch):
                if replay_cursor>=len(replay_blocks):rng.shuffle(replay_blocks);replay_cursor=0
                replay_items.append(replay_blocks[replay_cursor]);replay_cursor+=1
            lx,ly=make_batch(replay_streams,replay_items,args.seq,device)
            lr=lr_at(step,args.steps,args.peak_lr,args.min_lr,args.warmup_fraction)
            for group in opt.param_groups:group["lr"]=lr
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            sft=assistant_loss(model,x,y,args.eos_weight);base=lm_loss(model,lx,ly);total=args.sft_weight*sft+(1-args.sft_weight)*base
            if not torch.isfinite(total):raise FloatingPointError("non-finite G1 loss")
            total.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-tick
            update_seconds+=dt;input_tokens+=int((x!=TOKENIZER.pad_id).sum());answer_tokens+=int((y!=-100).sum());replay_tokens+=lx.numel()
            row={"step":step,"loss":float(total),"sft_loss":float(sft.detach()),"base_replay_loss":float(base.detach()),"lr":lr,"grad_norm":float(gn),"seconds":dt,
                 "input_tokens":input_tokens,"answer_tokens":answer_tokens,"replay_tokens":replay_tokens}
            if step%args.eval_every==0 or step==args.steps:
                sft_val=evaluate_sft(model,val,device,args.seq);gen_val=evaluate_free_dev(model,dev_rows,device)
                lm_val=validate_document_bounded(model,replay_val,replay_val_schedule,args.seq,device);math_val=dev_math_nll(model,device,min(args.seq,256))
                lm_ok=lm_val["macro"]<=baseline_lm["macro"]*(1+args.max_lm_regression)
                math_ok=math_val["nll"]<=baseline_math["nll"]*(1+args.max_math_regression)
                utf8_ok=gen_val["utf8_valid_pct"]>=99.0;length_ok=gen_val["mean_generated_tokens"]>=8.0
                sft_ok=sft_val["macro_answer_nll"]<=baseline_sft["macro_answer_nll"]-args.min_sft_nll_gain
                health_improved=gen_val["health_score"]>=best_health+args.min_health_gain
                improved=lm_ok and math_ok and utf8_ok and length_ok and sft_ok and health_improved
                row.update(sft_validation=sft_val,free_generation_dev=gen_val,base_validation=lm_val,math_dev=math_val,
                           lm_guard=lm_ok,math_guard=math_ok,utf8_guard=utf8_ok,length_guard=length_ok,sft_guard=sft_ok,selected=improved)
                if improved:
                    best_health=gen_val["health_score"];best_sft=sft_val["macro_answer_nll"];best_gen=gen_val;selected_step=step;bad=0
                    atomic_save(checkpoint(),out/"best_repair.pt")
                else:bad+=1
                row["bad_evals"]=bad;atomic_save(checkpoint(),out/"latest.pt");print(json.dumps(row),flush=True)
                if bad>=args.patience:stop="early_stopping";break
            with (out/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
    finally:
        atomic_save(checkpoint(),out/"latest.pt")
        summary={
            "stage":"G1 generation repair","stop":stop,"complete":stop in {"max_steps","early_stopping"},"completed_step":step,"selected_step":selected_step,
            "effective_weight_candidate":selected_step>0,"source_payload_tokens":source_tokens,"sft_input_tokens":input_tokens,"sft_answer_tokens":answer_tokens,"base_replay_tokens":replay_tokens,
            "baseline_free_dev":baseline_gen,"selected_free_dev":best_gen,"baseline_sft_macro_answer_nll":baseline_sft["macro_answer_nll"],"best_sft_macro_answer_nll":best_sft,
            "optimizer_seconds":update_seconds,"elapsed_seconds":time.perf_counter()-start,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
            "best_sha256":sha(out/"best_repair.pt"),"latest_sha256":sha(out/"latest.pt"),"source_sha256":source_sha,"solve_used":False,"tools_used":False,
            "next_required_action":"Run complete frozen benchmark before any additional weight-changing stage.",
        }
        (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8");print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
