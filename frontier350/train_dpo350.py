from __future__ import annotations
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from chat350_contract import EXPECTED_TOKENIZER_SHA256
from chat350_preference_format import encode_final_response
from model350 import FRONTIER_350,Neral350M
from dpo_math350 import dpo_loss_value,dpo_grad_coeff

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
    return h.hexdigest()

def atomic_save(obj,path: Path):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)

def rows(path):
    out=[]
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if line.strip():out.append(json.loads(line))
    if not out:raise ValueError("empty preference data")
    return out

def logp(model,e,dev):
    x=torch.tensor([e.input_ids],dtype=torch.long,device=dev);y=torch.tensor(e.targets,dtype=torch.long,device=dev)
    logits,_=model(x);mask=y!=-100;lp=F.log_softmax(logits[0].float(),dim=-1);idx=torch.arange(y.numel(),device=dev)[mask]
    return lp[idx,y[mask]].sum()

def lr(step,total,peak,floor,warm=0.05):
    w=max(1,round(total*warm))
    if step<=w:return peak*step/w
    p=(step-w)/max(1,total-w)
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*min(1,max(0,p))))

@torch.inference_mode()
def eval_pref(model,rs,tok,refmap,max_seq,dev,beta):
    model.eval();correct=0;margins=[];losses=[]
    for r in rs:
        c=encode_final_response(tok,r["context"],r["chosen"],max_seq_len=max_seq);j=encode_final_response(tok,r["context"],r["rejected"],max_seq_len=max_seq)
        pc=float(logp(model,c,dev));pr=float(logp(model,j,dev));ref=refmap[r["id"]]
        margin=(pc-pr)-(float(ref["chosen_logp"])-float(ref["rejected_logp"]))
        margins.append(margin);correct+=int(pc>pr);losses.append(dpo_loss_value(margin,beta))
    model.train();return {"pairs":len(rs),"raw_preference_accuracy":correct/len(rs),"mean_dpo_margin":sum(margins)/len(margins),"mean_dpo_loss":sum(losses)/len(losses),"beta":beta}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--init-from",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--train",required=True);ap.add_argument("--val",required=True);ap.add_argument("--train-ref",required=True);ap.add_argument("--val-ref",required=True);ap.add_argument("--out",required=True);ap.add_argument("--steps",type=int,default=200);ap.add_argument("--max-seq",type=int,default=768);ap.add_argument("--beta",type=float,default=.1);ap.add_argument("--peak-lr",type=float,default=8e-7);ap.add_argument("--min-lr",type=float,default=8e-8);ap.add_argument("--eval-every",type=int,default=25);ap.add_argument("--seed",type=int,default=350727);ap.add_argument("--gradient-checkpointing",action="store_true");a=ap.parse_args()
    if not 0<a.beta<=1 or not 0<a.min_lr<=a.peak_lr:raise ValueError("invalid DPO hyperparameters")
    if sha256_file(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("tokenizer SHA mismatch")
    tok=Tokenizer.from_file(a.tokenizer);ck=torch.load(a.init_from,map_location="cpu",weights_only=False)
    if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
    tr,va=rows(a.train),rows(a.val)
    def loadref(path,prefpath):
        x=json.loads(Path(path).read_text(encoding="utf8"))
        if x.get("checkpoint_sha256")!=sha256_file(a.init_from) or x.get("preference_sha256")!=sha256_file(prefpath):raise ValueError("reference logprob identity mismatch")
        m={r["id"]:r for r in x["rows"]}
        if len(m)!=len(x["rows"]):raise ValueError("duplicate reference ids")
        return m
    trr=loadref(a.train_ref,a.train);var=loadref(a.val_ref,a.val)
    if {r["id"] for r in tr}!=set(trr) or {r["id"] for r in va}!=set(var):raise ValueError("reference/data id mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
    dev=torch.device("cuda");torch.manual_seed(a.seed);random.seed(a.seed)
    model=Neral350M().to(device=dev,dtype=torch.bfloat16);model.load_state_dict(ck["model"],strict=True);model.set_gradient_checkpointing(a.gradient_checkpointing);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=a.peak_lr,betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("DPO output must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    base=eval_pref(model,va,tok,var,a.max_seq,dev,a.beta);best=base["mean_dpo_margin"];best_step=0
    order=list(range(len(tr)));rng=random.Random(a.seed+1);rng.shuffle(order);cur=0;metrics=out/"metrics.jsonl";start=time.perf_counter()
    for step in range(1,a.steps+1):
        if cur>=len(order):rng.shuffle(order);cur=0
        r=tr[order[cur]];cur+=1;ref=trr[r["id"]]
        c=encode_final_response(tok,r["context"],r["chosen"],max_seq_len=a.max_seq);j=encode_final_response(tok,r["context"],r["rejected"],max_seq_len=a.max_seq)
        rate=lr(step,a.steps,a.peak_lr,a.min_lr)
        for g in opt.param_groups:g["lr"]=rate
        opt.zero_grad(set_to_none=True)
        pc=logp(model,c,dev)
        with torch.no_grad():
            pr0=logp(model,j,dev)
        refmargin=float(ref["chosen_logp"])-float(ref["rejected_logp"])
        adv_value=float(pc.detach())-float(pr0)-refmargin
        coeff=dpo_grad_coeff(adv_value,a.beta)
        (pc*coeff).backward()
        del pc,pr0
        pr=logp(model,j,dev)
        (pr*(-coeff)).backward()
        del pr
        loss_value=dpo_loss_value(adv_value,a.beta)
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        row={"step":step,"loss":loss_value,"advantage":adv_value,"lr":rate,"grad_norm":float(gn),"sequential_backward":True}
        if step%a.eval_every==0 or step==a.steps:
            ev=eval_pref(model,va,tok,var,a.max_seq,dev,a.beta);row["validation"]=ev
            if ev["mean_dpo_margin"]>best+1e-4 and ev["raw_preference_accuracy"]>=base["raw_preference_accuracy"]:
                best=ev["mean_dpo_margin"];best_step=step
                atomic_save({"model":model.state_dict(),"config":FRONTIER_350.to_dict(),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"source_checkpoint_sha256":sha256_file(a.init_from),"dpo_step":step,"validation":ev,"beta":a.beta},out/"best_dpo.pt")
        with metrics.open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
        if step%10==0 or step==a.steps:print(json.dumps(row),flush=True)
    final=eval_pref(model,va,tok,var,a.max_seq,dev,a.beta)
    summary={"baseline":base,"final":final,"best_mean_dpo_margin":best,"best_step":best_step,"best_checkpoint_exists":(out/"best_dpo.pt").is_file(),"elapsed_seconds":time.perf_counter()-start,"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"sequential_backward":True,"gradient_checkpointing":a.gradient_checkpointing}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8");print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
