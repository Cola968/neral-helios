"""Bounded knowledge-repair SFT sweep on the Phoenix 4Mi FP32 foundation checkpoint.

Development experiment only. Uses the pinned human-only OASST2 source already admitted by
project policy, excludes public development prompts exactly and by a conservative lexical
near-overlap screen, selects factual/explanatory DE/EN rows, mixes verified foundation replay,
and keeps a checkpoint only when raw free-answer behavior improves without protected public
development regressions.

This is intentionally NOT an independent holdout, production promotion, or long training run.
"""
from __future__ import annotations

import argparse,hashlib,json,math,os,random,re,sys,time,unicodedata
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))

from benchmark_nis2_chat50 import PAIR_CASES,EXACT_GENERATION_CASES
from fetch_oasst2_sft50 import UPSTREAM_REVISION,UPSTREAM_SHA256,choose_pairs
from helios350_preview import TOKENIZER_SHA,trusted_asset
from helios_domain_dev_probe import PUBLIC_DEV_CASES,validate_public_cases
from phoenix_auto_sft_diagnostic import _cpu_tree,_rejected,atomic_save,sha256_file
from phoenix_foundation_refresh_dev import verify_refresh_manifest
from phoenix_foundation_refresh_public_dev_eval import load_candidate,run_model,knowledge_choice_eval
from phoenix_knowledge_probe import CASES as KNOWLEDGE_CASES
from phoenix_oasst2_bridge import load_archive,local_input,select_tree_disjoint
from phoenix_master_precision import require_fp32_trainable_subset
from sft_contract import encode_row,prompt_hash
from sft_eval_contract import dynamic_batch_width,token_weighted_loss,validation_indices
from stage_oasst1_sft import frozen_chat_hashes

PARAMETERS=349_825_152
SCHEMA="phoenix_knowledge_repair_sft_dev_v1"
CLASSIFICATION="DEVELOPMENT_ONLY_HUMAN_KNOWLEDGE_REPAIR_SFT"
FOUNDATION_SHA="d54d3f752b22be09257164459c3bf56b0dd5c430db023882c4546430988a4963"
FOUNDATION_NEW_TOKENS=4_194_304
PAIR_KEYS=(
    "nis2_instruction_pair_pct","nis2_dialogue_state_pct","nis2_truthfulness_pct",
    "nis2_robustness_pct","nis2_compositional_reasoning_pct",
)
CONFIGS=(
    {"blocks":8,"lr":2e-6,"steps":40,"replay_weight":0.30},
    {"blocks":8,"lr":5e-6,"steps":40,"replay_weight":0.30},
    {"blocks":8,"lr":5e-6,"steps":60,"replay_weight":0.40},
)
TARGET_CATEGORY_CYCLE=("math","science","aviation_transport","science","math","explanation","everyday")
WORD=re.compile(r"[a-zA-ZÀ-ÿ0-9]+",re.UNICODE)
SCIENCE=("physics","physik","chemistry","chemie","biology","biologie","astronomy","astronomie",
         "planet","star","stern","atom","molecule","molekül","gravity","gravitation","energy","energie",
         "science","wissenschaft","electric","elektr","temperature","temperatur")
MATH=("math","mathe","mathemat","calculate","berech","equation","gleichung","percent","prozent",
      "fraction","bruch","geometry","geometr","algebra","probability","wahrscheinlichkeit")
TRANSPORT=("aviation","aircraft","airplane","airport","flight","airline","luftfahrt","flugzeug","flughafen",
           "train","railway","railroad","bahn","zug","verkehr","transport")
FACT=("capital","hauptstadt","country","land","population","bevölkerung","history","geschichte","definition",
      "meaning","bedeutet","year","jahr","currency","währung","language","sprache")
EXPLAIN=("explain","why","how does","how do","describe","erklär","warum","wie funktioniert","beschreib")
QUESTION_START=("what ","who ","where ","when ","which ","how ","was ","wer ","wo ","wann ","welch ","wie ")
NEGATIVE=("write a poem","schreibe ein gedicht","write a story","schreibe eine geschichte","roleplay","rollenspiel",
          "your opinion","deine meinung","relationship advice","beziehungsrat","tell me a joke","erzähl einen witz")


def norm(text:str)->str:
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())

def words(text:str)->set[str]:
    return {m.group(0).casefold() for m in WORD.finditer(unicodedata.normalize("NFKC",text)) if len(m.group(0))>1}

def near_public(prompt:str,public_prompts:tuple[str,...])->bool:
    p=norm(prompt);pw=words(prompt)
    for raw in public_prompts:
        e=norm(raw)
        if p==e or (len(e)>=12 and (e in p or p in e)):return True
        ew=words(raw)
        union=pw|ew;inter=pw&ew
        if len(inter)>=3 and union and len(inter)/len(union)>=0.72:return True
    return False

def knowledge_score(prompt:str,answer:str)->tuple[int,str]:
    p=norm(prompt);score=0;category="everyday"
    if any(x in p for x in NEGATIVE):return -10,"excluded"
    if p.startswith(QUESTION_START) or "?" in prompt:score+=2
    if any(x in p for x in MATH) or re.search(r"\d\s*[-+*/=×÷]\s*\d",p):
        score+=4;category="math"
    if any(x in p for x in SCIENCE):
        score+=4;category="science"
    if any(x in p for x in TRANSPORT):
        score+=4;category="aviation_transport"
    if any(x in p for x in FACT):
        score+=3;category="everyday"
    if any(x in p for x in EXPLAIN):
        score+=2
        if category=="everyday":category="explanation"
    if 20<=len(answer.strip())<=900:score+=1
    if len(prompt)>600 or len(answer)>1600:score-=2
    return score,category

def public_prompts()->tuple[str,...]:
    validate_public_cases();out=[]
    out.extend(row[3] for row in PUBLIC_DEV_CASES)
    out.extend(row[1] for row in KNOWLEDGE_CASES)
    for _,prefix,_,_ in PAIR_CASES:
        if "User: " in prefix and "\nAssistant:" in prefix:
            out.append(prefix.rsplit("User: ",1)[1].split("\nAssistant:",1)[0])
        out.append(prefix)
    out.extend(row[1] for row in EXACT_GENERATION_CASES)
    return tuple(out)

def _balanced(rows:list[dict],cap:int,min_de:int)->list[dict]:
    by={"de":[],"en":[]}
    for row in rows:by[row["_lang"]].append(row)
    for lang in by:by[lang].sort(key=lambda r:(-r["_score"],r["id"]))
    de_take=min(len(by["de"]),max(min_de,cap//3))
    selected=by["de"][:de_take]+by["en"][:max(0,cap-de_take)]
    used={r["id"] for r in selected}
    if len(selected)<cap:
        rest=sorted((r for r in rows if r["id"] not in used),key=lambda r:(-r["_score"],r["id"]))
        selected.extend(rest[:cap-len(selected)])
    return sorted(selected,key=lambda r:r["id"])

def prepare_knowledge(repo_root:Path,archive:Path,private_root:Path,tokenizer_path:Path):
    frozen=frozen_chat_hashes(repo_root/"benchmark_nis2_chat50.py")
    prompts=public_prompts()
    for text in prompts:frozen.add(prompt_hash(text))
    messages=load_archive(local_input(archive,private_root))
    pairs=choose_pairs(messages,max_en=1600,max_de=800)
    lang_by_id={row["id"]:lang for lang in ("de","en") for row in pairs[lang]}
    raw,source_summary=select_tree_disjoint(pairs,messages,frozen)

    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path))
    if tok.get_vocab_size()!=16_384:raise ValueError("locked tokenizer vocabulary changed")
    bos,eos=tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos==eos:raise ValueError("invalid BOS/EOS")

    packed={};category_counts={};rejects={}
    for split,cap,min_de in (("train",900,30),("validation",160,6)):
        candidates=[]
        for row in raw[split]:
            lang=lang_by_id.get(row["id"])
            if lang not in {"de","en"}:continue
            reason=_rejected(row["user"]) or _rejected(row["assistant"])
            if reason:
                rejects[reason]=rejects.get(reason,0)+1;continue
            if near_public(row["user"],prompts):
                rejects["public_dev_near_overlap"]=rejects.get("public_dev_near_overlap",0)+1;continue
            score,category=knowledge_score(row["user"],row["assistant"])
            if score<3:
                rejects["low_knowledge_score"]=rejects.get("low_knowledge_score",0)+1;continue
            item={**row,"_lang":lang,"_score":score,"_category":category};candidates.append(item)
        selected=_balanced(candidates,cap,min_de)
        de=sum(r["_lang"]=="de" for r in selected)
        if len(selected)<(180 if split=="train" else 24) or de<min_de:
            raise RuntimeError(f"insufficient knowledge-focused {split} rows after screening")
        packed[split]=[]
        counts={}
        for row in selected:
            try:encoded=encode_row({k:row[k] for k in ("id","source_id","user","assistant")},
                                   tok,int(bos),int(eos),16_384,512)
            except ValueError:
                rejects["sequence_or_token_contract"]=rejects.get("sequence_or_token_contract",0)+1;continue
            encoded["_category"]=row["_category"]
            packed[split].append(encoded);counts[row["_category"]]=counts.get(row["_category"],0)+1
        category_counts[split]=counts
    if len(packed["train"])<160 or len(packed["validation"])<20:raise RuntimeError("too little packed knowledge data")
    return tok,int(bos),int(eos),packed,{
        "classification":"auto_screened_human_oasst2_knowledge_repair_development_only",
        "source_revision":UPSTREAM_REVISION,"source_archive_sha256":UPSTREAM_SHA256,
        "source_summary":source_summary,"packed_rows":{k:len(v) for k,v in packed.items()},
        "category_counts":category_counts,"rejected":rejects,
        "public_dev_prompts_screened":len(prompts),"exact_hash_registry_size":len(frozen),
        "near_overlap_method":"normalized containment + token-Jaccard>=0.72 with >=3 shared tokens",
        "human_source_only":True,"independent_holdout_checked":False,"production_eligible":False,
    }

def candidate_qualifies(base:dict,cand:dict,base_knowledge:dict,cand_knowledge:dict,
                        before_loss:float,after_loss:float)->bool:
    free_gain=(cand["free_answer_strict"]>base["free_answer_strict"] or
               cand["free_answer_digit_prefix"]>base["free_answer_digit_prefix"])
    protected=(cand["nis2_pair_overall_pct"]>=base["nis2_pair_overall_pct"] and
               cand["nis2_exact_generation_pct"]>=base["nis2_exact_generation_pct"] and
               all(cand[k]>=base[k] for k in PAIR_KEYS) and
               cand_knowledge["chat"]["accuracy_pct"]>=base_knowledge["chat"]["accuracy_pct"] and
               cand_knowledge["qa"]["accuracy_pct"]>=base_knowledge["qa"]["accuracy_pct"])
    return bool(free_gain and protected and after_loss<before_loss)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--assets-root",type=Path,required=True);p.add_argument("--private-root",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True);p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:p.error("explicit own-checkpoint opt-in required")
    if a.checkpoint_sha256!=FOUNDATION_SHA or sha256_file(a.checkpoint)!=FOUNDATION_SHA:raise ValueError("4Mi foundation checkpoint identity mismatch")
    if a.out.exists() or a.out.is_symlink() or not a.out.is_absolute():p.error("new absolute output required")

    assets=a.assets_root.resolve(strict=True);private=a.private_root.resolve(strict=True)
    tokenizer_path=trusted_asset(assets,assets/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    tok,bos,eos,splits,data_report=prepare_knowledge(REPO_ROOT,a.archive,private,tokenizer_path)
    refresh=verify_refresh_manifest(a.corpus_root,a.manifest_sha256,TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np,torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    from protection_v9 import configure_trainable
    from train350_compiled import PersistentMMapPool
    guard.require_cuda_ready(torch)

    state=load_candidate(torch,a.checkpoint,FOUNDATION_SHA,FRONTIER_350.to_dict())
    if state.get("processed_tokens")!=FOUNDATION_NEW_TOKENS:raise ValueError("expected exact 4Mi foundation boundary")
    base_weights=state["model"]
    base_meta={"foundation_new_tokens":state["processed_tokens"],"foundation_step":state["step"],
               "foundation_manifest_sha256":state["refresh_manifest_sha256"]}
    del state

    device=torch.device("cuda");torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision("high")
    replay=PersistentMMapPool(refresh["root"]/"train",FRONTIER_350.vocab_size)
    val_indices=validation_indices(len(splits["validation"]))

    def fresh_model():
        m=Neral350M().to(device=device,dtype=torch.float32);m.load_state_dict(base_weights,strict=True)
        if m.parameter_report()["total"]!=PARAMETERS:raise RuntimeError("parameter contract changed")
        return m
    def batch(rows,picks):
        selected=[rows[i] for i in picks];width=dynamic_batch_width(selected,512)
        x=torch.full((len(selected),width),eos,dtype=torch.long);y=torch.full((len(selected),width),-100,dtype=torch.long)
        for i,row in enumerate(selected):
            n=len(row["input_ids"]);x[i,:n]=torch.tensor(row["input_ids"]);y[i,:n]=torch.tensor(row["labels"])
        return x.to(device),y.to(device)
    def stratified_picks(rows,rng,step,batch_size=4):
        buckets={}
        for i,row in enumerate(rows):buckets.setdefault(row.get("_category","everyday"),[]).append(i)
        picks=[]
        for offset in range(batch_size):
            wanted=TARGET_CATEGORY_CYCLE[((step-1)*batch_size+offset)%len(TARGET_CATEGORY_CYCLE)]
            pool=buckets.get(wanted) or list(range(len(rows)))
            picks.append(pool[rng.randrange(len(pool))])
        return picks
    def validation(m):
        m.eval();vals=[]
        with torch.no_grad():
            for idx in val_indices:
                x,y=batch(splits["validation"],[idx])
                with torch.autocast("cuda",dtype=torch.bfloat16):_,loss=m(x,y)
                if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite knowledge validation")
                vals.append((float(loss),int((y!=-100).sum().item())))
        m.train();return token_weighted_loss(vals)
    def dev(m):
        m.eval();result=run_model(torch,m,tok,bos,eos);knowledge=knowledge_choice_eval(torch,m,tok,bos);m.train()
        return result["summary"],knowledge

    base=fresh_model();before=validation(base);base_summary,base_knowledge=dev(base);del base;torch.cuda.empty_cache()
    results=[];best=None;best_key=None
    for index,cfg in enumerate(CONFIGS):
        guard.require_safe_runtime_temperature()
        seed=20260925;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
        sft_rng=random.Random(seed+index);replay_rng=random.Random(seed+100+index);torch.cuda.reset_peak_memory_stats()
        m=fresh_model();named=configure_trainable(m,cfg["blocks"]);params=[p for _,p in named]
        opt=torch.optim.AdamW(params,lr=cfg["lr"],betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
        require_fp32_trainable_subset(m,opt,torch,require_moments=False)
        supervised=replay_tokens=0;started=time.monotonic();m.train()
        for step in range(1,cfg["steps"]+1):
            picks=stratified_picks(splits["train"],sft_rng,step,4)
            sx,sy=batch(splits["train"],picks);rx,ry=replay.batch(1,512,device,replay_rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):_,sloss=m(sx,sy);_,rloss=m(rx,ry)
            if sloss is None or rloss is None or not bool(torch.isfinite(sloss)) or not bool(torch.isfinite(rloss)):
                raise FloatingPointError("nonfinite knowledge/replay loss")
            w=cfg["replay_weight"];((sloss+w*rloss)/(1.0+w)).backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            if not math.isfinite(float(grad)):raise FloatingPointError("nonfinite gradient")
            opt.step()
            if step==1:require_fp32_trainable_subset(m,opt,torch,require_moments=True)
            supervised+=int((sy!=-100).sum().item());replay_tokens+=512
            if step%20==0:torch.cuda.synchronize();guard.require_safe_runtime_temperature()
        elapsed=time.monotonic()-started
        after=validation(m);summary,knowledge=dev(m)
        qualifies=candidate_qualifies(base_summary,summary,base_knowledge,knowledge,before,after)
        row={"index":index,**cfg,"validation_loss_before":before,"validation_loss_after":after,
             "supervised_answer_tokens":supervised,"foundation_replay_tokens":replay_tokens,
             "sampling":"category_stratified_v2",
             "training_wall_seconds":round(elapsed,3),"peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1048576,1),
             "summary":summary,"knowledge_choice":{k:v["accuracy_pct"] for k,v in knowledge.items()},"qualifies":qualifies}
        results.append(row);print(json.dumps({"event":"knowledge_repair_candidate",**row},sort_keys=True),flush=True)
        key=(summary["free_answer_strict"],summary["free_answer_digit_prefix"],knowledge["qa"]["accuracy_pct"],
             knowledge["chat"]["accuracy_pct"],summary["nis2_pair_overall_pct"],-after)
        if qualifies and (best_key is None or key>best_key):
            best_key=key;best={"selected":row,
                "model":{n:t.detach().cpu().float() for n,t in m.state_dict().items()},
                "optimizer":_cpu_tree(torch,opt.state_dict()),"trainable_names":[n for n,_ in named]}
        del opt,params,named,m;torch.cuda.empty_cache()

    a.out.mkdir(parents=True,exist_ok=False);candidate_sha=None;selected=None
    if best is not None:
        selected=best["selected"]
        cp=a.out/"candidate_knowledge_repair.pt"
        payload={"checkpoint_schema":SCHEMA,"classification":CLASSIFICATION,"model":best["model"],
                 "optimizer":best["optimizer"],"config":FRONTIER_350.to_dict(),"tokenizer_sha256":TOKENIZER_SHA,
                 "base_checkpoint_sha256":FOUNDATION_SHA,**base_meta,"selected":selected,
                 "source_dataset":"OpenAssistant/oasst2","source_revision":UPSTREAM_REVISION,
                 "source_archive_sha256":UPSTREAM_SHA256,"data_report":data_report,
                 "trainable_names":best["trainable_names"],"master_parameter_dtype":"float32",
                 "adamw_moment_dtype":"float32","compute_autocast_dtype":"bfloat16",
                 "independent_holdout_checked":False,"quality_gain_verified":False,"production_eligible":False}
        atomic_save(torch,payload,cp);candidate_sha=sha256_file(cp)
    report={"event":"phoenix_knowledge_repair_sft_finished","classification":CLASSIFICATION,
            "base_checkpoint_sha256":FOUNDATION_SHA,"base_summary":base_summary,
            "base_knowledge_choice":{k:v["accuracy_pct"] for k,v in base_knowledge.items()},
            "knowledge_validation_loss_before":before,"data_report":data_report,"configs":results,
            "qualified_count":sum(r["qualifies"] for r in results),"selected":selected,
            "candidate_checkpoint_written":candidate_sha is not None,"candidate_sha256":candidate_sha,
            "independent_holdout_checked":False,"quality_gain_verified":False,"production_eligible":False}
    (a.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"event":report["event"],"base_summary":base_summary,
                      "base_knowledge_choice":report["base_knowledge_choice"],
                      "qualified_count":report["qualified_count"],"selected":selected,
                      "candidate_sha256":candidate_sha,"production_eligible":False},sort_keys=True),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
