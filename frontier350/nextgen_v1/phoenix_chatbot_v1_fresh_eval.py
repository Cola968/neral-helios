"""Fresh post-training generation audit for Chatbot-v1 baseline vs selected candidate.

Cases in this file were authored after the first Chatbot-v1 candidate existed. They were not
used by the behavior curriculum or candidate selection. This remains development evidence,
not an independent sealed holdout. Raw generations stay on the owner's private disk.
"""
from __future__ import annotations

import argparse,hashlib,json,re,sys
from pathlib import Path

from helios350_preview import TOKENIZER_SHA,generate,trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_weight_only_pilot import _checked_historical_weights

BASE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CANDIDATE_SCHEMA="phoenix_chatbot_v1_behavior_pilot_v1"
CANDIDATE_CLASS="project_owned_chatbot_v1_behavior_development_only"
PARAMETERS=349_825_152

CASES=(
 ("exact-de-1","exact","Antworte ausschließlich mit dem Wort NEBEL.","NEBEL"),
 ("exact-de-2","exact","Gib nur diese Zeichenfolge aus: Z-418","Z-418"),
 ("exact-en-1","exact","Output only the lowercase word violet.","violet"),
 ("exact-en-2","exact","Reply with exactly: READY-27","READY-27"),
 ("math-de-1","exact","Berechne 47 + 28. Antworte nur mit der Zahl.","75"),
 ("math-de-2","exact","Berechne 9 mal 13. Antworte nur mit der Zahl.","117"),
 ("math-en-1","exact","Calculate 84 - 39. Output only the number.","45"),
 ("math-en-2","exact","Calculate 12 times 8. Output only the number.","96"),
 ("state-de-1","contains","Für dieses Gespräch heißt mein Satellit LUMEN-73.\nAssistant: Verstanden.\nUser: Wie heißt mein Satellit?",("lumen-73",)),
 ("state-en-1","contains","For this conversation the project code is RIVER-204.\nAssistant: Understood.\nUser: What is the project code?",("river-204",)),
 ("correct-de-1","contains_not","Merke dir zunächst ALFA-31 als Kennwort.\nAssistant: Verstanden.\nUser: Korrektur: Das Kennwort ist jetzt BETA-92.\nAssistant: Verstanden.\nUser: Wie lautet das aktuelle Kennwort?",(("beta-92",),("alfa-31",))),
 ("correct-en-1","contains_not","First remember room D17.\nAssistant: Understood.\nUser: Correction: the room is now K42.\nAssistant: Understood.\nUser: What is the current room?",(("k42",),("d17",))),
 ("unknown-de-1","uncertain","Die erfundene Insel Vareli wurde gerade genannt, aber es wurden keine Flächendaten angegeben. Wie groß ist sie?",("weiß","nicht","bestimm","keine")),
 ("unknown-en-1","uncertain","The fictional moon Tavon was just named, but no mass was provided. What is its exact mass?",("cannot","don't know","not enough","not provided","unknown")),
 ("fact-de-1","contains","Ist Saturn ein Stern? Antworte kurz.",("nein",)),
 ("fact-en-1","contains","Is the Moon a planet? Answer briefly.",("no",)),
 ("fact-de-2","contains","Welche Kraft zieht Körper zur Erde? Antworte mit einem kurzen Begriff.",("grav",)),
 ("fact-en-2","contains","What gas do humans normally need to breathe? Answer briefly.",("oxygen",)),
 ("translate-de-1","exact","Übersetze ins Englische und gib nur die Übersetzung aus: Der Himmel ist klar.","The sky is clear."),
 ("translate-en-1","exact","Translate into German and output only the translation: The door is open.","Die Tür ist offen."),
 ("format-de-1","two_bullets","Nenne genau zwei kurze Vorteile regelmäßiger Pausen beim Lernen.",None),
 ("format-en-1","two_bullets","Give exactly two short reasons to write tests for code.",None),
 ("identity-de-1","contains","Wer bist du? Antworte in einem kurzen Satz.",("helios",)),
 ("identity-en-1","contains","Who are you? Answer in one short sentence.",("helios",)),
)


def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def suite_sha()->str:
    return hashlib.sha256(json.dumps(CASES,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def normalized(text:str)->str:
    return " ".join(text.casefold().strip().split())


def score(kind:str,answer:str,spec)->bool:
    clean=normalized(answer)
    if kind=="exact":return clean==normalized(spec)
    if kind=="contains":return all(token.casefold() in clean for token in spec)
    if kind=="contains_not":
        required,forbidden=spec
        return all(x.casefold() in clean for x in required) and all(x.casefold() not in clean for x in forbidden)
    if kind=="uncertain":return any(token.casefold() in clean for token in spec)
    if kind=="two_bullets":
        lines=[line.strip() for line in answer.splitlines() if line.strip()]
        return len(lines)==2 and all(line.startswith(("- ","• ")) for line in lines)
    raise ValueError("unknown scorer")


def load_candidate(torch,path:Path,expected_sha:str,config:dict)->dict:
    if sha_file(path)!=expected_sha:raise ValueError("candidate SHA mismatch")
    state=torch.load(path,map_location="cpu",weights_only=True)
    if (not isinstance(state,dict) or state.get("checkpoint_schema")!=CANDIDATE_SCHEMA or
        state.get("classification")!=CANDIDATE_CLASS or
        state.get("base_checkpoint_sha256")!=BASE_SHA or state.get("config")!=config or
        state.get("tokenizer_sha256")!=TOKENIZER_SHA or state.get("production_eligible") is not False or
        not isinstance(state.get("model"),dict)):
        raise ValueError("candidate metadata contract mismatch")
    return state


def evaluate(model,tok,bos,eos,torch)->dict:
    rows=[]
    for ident,kind,prompt,spec in CASES:
        answer,used=generate(model,tok,bos,eos,torch,prompt,48)
        rows.append({"id":ident,"kind":kind,"correct":score(kind,answer,spec),
                     "generated_tokens":used,"answer":answer})
    by_kind={}
    for kind in sorted({row["kind"] for row in rows}):
        selected=[row for row in rows if row["kind"]==kind]
        by_kind[kind]={"correct":sum(row["correct"] for row in selected),"total":len(selected)}
    return {"correct":sum(row["correct"] for row in rows),"total":len(rows),
            "accuracy_pct":100*sum(row["correct"] for row in rows)/len(rows),
            "by_kind":by_kind,"rows":rows}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--candidate-sha256",required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute:p.error("--execute required")
    if a.out.exists() or not a.out.is_absolute():p.error("new absolute output required")
    baseline=require_pinned_checkpoint(a.baseline)
    if sha_file(baseline)!=BASE_SHA:raise ValueError("baseline SHA mismatch")
    assets=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(assets,assets/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    guard.require_cuda_ready(torch)
    tok=Tokenizer.from_file(str(tok_path));bos=tok.token_to_id("<|bos|>");eos=tok.token_to_id("<|eos|>")
    if tok.get_vocab_size()!=FRONTIER_350.vocab_size or bos is None or eos is None:raise ValueError("tokenizer contract changed")
    device=torch.device("cuda")

    base_state=_checked_historical_weights(baseline,torch,Neral350M,FRONTIER_350.to_dict())
    base=Neral350M().to(device=device,dtype=torch.bfloat16);base.load_state_dict(base_state["model"],strict=True);base.eval()
    del base_state
    base_result=evaluate(base,tok,int(bos),int(eos),torch)
    del base;torch.cuda.empty_cache()

    candidate_state=load_candidate(torch,a.candidate,a.candidate_sha256,FRONTIER_350.to_dict())
    candidate=Neral350M().to(device=device,dtype=torch.bfloat16);candidate.load_state_dict(candidate_state["model"],strict=True);candidate.eval()
    candidate_meta={k:candidate_state.get(k) for k in ("sft_steps","sft_supervised_tokens","train_blocks","learning_rate")}
    del candidate_state
    candidate_result=evaluate(candidate,tok,int(bos),int(eos),torch)
    del candidate;torch.cuda.empty_cache();guard.require_safe_runtime_temperature()

    delta_correct=candidate_result["correct"]-base_result["correct"]
    regression_ids=[b["id"] for b,c in zip(base_result["rows"],candidate_result["rows"]) if b["correct"] and not c["correct"]]
    improvement_ids=[b["id"] for b,c in zip(base_result["rows"],candidate_result["rows"]) if not b["correct"] and c["correct"]]
    proceed=(delta_correct>=2 and len(regression_ids)<=2)
    report={
      "schema":"phoenix_chatbot_v1_fresh_generation_audit_v1",
      "classification":"fresh_post_training_development_not_independent_holdout",
      "suite_sha256":suite_sha(),"base_sha256":BASE_SHA,"candidate_sha256":a.candidate_sha256,
      "candidate_meta":candidate_meta,
      "baseline":base_result,"candidate":candidate_result,
      "delta_correct":delta_correct,"improvement_ids":improvement_ids,"regression_ids":regression_ids,
      "proceed_to_knowledge_lane":proceed,
      "production_eligible":False,"independent_holdout":False,"quality_gain_verified":False,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"chatbot_v1_fresh_generation_audit","suite_sha256":suite_sha(),
      "baseline_correct":base_result["correct"],"candidate_correct":candidate_result["correct"],
      "total":base_result["total"],"delta_correct":delta_correct,
      "improvements":improvement_ids,"regressions":regression_ids,
      "proceed_to_knowledge_lane":proceed,"production_eligible":False},sort_keys=True),flush=True)
    return 0 if proceed else 2


if __name__=="__main__":
    raise SystemExit(main())
