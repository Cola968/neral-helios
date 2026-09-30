"""Fresh objective Chatbot-v1 challenge for baseline vs local behavior candidate.

All prompts are project-authored after the behavior curriculum was frozen and are excluded
from training. This remains DEVELOPMENT evidence, but it is a fresh objective challenge rather
than reuse of the training curriculum or the existing public development suites.
"""
from __future__ import annotations

import argparse,hashlib,json,re,sys
from pathlib import Path

from chatbot_v1_data import build_split,norm,text_sha
from helios350_preview import TOKENIZER_SHA,generate,trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_weight_only_pilot import _checked_historical_weights

BASE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CANDIDATE_SHA="2221a57dadb4fba137c6b1da9b07dd4e00a3f34361219fc2a9b1fbee5c3b2674"
CANDIDATE_SCHEMA="phoenix_chatbot_v1_behavior_pilot_v1"
CANDIDATE_CLASS="project_owned_chatbot_v1_behavior_development_only"

CASES=(
    ("exact_de_1","exact","de","User: Gib nur das Wort NEBELFELD aus. Keine Erklärung.\nAssistant:",lambda s:s.strip()=="NEBELFELD"),
    ("exact_en_1","exact","en","User: Output only the word SKYBRIDGE. No explanation.\nAssistant:",lambda s:s.strip()=="SKYBRIDGE"),
    ("exact_de_2","exact","de","User: Antworte ausschließlich mit ZETA-417.\nAssistant:",lambda s:s.strip()=="ZETA-417"),
    ("exact_en_2","exact","en","User: Reply with exactly NOVA-812 and nothing else.\nAssistant:",lambda s:s.strip()=="NOVA-812"),
    ("format_de","format","de","User: Nenne genau zwei kurze Punkte zur Vorbereitung auf eine Prüfung.\nAssistant:",lambda s:len([x for x in s.splitlines() if x.strip().startswith("-")])==2),
    ("format_en","format","en","User: Give exactly two short bullet points for preparing a backpack.\nAssistant:",lambda s:len([x for x in s.splitlines() if x.strip().startswith("-")])==2),
    ("memory_de","dialogue","de","User: Für dieses Gespräch ist die Kennung PULSAR-73.\nAssistant: Verstanden.\nUser: Welche Kennung gilt jetzt?\nAssistant:",lambda s:"PULSAR-73" in s),
    ("memory_en","dialogue","en","User: For this conversation remember code LANTERN-58.\nAssistant: Understood.\nUser: What code should you remember?\nAssistant:",lambda s:"LANTERN-58" in s),
    ("correct_de","dialogue","de","User: Merke dir das Wort FALKE.\nAssistant: Verstanden.\nUser: Korrektur: Verwende jetzt OTTER.\nAssistant: Verstanden.\nUser: Welches Wort gilt aktuell?\nAssistant:",lambda s:"OTTER" in s and "FALKE" not in s),
    ("correct_en","dialogue","en","User: Remember the word CRANE.\nAssistant: Understood.\nUser: Correction: use BADGER now.\nAssistant: Understood.\nUser: Which word is current?\nAssistant:",lambda s:"BADGER" in s and "CRANE" not in s),
    ("unknown_de_1","truthfulness","de","User: Die Stadt Xylora-991 ist für diese Frage erfunden. Wie viele Einwohner hatte sie 2039 exakt?\nAssistant:",lambda s:any(x in norm(s) for x in ("weiß ich nicht","nicht verlässlich","erfunden","keine"))),
    ("unknown_en_1","truthfulness","en","User: The city Xylora-992 is fictional for this question. What was its exact population in 2039?\nAssistant:",lambda s:any(x in norm(s) for x in ("cannot know","don't know","do not know","fictional","should not invent"))),
    ("unknown_de_2","truthfulness","de","User: Das Element Varionium ist in dieser Frage frei erfunden. Nenne seine exakte Ordnungszahl.\nAssistant:",lambda s:any(x in norm(s) for x in ("erfunden","weiß ich nicht","nicht bekannt","keine"))),
    ("unknown_en_2","truthfulness","en","User: The element Varionium is invented for this question. Give its exact atomic number.\nAssistant:",lambda s:any(x in norm(s) for x in ("invented","cannot","don't know","do not know","no exact"))),
    ("fact_de_1","knowledge","de","User: Wie viele Kontinente werden im üblichen Sieben-Kontinente-Modell gezählt? Antworte mit einer Ziffer.\nAssistant:",lambda s:bool(re.match(r"\s*7\b",s))),
    ("fact_en_1","knowledge","en","User: How many legs does a typical insect have? Answer with one digit.\nAssistant:",lambda s:bool(re.match(r"\s*6\b",s))),
    ("fact_de_2","knowledge","de","User: Wie viele Minuten hat eine Stunde? Antworte nur mit der Zahl.\nAssistant:",lambda s:bool(re.match(r"\s*60\b",s))),
    ("fact_en_2","knowledge","en","User: How many degrees are in a right angle? Answer only with the number.\nAssistant:",lambda s:bool(re.match(r"\s*90\b",s))),
    ("math_de","math","de","User: Berechne 14 + 9. Antworte nur mit der Zahl.\nAssistant:",lambda s:bool(re.match(r"\s*23\b",s))),
    ("math_en","math","en","User: Calculate 7 times 8. Answer only with the number.\nAssistant:",lambda s:bool(re.match(r"\s*56\b",s))),
    ("translate_de","bilingual","de","User: Übersetze ins Englische und gib nur die Übersetzung aus: Das Fenster ist offen.\nAssistant:",lambda s:"the window is open" in norm(s)),
    ("translate_en","bilingual","en","User: Translate into German and output only the translation: The book is on the table.\nAssistant:",lambda s:"buch" in norm(s) and "tisch" in norm(s)),
    ("concise_de","concise","de","User: Was ist Reibung? Antworte in genau einem kurzen Satz.\nAssistant:",lambda s:0<len(s.strip())<220 and s.count(".")<=2),
    ("concise_en","concise","en","User: What is evaporation? Answer in exactly one short sentence.\nAssistant:",lambda s:0<len(s.strip())<220 and s.count(".")<=2),
)


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def challenge_hashes():
    return {text_sha(norm(prompt)) for _id,_cat,_lang,prompt,_score in CASES}


def verify_disjoint():
    train_valid={text_sha(norm(turn["user"])) for split in ("train","validation")
                 for row in build_split(split) for turn in row["turns"]}
    challenge=challenge_hashes()
    if train_valid&challenge:raise RuntimeError("fresh challenge overlaps behavior curriculum")
    return {"cases":len(CASES),"curriculum_overlap":0,"challenge_sha256":hashlib.sha256(
        "\n".join(sorted(challenge)).encode()).hexdigest()}


def load_candidate(torch,path:Path):
    if sha256_file(path)!=CANDIDATE_SHA:raise ValueError("candidate SHA mismatch")
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state.get("checkpoint_schema")!=CANDIDATE_SCHEMA or state.get("classification")!=CANDIDATE_CLASS:
        raise ValueError("candidate contract mismatch")
    if state.get("base_checkpoint_sha256")!=BASE_SHA or state.get("tokenizer_sha256")!=TOKENIZER_SHA:
        raise ValueError("candidate lineage mismatch")
    if state.get("production_eligible") is not False:raise ValueError("unsafe candidate state")
    return state


def evaluate(model,tokenizer,bos,eos,torch):
    rows=[]
    for ident,category,language,prompt,scorer in CASES:
        answer,used=generate(model,tokenizer,bos,eos,torch,prompt,48)
        rows.append({"id":ident,"category":category,"language":language,
                     "correct":bool(scorer(answer)),"generated_tokens":used,
                     "answer_sha256":hashlib.sha256(answer.encode()).hexdigest(),
                     "answer_chars":len(answer)})
    cats={}
    for category in sorted({r["category"] for r in rows}):
        group=[r for r in rows if r["category"]==category]
        cats[category]={"correct":sum(r["correct"] for r in group),"total":len(group),
                        "accuracy_pct":100*sum(r["correct"] for r in group)/len(group)}
    return {"correct":sum(r["correct"] for r in rows),"total":len(rows),
            "accuracy_pct":100*sum(r["correct"] for r in rows)/len(rows),
            "categories":cats,"rows":rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--base-checkpoint",type=Path,required=True)
    p.add_argument("--candidate-checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if not a.execute:p.error("explicit execute required")
    if a.out.exists() or not a.out.is_absolute():p.error("new absolute output required")
    disjoint=verify_disjoint()
    base=require_pinned_checkpoint(a.base_checkpoint)
    root=a.assets_root.resolve(strict=True)
    tok_path=trusted_asset(root,root/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    guard.require_cuda_ready(torch)

    tok=Tokenizer.from_file(str(tok_path));bos=tok.token_to_id("<|bos|>");eos=tok.token_to_id("<|eos|>")
    historical=_checked_historical_weights(base,torch,Neral350M,FRONTIER_350.to_dict())
    base_model=Neral350M().to("cuda",dtype=torch.bfloat16);base_model.load_state_dict(historical["model"],strict=True)
    baseline=evaluate(base_model,tok,int(bos),int(eos),torch)
    del base_model,historical;torch.cuda.empty_cache()

    candidate_state=load_candidate(torch,a.candidate_checkpoint)
    candidate_model=Neral350M().to("cuda",dtype=torch.bfloat16);candidate_model.load_state_dict(candidate_state["model"],strict=True)
    candidate=evaluate(candidate_model,tok,int(bos),int(eos),torch)
    del candidate_model,candidate_state;torch.cuda.empty_cache();guard.require_safe_runtime_temperature()

    delta=candidate["accuracy_pct"]-baseline["accuracy_pct"]
    category_delta={k:candidate["categories"][k]["accuracy_pct"]-baseline["categories"][k]["accuracy_pct"]
                    for k in baseline["categories"]}
    truth_ok=category_delta["truthfulness"]>=0
    knowledge_ok=category_delta["knowledge"]>=0 and category_delta["math"]>=0
    behavior_gain=sum(category_delta[k] for k in ("exact","format","dialogue","concise"))>0
    qualifies=delta>0 and truth_ok and knowledge_ok and behavior_gain
    report={"classification":"FRESH_PROJECT_OWNED_OBJECTIVE_CHALLENGE_DEVELOPMENT_ONLY",
            "disjoint":disjoint,"base_sha256":BASE_SHA,"candidate_sha256":CANDIDATE_SHA,
            "baseline":baseline,"candidate":candidate,"overall_delta_pct":delta,
            "category_delta_pct":category_delta,"qualifies_for_knowledge_lane":qualifies,
            "independent_holdout":False,"production_eligible":False}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"chatbot_v1_fresh_challenge_complete","baseline_correct":baseline["correct"],
      "candidate_correct":candidate["correct"],"overall_delta_pct":delta,
      "category_delta_pct":category_delta,"qualifies_for_knowledge_lane":qualifies,
      "production_eligible":False},sort_keys=True),flush=True)
    return 0 if qualifies else 2

if __name__=="__main__":raise SystemExit(main())
