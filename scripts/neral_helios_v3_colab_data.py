"""Synthetic, disjoint Helios v3 preparation with isolated Linux Docker proof.

No training and no model calls. Forty-case test, 32 validation and DEV24
are used exclusively for hash-based exclusion, not oracle selection.
Each mathematical/programmatic example is independently recomputed in Docker.
The traces represent synthetic *provided* tool histories, never live calls.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory

TRAIN = {"z3":100,"sympy":100,"networkx":100,"python":100,"rag_style":80}
VAL = {"z3":12,"sympy":12,"networkx":12,"python":12,"rag_style":12}
SEED = "NERAL-HELIOS-V3-COLAB-20261117-01"
IMAGE = "neral-v3-verifier:py312-20261117"
TEMPLATE = "Qwen/Qwen3.5-9B"
SYSTEM = ("Du bist NERAL. Nutze einen vorliegenden Werkzeug-Prüfbericht nur als "
          "Beweismittel. Behaupte keine selbst ausgeführten Werkzeugaufrufe. "
          "Antworte auf Deutsch klar, exakt und mit Quellenbezug.")


def compact(v): return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(v): return hashlib.sha256(v.encode("utf8") if isinstance(v,str) else v).hexdigest()
def normalize(v): return " ".join(v.casefold().split())
def fingerprint(v): return sha(normalize(v))


def recipe(tool, index, split):
    j = index + (5000 if split=="validation" else 0)
    if tool=="z3":
        x, y = 13+3*j, 19+5*j
        spec={"x_min":0,"y_min":0,"sum":x+y,"difference":x-y}
        wrong={**spec,"sum":x+y+1,"also_sum":x+y}
        question=(f"In einem Zweier-System sind x und y nichtnegative ganze Zahlen. "
                  f"x+y={x+y}, x-y={x-y}. Ermittle die einzige Lösung mit Z3.")
        result={"x":x,"y":y}
        bad="WIDERSPRUeCHLICHE_SUMMEN"
    elif tool=="sympy":
        n=38+2*j;k=2+j%7
        spec={"operation":"binomial","n":n,"k":k}
        expected=math.comb(n,k)
        wrong={**spec,"proposed_result":expected+1}
        question=(f"Von {n} unterschiedlichen Komponenten werden {k} ohne "
                  f"Reihenfolge ausgewählt. Wie viele Auswahlen gibt es? "
                  f"Kontrolliere das Ergebnis mit SymPy.")
        result=expected
        bad="BINOMIAL_ERGEBNIS_FALSCH"
    elif tool=="networkx":
        ab=5+j%17;bd=8+j%13;ac=3+j%19;cd=7+j%23;ad=23+j%29
        edges=[["A","B",ab],["B","D",bd],["A","C",ac],
               ["C","D",cd],["A","D",ad]]
        spec={"nodes":["A","B","C","D"],"edges":edges,"source":"A","target":"D"}
        wrong={**spec,"target":"X"}
        question=(f"Bestimme die minimalen Kosten von A nach D in diesem "
                  f"gerichteten Graphen: {compact(edges)}. Prüfe mit NetworkX.")
        result=min(ab+bd,ac+cd,ad)
        bad="ZIELKNOTEN_UNBEKANNT"
    elif tool=="python":
        n=21+j%31; a=3+j%9; b=5+j%13
        correct=(f"total = 0\nfor i in range({n}):\n"
                 f"    total += i*{a}+{b}\nprint(total)")
        broken=correct.replace(f"for i in range({n}):",f"for i in range({n})")
        spec={"code":correct}
        wrong={"code":broken}
        question=(f"Das folgende Python-3-Programm ist durch einen Syntaxfehler "
                  f"unterbrochen. Repariere exakt einen Fehler und ermittle die "
                  f"Ausgabe mit der isolierten Python-Sandbox:\n{broken}")
        result=a*n*(n-1)//2+b*n
        bad="SYNTAX_ERROR_MISSING_COLON"
    else: raise ValueError("Unknown tool")
    return question,spec,wrong,result,bad


def mk_tool(tool,index,split):
    question,spec,wrong,result,error=recipe(tool,index,split)
    prefix=("Ein synthetischer Werkzeugbericht meldet zunächst einen "
            "fehlgeschlagenen Vorschlag. Werte den Fehler aus, "
            "schlage einen korrigierten Aufruf vor und gib das unabhängig "
            "verifizierbare Ergebnis an. ")
    prompt=prefix+question
    answer=compact({"plan":{"tool":tool,"arguments":spec},
        "initial_rejected":{"arguments":wrong,"error_code":error},
        "corrected_call":{"tool":tool,"arguments":spec},
        "verified_expected":result,"final_answer":result})
    task={"tool":tool,"spec":spec,"wrong":wrong,"result":result,
          "error_code":error}
    return {"id":fingerprint(prompt),"family":"tool_trace_"+tool,
            "messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":prompt},
                        {"role":"assistant","content":answer}],
            "source_task":task,
            "evidence":{"method":"synthetic_independent_docker_pending",
                        "actual_runtime_tool_call":False,
                        "synthetic_trace":True}}


STYLE_TOPICS=(
 ("Schach","ein Springer als Leichtfigur","Springer und Läufer sind Leichtfiguren"),
 ("Flugverkehr","eine Drehkreuz-Welle","Ankünfte und Abflüge werden zeitlich gebündelt"),
 ("Mathematik","einen Ableitungsschritt","Die Ableitung beschreibt eine lokale Änderungsrate"),
 ("Physik","elektrische Ladung","Die Ladung Q ergibt sich aus Stromstärke I mal Zeit t"),
 ("Stadtklima","Albedo","Helle Oberflächen reflektieren tendenziell mehr Sonnenstrahlung"),
 ("Software","einen Regressionstest","Der Test hält erwartetes Verhalten fest"),
 ("Astronomie","ein Lichtjahr","Ein Lichtjahr misst Entfernung und keine Zeit"),
 ("Datenbanken","einen JOIN","Ein JOIN verknüpft Datensätze anhand einer Bedingung"),
)


def mk_style(index,split):
    i=index+(9000 if split=="validation" else 0)
    topic,term,fact=STYLE_TOPICS[index%len(STYLE_TOPICS)]
    code=f"SYN-{split[0].upper()}-{i:05d}"
    mode=index%4
    if mode==0:
        value=4+i%43
        text=f"Fiktive Notiz {code}: Der Prüfzyklus beträgt {value} Tage."
        question=f"Was steht in Notiz {code} über den Prüfzyklus?"
        reply=f"Der Prüfzyklus beträgt {value} Tage. [Quelle: {code}]"
        kind="rag_present"
    elif mode==1:
        text=f"Fiktive Notiz {code}: Der Prüfzyklus ist dokumentiert. Eine Farbe wird nicht erwähnt."
        question=f"Welche Farbe hat das Gerät in Notiz {code}?"
        reply="Dazu gibt es in der vorliegenden Notiz keine Angabe."
        kind="rag_missing"
    elif mode==2:
        text=""
        question=f"Erstelle für mein Vorhaben {code} einen konkreten Terminplan."
        reply="Welche Frist und welche verfügbaren Stunden soll ich berücksichtigen? Ohne diese Angaben wäre ein konkreter Zeitplan geraten."
        kind="clarify"
    else:
        text=""
        question=f"Erkläre kurz {term} für meine Lernkarte {code}."
        reply=f"{fact}. Das ist eine allgemeine Erklärung, keine Messung für einen konkreten Fall."
        kind="concise_de"
    context=(f"Kontext aus einer ausdrücklich synthetischen Quelle:\n"
             f"[Quelle: {code}] {text}\n" if text else "")
    prompt=context+"Nutzerfrage: "+question
    return {"id":fingerprint(prompt),"family":"rag_style_"+kind,
            "messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":prompt},
                        {"role":"assistant","content":reply}],
            "source_task":{"kind":kind,"document":text,"source":code,
                           "fact":fact,"query":question},
            "evidence":{"method":"synthetic_style_not_human_reviewed",
                        "synthetic":True,"human_reviewed":False}}


def generate(split):
    if split not in ("train","validation"):raise ValueError("Unknown split")
    counts=TRAIN if split=="train" else VAL
    rows=[]
    for family,n in counts.items():
        for i in range(n):
            rows.append(mk_style(i,split) if family=="rag_style"
                        else mk_tool(family,i,split))
    if len(rows)!=sum(counts.values()) or len({r["id"] for r in rows})!=len(rows):
        raise RuntimeError("Duplicate or missing v3 examples")
    return rows


def exclusion_check(root,train,val):
    # Reuse the existing pinned read-only 40/32/DEV24 exclusion mechanism.
    from scripts.neral_v2_mix_build import protected,PROTECTED_SHA
    forbidden_prompts,forbidden_tasks=protected(root)
    blocked_tasks=set(forbidden_tasks)
    observed_prompts=set()
    observed_tasks=set()
    for row in train+val:
        prompt=row["messages"][1]["content"]
        if row["id"]!=fingerprint(prompt) or row["id"] in forbidden_prompts:
            raise RuntimeError("Overlap with protected test/validation/DEV24 prompt")
        if row["id"] in observed_prompts:raise RuntimeError("Cross-split prompt overlap")
        observed_prompts.add(row["id"])
        task=row["source_task"]
        key=sha(compact({"tool":task["tool"],"spec":task["spec"]}) if "tool" in task
                 else compact(task))
        if key in blocked_tasks or key in observed_tasks:
            raise RuntimeError("Protected or duplicate source task")
        observed_tasks.add(key)
    return {"protected_split_sha256":dict(PROTECTED_SHA),
            "protected_dev24":"exact_public_dev24_prompts_checked",
            "overlap_count":0,"protected_prompts_used_for_training":False,
            "protected_answers_used_for_training":False,
            "train_val_overlap_count":0}


def _oracle(row):
    task=row["source_task"]
    if task["kind"] if "kind" in task else False:
        q=row["messages"][1]["content"];a=row["messages"][2]["content"]
        kind=task["kind"]
        if kind=="rag_present":
            number=task["document"].split("beträgt ")[1].split(" Tage")[0]
            if (number+" Tage") not in a or f"[Quelle: {task['source']}]" not in a:
                raise RuntimeError("Synthetic RAG citation mismatch")
        elif kind=="rag_missing":
            if "keine Angabe" not in a or "Farbe" not in task["query"]:
                raise RuntimeError("Synthetic missing-fact response mismatch")
        elif kind=="clarify":
            if "Welche Frist" not in a or "Terminplan" not in q:raise RuntimeError("Clarification mismatch")
        else:
            if task["fact"] not in a:raise RuntimeError("Concise explanation mismatch")
        return "synthetic_style_rule_checked"
    t=task["tool"];s=task["spec"];w=task["wrong"];expected=task["result"]
    reply=json.loads(row["messages"][2]["content"])
    if reply["final_answer"]!=expected or reply["corrected_call"]!={"tool":t,"arguments":s}:
        raise RuntimeError("Tool trace is not the verified correction")
    if reply["initial_rejected"]!={"arguments":w,"error_code":task["error_code"]}:
        raise RuntimeError("Bad-proposal rejection missing")
    if t=="z3":
        import z3
        x,y=z3.Int("x"),z3.Int("y")
        solver=z3.Solver()
        solver.add(x>=s["x_min"],y>=s["y_min"],x+y==s["sum"],x-y==s["difference"])
        if solver.check()!=z3.sat:raise RuntimeError("Z3 target unsatisfiable")
        model=solver.model()
        sol={"x":model.eval(x).as_long(),"y":model.eval(y).as_long()}
        if sol!=expected:raise RuntimeError("Wrong Z3 solution")
        solver.add(z3.Or(x!=sol["x"],y!=sol["y"]))
        if solver.check()!=z3.unsat:raise RuntimeError("Z3 not unique")
        bad=z3.Solver()
        bad.add(x+y==w["sum"],x+y==w["also_sum"])
        if bad.check()!=z3.unsat:raise RuntimeError("Invalid Z3 rejection")
    elif t=="sympy":
        import sympy
        exact=int(sympy.binomial(s["n"],s["k"]))
        if exact!=expected or w["proposed_result"]==expected:
            raise RuntimeError("Incorrect SymPy verification")
    elif t=="networkx":
        import networkx as nx
        g=nx.DiGraph()
        g.add_nodes_from(s["nodes"])
        for a,b,c in s["edges"]:g.add_edge(a,b,weight=c)
        got=nx.shortest_path_length(g,source=s["source"],target=s["target"],weight="weight")
        if got!=expected or w["target"] in g:
            raise RuntimeError("Incorrect NetworkX verification")
    elif t=="python":
        code=s["code"];wrong=w["code"]
        ast.parse(code)
        try:ast.parse(wrong)
        except SyntaxError:pass
        else:raise RuntimeError("Python initial attempt not invalid")
        if code.replace(":", "",1)!=wrong:
            raise RuntimeError("Repair changes more than one colon")
        from scripts.neral_v2_docker_proof import safe_program
        safe_program(code)
        p=subprocess.run([sys.executable,"-I","-B","-c",code],
                         capture_output=True,text=True,check=True,timeout=3)
        if p.stderr or p.stdout.strip()!=str(expected):
            raise RuntimeError("Isolated container Python mismatch")
    else:raise RuntimeError("Unknown tool")
    return "math_python_oracle_verified"


def worker(dataset:Path):
    meta=json.loads((dataset/"manifest.json").read_text("utf8"))
    counts={}
    for key,expected in [("train",480),("validation",60)]:
        file=dataset/(key+".jsonl")
        if sha(file.read_bytes())!=meta["split_sha256"][key]:
            raise RuntimeError("Dataset hash mismatch")
        rows=[json.loads(x) for x in file.read_text("utf8").splitlines()]
        if len(rows)!=expected:raise RuntimeError("Count mismatch")
        for row in rows:
            family=row["family"]
            counts[family]=counts.get(family,0)+1
            _oracle(row)
    report={"status":"v3_all_oracles_verified_linux_docker",
        "counts":counts,"verified_rows":540,"tool_rows":448,"style_rows":92,
        "network_isolated_required":True,
        "train_sha256":meta["split_sha256"]["train"],
        "validation_sha256":meta["split_sha256"]["validation"],
        "no_gpu":True,"no_training":True,
        "original_eval_questions_used_as_training":False}
    print("NERAL_V3_DOCKER_PROOF "+compact(report),flush=True)
    return report


def prepare(root:Path,out:Path):
    if out.exists() or out.is_symlink():raise FileExistsError("No overwriting v3 dataset")
    train=generate("train");val=generate("validation")
    exclusion=exclusion_check(root,train,val)
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,rows in (("train",train),("validation",val)):
        path=out/(name+".jsonl")
        payload="".join(compact(row)+"\n" for row in rows)
        path.write_text(payload,"utf8",newline="\n")
        hashes[name]=sha(payload)
    meta={"status":"awaiting_independent_linux_docker_oracles",
        "dataset_id":SEED,"base_model":TEMPLATE,
        "expected_train":480,"expected_validation":60,
        "train_categories":TRAIN,"val_categories":VAL,
        "split_sha256":hashes,"exclusion":exclusion,
        "python_docker_verified":False,"synthetic_not_human_reviewed":True,
        "no_model_calls":True,"training_performed":False,
        "format":"three_message_system_user_assistant_answer_only",
        "protected_holdouts_included":False}
    (out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+"\n","utf8")
    print("NERAL_V3_DATA_GENERATED "+compact({"train":480,"validation":60,"hashes":hashes}),flush=True)
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--dataset",type=Path)
    args=p.parse_args()
    if args.worker:
        if args.dataset is None:raise ValueError("--dataset required")
        worker(args.dataset)
    else:
        if args.out is None:raise ValueError("--out required")
        prepare(Path(__file__).resolve().parents[1],args.out)


if __name__=="__main__":main()
