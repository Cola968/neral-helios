"""Re-execute EVERY synthetic Python example inside an isolated Docker container.

Host mode needs Docker; worker mode intentionally requires stdlib only.
This script never reads DEV24, holdout 40 or protected validation 32.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, re, subprocess, sys
from pathlib import Path

IMAGE="python:3.12.8-alpine3.21"
_ALLOWED=(ast.Module,ast.Expr,ast.Assign,ast.AugAssign,ast.For,ast.If,
          ast.Name,ast.Load,ast.Store,ast.Constant,ast.BinOp,ast.Add,
          ast.Mult,ast.Mod,ast.Sub,ast.Div,ast.FloorDiv,ast.Call,
          ast.Compare,ast.Eq,ast.USub,ast.UnaryOp)

def safe_program(source):
    if type(source) is not str or not 1<=len(source)<=1200:
        raise ValueError("invalid Python source length")
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node,_ALLOWED):
            raise ValueError("unexpected Python AST node "+type(node).__name__)
        if isinstance(node,ast.Call) and (
             not isinstance(node.func,ast.Name) or
             node.func.id not in ("print","sum","range") or node.keywords):
            raise ValueError("disallowed call")
    return tree

def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()

def worker(dataset):
    train=dataset/"helios_terra_full.train.jsonl"
    val=dataset/"helios_terra_full.val.jsonl"
    manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf8"))
    for name,path in [("train",train),("validation",val)]:
        if sha(path)!=manifest["split_sha256"][name]:
            raise RuntimeError("Dataset bytes differ from pinned manifest")
    records=[]
    for name,path in [("train",train),("validation",val)]:
        for line in path.read_text(encoding="utf8").splitlines():
            obj=json.loads(line)
            if obj["family"]!="core11" or obj["evidence"]["method"]!="python_reference_requires_docker":
                continue
            records.append((name,obj))
    expected_count=(manifest["train_categories"]["core11"]-((manifest["train_categories"]["core11"]+1)//2)) + (manifest["val_categories"]["core11"]-((manifest["val_categories"]["core11"]+1)//2))
    if len(records)!=expected_count:raise RuntimeError("Unexpected count of Python core cases")
    repairs=0
    for name,row in records:
        task=row["source_task"]
        code=task["code"]
        safe_program(code)
        first=task["original_broken"]
        if first:
            repairs+=1
            try:ast.parse(first)
            except SyntaxError:pass
            else:raise RuntimeError("Original program is not actually broken")
            if len(code)!=len(first)+1 or code.replace(":","",1)!=first.replace(":","",1):
                # Structural fallback: exactly one colon added, not a full rewrite.
                if not any(code[:i]+code[i+1:]==first for i,c in enumerate(code) if c==":"):
                    raise RuntimeError("More than one repair edit")
        target=json.loads(row["messages"][-1]["content"])
        if target.get("answer")!=task["expected"]:
            raise RuntimeError("Training target did not match independently computed expected integer")
        if first and target.get("corrected_code")!=code:
            raise RuntimeError("Broken code and corrected target differ")
        try:
            result=subprocess.run([sys.executable,"-I","-B","-c",code],
                                  capture_output=True,text=True,timeout=3,check=True)
        except subprocess.SubprocessError:
            raise RuntimeError("Isolated Docker Python task failed")
        if result.stdout.strip()!=str(task["expected"]) or result.stderr:
            raise RuntimeError("Docker output mismatch or stderr")
    output={"status":"python_docker_all_pass",
            "python_cases_verified":len(records),
            "one_step_ast_repairs_verified":repairs,
            "train_sha256":sha(train),"val_sha256":sha(val),
            "no_network":True,"no_model_invocation":True}
    print("NERAL_V2_DOCKER_PROOF "+json.dumps(output,sort_keys=True),flush=True)
    return output


def host(dataset,out):
    if out.exists():raise FileExistsError("Proof cannot overwrite existing report")
    import shutil
    docker=shutil.which("docker")
    if not docker:raise RuntimeError("Docker executable unavailable; fail closed")
    if not(dataset/"manifest.json").is_file():raise FileNotFoundError("Curriculum missing")
    image=subprocess.run([docker,"image","inspect",IMAGE,"--format","{{.Id}}"],
                         capture_output=True,text=True)
    if image.returncode!=0:
        pull=subprocess.run([docker,"pull",IMAGE],capture_output=True,text=True,timeout=160)
        if pull.returncode!=0:raise RuntimeError("Pinned Docker runtime image unavailable")
        image=subprocess.run([docker,"image","inspect",IMAGE,"--format","{{.Id}}"],
                             capture_output=True,text=True,check=True)
    image_id=image.stdout.strip()
    if not image_id.startswith("sha256:"):raise RuntimeError("Unpinned Docker image metadata")
    source=Path(__file__).resolve().parents[1]
    cmd=[docker,"run","--rm","--network","none","--read-only",
         "--cap-drop","ALL","--security-opt","no-new-privileges",
         "--memory","256m","--cpus","1.5","--pids-limit","64",
         "--user","65534:65534",
         "-v",str(source)+":/source:ro",
         "-v",str(dataset.resolve())+":/data:ro",
         IMAGE,"python","-B","/source/scripts/neral_v2_docker_proof.py",
         "--worker","--dataset","/data"]
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    if res.returncode:raise RuntimeError("Sandbox verifier blocked: "+res.stderr[-600:])
    lines=[l for l in res.stdout.splitlines() if l.startswith("NERAL_V2_DOCKER_PROOF ")]
    if len(lines)!=1:raise RuntimeError("No singular Docker proof receipt")
    proof=json.loads(lines[0].split(" ",1)[1])
    if proof["status"]!="python_docker_all_pass" or not 150<=proof["python_cases_verified"]<=260:
        raise RuntimeError("Invalid sandbox evidence")
    proof["image_id"]=image_id
    proof["image_tag"]=IMAGE
    proof["docker_verified"]=True
    proof["source_sha256"]=sha(source/"scripts/neral_v2_docker_proof.py")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_V2_DOCKER_CERTIFIED "+json.dumps(proof,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.worker:worker(args.dataset)
    else:
        if args.out is None:p.error("--out required for Docker host")
        host(args.dataset,args.out)

if __name__=="__main__":main()
