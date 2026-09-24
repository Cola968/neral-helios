"""Explicit import of an already-produced own Frontier 350M checkpoint into Helios assets.

This registers a deployment bundle only. It never trains, promotes quality, scans training
runs, or starts the Frontier runtime.
"""
from __future__ import annotations

import argparse,hashlib,json,os,re,shutil,tempfile
from pathlib import Path

from frontier350_contract import (
    FRONTIER350_CONFIG,FRONTIER350_PARAMETERS,FRONTIER350_RUNTIME,
    FRONTIER350_TOKENIZER_SHA256,expected_state_shapes,
)

MAX_CHECKPOINT_BYTES=4*1024**3

def sha256_file(path:Path)->str:
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def _validate_identity(key,label,status):
    if not re.fullmatch(r"neral-[a-z0-9-]{1,48}",key):raise ValueError("Invalid identity")
    if not isinstance(label,str) or not 1<=len(label)<=80:raise ValueError("Invalid label")
    if status not in {"development","candidate","release"}:raise ValueError("Invalid status")

def validate_frontier_checkpoint(source:Path,expected_sha:str,tokenizer:Path,tokenizer_sha:str):
    if not source.is_file() or source.is_symlink():raise ValueError("Checkpoint must be a regular local file")
    if not tokenizer.is_file() or tokenizer.is_symlink():raise ValueError("Tokenizer must be a regular local file")
    if source.stat().st_size>MAX_CHECKPOINT_BYTES:raise ValueError("Checkpoint exceeds 4 GiB")
    if not re.fullmatch(r"[a-f0-9]{64}",expected_sha):raise ValueError("Expected checkpoint SHA-256 required")
    if not re.fullmatch(r"[a-f0-9]{64}",tokenizer_sha):raise ValueError("Expected tokenizer SHA-256 required")
    actual=sha256_file(source)
    if actual!=expected_sha:raise ValueError("Checkpoint integrity mismatch")
    actual_tokenizer=sha256_file(tokenizer)
    if actual_tokenizer!=tokenizer_sha or actual_tokenizer!=FRONTIER350_TOKENIZER_SHA256:
        raise ValueError("Frontier tokenizer integrity mismatch")

    import torch
    payload=torch.load(source,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict):raise ValueError("Invalid Frontier checkpoint payload")
    if payload.get("config")!=FRONTIER350_CONFIG:raise ValueError("Frontier architecture contract mismatch")
    if payload.get("tokenizer_sha256")!=FRONTIER350_TOKENIZER_SHA256:raise ValueError("Checkpoint tokenizer contract mismatch")
    processed=payload.get("processed_tokens")
    if type(processed) is not int or processed<=0:raise ValueError("Missing processed-token lineage")
    state=payload.get("model")
    shapes=expected_state_shapes()
    if not isinstance(state,dict) or set(state)!=set(shapes):raise ValueError("Frontier state-dict keys mismatch")
    allowed={torch.bfloat16,torch.float32}
    for name,shape in shapes.items():
        value=state[name]
        if not torch.is_tensor(value) or tuple(value.shape)!=shape:raise ValueError(f"Frontier tensor shape mismatch: {name}")
        if value.dtype not in allowed:raise ValueError(f"Frontier tensor dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):raise ValueError(f"Non-finite Frontier tensor: {name}")
    if not torch.equal(state["tok_emb.weight"],state["lm_head.weight"]):
        raise ValueError("Frontier tied embedding/head mismatch")
    return {"checkpoint_sha256":actual,"tokenizer_sha256":actual_tokenizer,
            "processed_tokens":processed,"dtype":str(state["tok_emb.weight"].dtype).replace("torch.",""),
            "state_entries":len(state),"parameters":FRONTIER350_PARAMETERS}

def import_frontier350(source,expected_sha,tokenizer,tokenizer_sha,key,label,assets,status="candidate",context_tokens=512,runtime_port=9683):
    _validate_identity(key,label,status)
    if type(context_tokens) is not int or not 1<=context_tokens<=FRONTIER350_CONFIG["max_seq_len"]:
        raise ValueError("Invalid context_tokens")
    if type(runtime_port) is not int or not 1024<=runtime_port<=65535:
        raise ValueError("Invalid runtime_port")
    source=Path(source);tokenizer=Path(tokenizer);assets=Path(assets)
    verified=validate_frontier_checkpoint(source,expected_sha,tokenizer,tokenizer_sha)

    root=assets.resolve()/"installed_models";root.mkdir(parents=True,exist_ok=True)
    lock=root/"import.lock"
    try:lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError("Another import is active; inspect import.lock after a crash")
    target=root/key
    try:
        if target.exists():raise ValueError("Model id is already installed")
        target.mkdir()
        shutil.copyfile(source,target/"model.pt")
        shutil.copyfile(tokenizer,target/"tokenizer.json")
        model_digest=sha256_file(target/"model.pt");tok_digest=sha256_file(target/"tokenizer.json")
        if model_digest!=expected_sha or tok_digest!=tokenizer_sha:raise ValueError("Deployment copy integrity mismatch")
        (target/"model.sha256").write_text(model_digest+"\n",encoding="ascii")
        (target/"tokenizer.sha256").write_text(tok_digest+"\n",encoding="ascii")
        report={
            "id":key,"label":label,"runtime_family":FRONTIER350_RUNTIME,"status":status,
            "parameters":FRONTIER350_PARAMETERS,"context_tokens":context_tokens,
            "runtime_port":runtime_port,**verified,"source_checkpoint":str(source.name),"source_tokenizer":str(tokenizer.name),
            "quality_promotion":False,"runtime_qualified":False,"training_started":False,
        }
        (target/"import-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        catalog_path=root/"catalog.json"
        rows=json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
        if not isinstance(rows,list) or len(rows)>=32 or any(isinstance(r,dict) and r.get("id")==key for r in rows):
            raise ValueError("Catalog full or duplicate")
        rows.append({"id":key,"label":label,"runtime_family":FRONTIER350_RUNTIME,
                     "status":status,"parameters":FRONTIER350_PARAMETERS,
                     "context_tokens":context_tokens,"runtime_port":runtime_port})
        with tempfile.NamedTemporaryFile("w",dir=root,delete=False,encoding="utf-8") as tmp:
            json.dump(rows,tmp,indent=2);tmp.write("\n");tmp.flush();os.fsync(tmp.fileno());tmp_name=tmp.name
        os.replace(tmp_name,catalog_path)
        return report
    except Exception:
        if target.exists():shutil.rmtree(target,ignore_errors=True)
        raise
    finally:
        os.close(lock_fd);lock.unlink(missing_ok=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--tokenizer-sha256",required=True)
    p.add_argument("--id",required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--assets",default="deployment")
    p.add_argument("--status",choices=("development","candidate","release"),default="candidate")
    p.add_argument("--context-tokens",type=int,default=512)
    p.add_argument("--runtime-port",type=int,default=9683)
    a=p.parse_args()
    print(json.dumps(import_frontier350(a.source,a.sha256,a.tokenizer,a.tokenizer_sha256,
        a.id,a.label,a.assets,a.status,a.context_tokens,a.runtime_port),indent=2,sort_keys=True))
