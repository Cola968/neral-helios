"""Find full Qwen3.5-4B BF16 checkpoint on C/D, copy to D, else pinned HF download.

No GGUF, no model conversion, no writes outside D:\\neral-qlora-v2.
"""
from __future__ import annotations
import argparse, json, os, shutil, time
from pathlib import Path

REPO="Qwen/Qwen3.5-4B"
REVISION="a7b0d22b993d71000cf2eadfb37222a67cee521e"
ALLOWED=("*.safetensors","*.json","*.txt","*.model","*.jinja","vocab.*","merges.*")
WEIGHT_MIN_BYTES=9_000_000_000

def complete_model(path:Path)->bool:
    if not path.is_dir() or not (path/"config.json").is_file():return False
    idx=path/"model.safetensors.index.json"
    if not idx.is_file():return False
    try:
        index=json.loads(idx.read_text(encoding="utf8"))
        names=set(index["weight_map"].values())
        if not names or not all(name.startswith("model") and name.endswith(".safetensors") for name in names):
            return False
        total=0
        for name in names:
            file=path/name
            if not file.is_file():return False
            total+=file.stat().st_size
        return total>=WEIGHT_MIN_BYTES and (path/"tokenizer_config.json").exists()
    except (ValueError,KeyError,OSError,TypeError):
        return False

def candidates(root:Path)->list[Path]:
    home=Path.home()
    locations=[
       Path("D:/neral-qlora/checkpoint-4b"),
       Path("D:/neral-qlora-v2/luna-checkpoint"),
       Path("D:/neral-models/Qwen3.5-4B"),
       home/".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots",
       Path("D:/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"),
       Path("D:/neral-qlora-v2/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots"),
    ]
    for env in ("HF_HOME","HF_HUB_CACHE","HUGGINGFACE_HUB_CACHE"):
        x=os.getenv(env)
        if x:
            base=Path(x)
            locations.extend([base/"hub/models--Qwen--Qwen3.5-4B/snapshots",
                              base/"models--Qwen--Qwen3.5-4B/snapshots"])
    choices=[]
    for location in locations:
        if location.name=="snapshots" and location.is_dir():
            choices.extend(p for p in location.iterdir() if p.is_dir())
        else:choices.append(location)
    return list(dict.fromkeys(choices))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    output=args.out
    if str(output.resolve()).lower().startswith(str(Path("D:/neral-qlora-v2").resolve()).lower()) is False:
        raise RuntimeError("4B weights may be placed only under D:/neral-qlora-v2")
    if complete_model(output):
        print("LUNA_4B_EXISTING_CHECKPOINT "+json.dumps({"path":str(output),"complete":True}),flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Existing nonempty 4B checkpoint directory is incomplete; do not overwrite")
    available=next((p for p in candidates(output) if p!=output and complete_model(p)),None)
    disk=shutil.disk_usage(str(Path("D:/")))
    free=disk.free
    if free<18*1024**3:
        raise RuntimeError("Less than 18 GiB free D:; refuse a 9.3GB model download/copy")
    output.parent.mkdir(parents=True,exist_ok=True)
    if available is not None:
        print("LUNA_4B_COPY_LOCAL "+str(available),flush=True)
        # Copies full verified safetensors shards, all additions stay on D:
        shutil.copytree(available,output,symlinks=False)
    else:
        print("LUNA_4B_DOWNLOAD_PINNED_FROM_HF "+REPO+"@"+REVISION,flush=True)
        os.environ["HF_HOME"]="D:/neral-qlora-v2/hf-cache"
        os.environ["HF_HUB_CACHE"]="D:/neral-qlora-v2/hf-cache/hub"
        os.environ["HF_XET_CACHE"]="D:/neral-qlora-v2/hf-cache/xet"
        os.environ.pop("HF_HUB_OFFLINE",None)
        os.environ.pop("TRANSFORMERS_OFFLINE",None)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=REPO,revision=REVISION,local_dir=str(output),
                          allow_patterns=list(ALLOWED),max_workers=3)
    if not complete_model(output):
        raise RuntimeError("Downloaded Qwen3.5-4B full safetensors incomplete; DO NOT train Luna")
    print("LUNA_4B_CHECKPOINT_VERIFIED "+json.dumps({
        "model":REPO,"revision":REVISION,"path":str(output),
        "safetensors_bytes":sum((output/f).stat().st_size for f in set(
        json.loads((output/"model.safetensors.index.json").read_text())["weight_map"].values())),
        "source":"existing_local" if available is not None else "pinned_huggingface_download"},sort_keys=True),flush=True)

if __name__=="__main__":main()
