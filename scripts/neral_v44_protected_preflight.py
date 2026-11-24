"""V4.4 CPU-only protected data check; no GPU, no service stop, no training."""
from __future__ import annotations
import argparse, hashlib, json, os, secrets, subprocess
from pathlib import Path
from scripts.neral_v43_rft_cpu import safe, rows, digest, catalog, protected_hashes, canonical, write_jsonl
from scripts.neral_v43_reference50_freeze_guard import FILES

FROZEN="c0a6be607a0797908e101db460e3ac297a70f088"
BLIND={
    "sealed200.jsonl":"f89002d9eee3b7a2c0689ed98fab8d53cba1c50b7973b3333ffa0bdd6501d9fe",
    "manifest.json":"49853f17931e288702618bb8ea98d9039d6037c98e30e031961036c99ffbb368",
    "private-details.jsonl":"710577daefb98046b3265d5bff902492b04ac3331e43bba499401b4ae977fdfe",
}
REF_RECEIPT="707ca66aed7bfbfbb15ac7aa39229b3eaad572b17722371056f17c0f1f6ae807"
REF_TASKS="a7c8e033b157e320d198da81793c052c3978ba61af4a63f22a909887afc4b43e"
OH_TASKS="2bd09a5ff21bc93d270e04ecf623a9e4994c1fcab585672e13041bc70f3c4e6f"
OH_MANIFEST="1dd8d693a05bfbcec8dfb3e3ef8b98ed4acb096d7c90c24674cc0d514509b01a"
OH_RESP="4453fe4b79ab6af9e18eff275d75d8db49800519620ee5399dcfec101193e7d7"

def sha(path):
    path=safe(path)
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def pin(path, expected):
    if sha(path)!=expected:
        raise ValueError("HISTORICAL_SHA_MISMATCH")
    return {"path":str(safe(path)),"sha256":expected}

def check_blind(root):
    records=[]
    for directory in (root/"v41-blind200/44935727057",root/"artifacts/blind200-partial-run44935727057"):
        for name,expected in BLIND.items():
            records.append(pin(directory/name,expected))
        if len(rows(safe(directory/"private-details.jsonl").read_bytes()))!=421:
            raise ValueError("BLIND200_COUNT_MISMATCH")
    return {"path":records[0]["path"],"sha256":BLIND["sealed200.jsonl"],"immutable_files":records}

def check_reference(root):
    directory=root/"v43-reference50/45264946953"
    receipt=directory/"baseline-receipt.json"
    records=[pin(receipt,REF_RECEIPT),pin(directory/"sealed50.jsonl",REF_TASKS)]
    data=json.loads(safe(receipt).read_text("utf-8"))
    expected={"sealed50.jsonl","manifest.json","raw9b-50.jsonl","neral10-50.jsonl","summary.json"}
    if data["source_commit"]!=FROZEN or data["complete_pairs"]!=50 or set(data["files_sha256"])!=expected:
        raise ValueError("REFERENCE50_RECEIPT_INVALID")
    records.extend(pin(directory/name,value) for name,value in sorted(data["files_sha256"].items()))
    return {"path":str(directory/"sealed50.jsonl"),"sha256":REF_TASKS,"immutable_files":records}

def check_hard(root):
    directory=root/"v43-obsidianhard"
    records=[pin(directory/"V1/private.jsonl",OH_TASKS),pin(directory/"V1/manifest.json",OH_MANIFEST),pin(directory/"V1-recovery-45457632035/responses.jsonl",OH_RESP)]
    return {"path":records[0]["path"],"sha256":OH_TASKS,"immutable_files":records}

def check_frozen(checkout,installed):
    # Compare actual release Git blob bytes rather than line-ending-converted checkout files.
    if subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()!=FROZEN:
        raise ValueError("FROZEN_SOURCE_PIN_MISMATCH")
    mismatches=0
    exact=0
    for name in FILES:
        blob=subprocess.check_output(["git","-C",str(checkout),"show",FROZEN+":"+name])
        try:
            path=safe(installed/name)
            actual=path.read_bytes()
        except (OSError,ValueError):
            print("FROZEN_FILE",name,"MISSING_OR_UNSAFE",flush=True)
            mismatches+=1
            continue
        if hashlib.sha256(actual).digest()==hashlib.sha256(blob).digest():
            exact+=1
            print("FROZEN_FILE",name,"EXACT_SHA256",flush=True)
        else:
            mismatches+=1
            if actual.replace(b"\r\n",b"\n")==blob.replace(b"\r\n",b"\n"):
                reason="LINE_ENDINGS_ONLY"
            else:
                reason="CONTENT_MISMATCH"
            print("FROZEN_FILE",name,reason,flush=True)
    print("FROZEN_EXACT_COUNT",exact,"OF",len(FILES),flush=True)
    if mismatches:
        raise ValueError("INSTALLED_FROZEN_RUNTIME_MISMATCH")
    return len(FILES)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    if os.environ.get("COMPUTERNAME")!="W39-GAMINGPC" or os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise SystemExit("WRONG_HOST_OR_CPU_ISOLATION")
    if not a.run_id.isascii() or not a.run_id.isdecimal():
        raise SystemExit("RUN_ID_REQUIRED")
    local=safe(Path(os.environ["LOCALAPPDATA"]))
    root=safe(local/"NERAL-private")
    out=safe(root/"v44-rft"/("protected-preflight-"+a.run_id))
    if out.exists():
        raise SystemExit("PREFLIGHT_OUTPUT_EXISTS")
    # Read old data only. No replacement seals generated for old files.
    stage="blind200"
    try:
        protected={"Blind200":check_blind(root)}
        print("BLIND200_421_ORIGINAL_AND_BACKUP_PINS_PASS",flush=True)
        stage="reference50"
        protected["Reference50"]=check_reference(root)
        print("REFERENCE50_HISTORICAL_RECEIPT_PASS",flush=True)
        stage="obsidianhard"
        protected["ObsidianHard"]=check_hard(root)
        print("OBSIDIANHARD_HISTORICAL_PINS_PASS",flush=True)
        stage="frozen_runtime"
        n=check_frozen(safe(a.frozen),safe(local/"NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local"))
        print("FROZEN_RUNTIME_FILES_PASS",n,flush=True)
        stage="candidate_catalog"
        deny=protected_hashes(protected)
        seed=secrets.token_hex(32)
        tasks=catalog(seed,600)
        keys={digest(x["prompt"]) for x in tasks}
        if len(tasks)!=600 or len(keys)!=600 or keys&deny:
            raise ValueError("CATALOG_OVERLAP_OR_COUNT")
        # Save only in the new private directory, no raw outputs in stdout.
        out.mkdir(parents=True,exist_ok=False)
        (out/"seed.txt").write_text(seed,encoding="ascii")
        (out/"protected.json").write_text(canonical(protected),encoding="utf-8")
        write_jsonl(out/"private_catalog.jsonl",tasks)
        write_jsonl(out/"model_prompts.jsonl",({"id":x["id"],"prompt":x["prompt"]} for x in tasks))
        stage="post_integrity"
        if protected_hashes(protected)!=deny:
            raise ValueError("PROTECTED_DENYLIST_CHANGED")
        check_frozen(safe(a.frozen),safe(local/"NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local"))
        print("CATALOG_600_UNIQUE_EXACT_HASH_DISJOINT",flush=True)
        print("PROTECTED_SHA256",sha(out/"protected.json"),flush=True)
        print("CATALOG_SHA256",sha(out/"private_catalog.jsonl"),flush=True)
        print("CPU_ONLY_PREFLIGHT_COMPLETE_GPU_READY_FALSE",flush=True)
    except Exception as exc:
        print("PREFLIGHT_FAILED",stage,type(exc).__name__,flush=True)
        raise SystemExit(1) from None

if __name__=="__main__":
    main()
