"""Inspect Windows RTX VRAM and unload only installed, trusted Qwen Ollama models.

Run only through the local explicitly authorized GitHub Actions workflow.
No user-data uploads, arbitrary PID kills, generic process stopping,
GPU reset, external network, model downloads, or weight changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import ntpath
import re
import subprocess
import time
import urllib.error
import urllib.request

_GPU_QUERY=["nvidia-smi","--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"]
_PROCESS_QUERY=["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"]
_TAG=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")
_TRUSTED=frozenset({"qwen3:8b","qwen3.5:9b"})
_LOCAL="http://127.0.0.1:11434"
_AUTHORIZE="UNLOAD_NERAL_OLLAMA_VRAM_ONLY"
_INSPECT="INSPECT_NERAL_GPU_PROCESSES_READONLY"


def parse_gpu_csv(csv_text:str)->dict:
    if type(csv_text) is not str:
        raise ValueError("Missing GPU metrics")
    parts=csv_text.strip().splitlines()[0].split(",") if csv_text.strip() else []
    if len(parts)!=4:
        raise ValueError("Unrecognized GPU metrics")
    try:
        vals=[int(v.strip()) for v in parts]
    except (ValueError,TypeError) as exc:
        raise ValueError("Invalid GPU metrics") from exc
    free,used,total,util=vals
    if not (0<=free<=total and 0<=used<=total and 0<total<=200000
            and 0<=util<=100 and free+used<=total+4096):
        raise ValueError("Impossible GPU capacity readings")
    return {"free_mib":free,"used_mib":used,"total_mib":total,
            "util_percent":util,"benchmark_ready":free>=9000 and util<=20}


def assert_idle(gpu:dict)->None:
    if type(gpu) is not dict or type(gpu.get("util_percent")) is not int:
        raise ValueError("No trusted GPU utilization")
    if gpu["util_percent"]>20:
        raise ValueError("GPU is actively busy: refuse unload")


def parse_gpu_process_rows(raw:str)->dict:
    """Never return paths, process names, PIDs, command lines or user data."""
    result={"ollama_processes":0,"python_processes":0,
            "other_processes":0,"gpu_process_reported_mib":0,
            "unreported_memory_entries":0}
    if not isinstance(raw,str):
        return result
    rows=raw.strip().splitlines()[:256]
    for row in rows:
        fields=[x.strip() for x in row.split(",")]
        if len(fields)<3 or not fields[0].isdigit():
            continue
        name=ntpath.basename(",".join(fields[1:-1]).strip().strip('"')).casefold()
        if name in ("ollama.exe","ollama_llama_server.exe","ollama"):
            result["ollama_processes"]+=1
        elif name in ("python.exe","pythonw.exe","python"):
            result["python_processes"]+=1
        else:
            result["other_processes"]+=1
        last=fields[-1].strip()
        cleaned=re.sub(r"\s*(?:MiB|MB)\s*$","",last,flags=re.I)
        if cleaned.isdigit():
            result["gpu_process_reported_mib"]+=int(cleaned)
        else:
            result["unreported_memory_entries"]+=1
    return result


def select_ollama_targets(models:list)->list[str]:
    if type(models) is not list or len(models)>64:
        raise ValueError("Untrusted local Ollama inventory")
    seen=set()
    selected=[]
    for row in models:
        if type(row) is not dict or type(row.get("name")) is not str:
            raise ValueError("Missing local Ollama model tag")
        name=row["name"]
        if not _TAG.fullmatch(name) or name in seen:
            raise ValueError("Unexpected or duplicate local model tag")
        seen.add(name)
        if name in _TRUSTED:
            selected.append(name)
    return sorted(selected)


def unload_known_models(models,post,capacity)->dict:
    assert_idle(capacity)
    selected=select_ollama_targets(models)
    for tag in selected:
        reply=post(tag)
        if type(reply) is not dict or reply.get("done") is not True:
            raise ValueError("Ollama did not confirm model unload")
        # Do not stop or signal any other model, PID, app or service.
    return {"unloaded_count":len(selected),"other_programs_modified":False}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect from local Ollama forbidden")


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())


def _local_json(endpoint:str, payload:dict|None=None)->dict:
    if endpoint not in ("/api/ps","/api/generate"):
        raise ValueError("Unsupported localhost-only endpoint")
    if payload is None and endpoint!="/api/ps":
        raise ValueError("Only local model inventory permits GET")
    if payload is not None and endpoint!="/api/generate":
        raise ValueError("Only fixed Ollama keep_alive unload permits POST")
    if payload is not None and (
        type(payload) is not dict or
        set(payload)!={"model","keep_alive"} or
        payload["model"] not in _TRUSTED or payload["keep_alive"]!=0):
        raise ValueError("No arbitrary Ollama API actions allowed")
    data=None if payload is None else json.dumps(payload).encode("utf8")
    req=urllib.request.Request(
        _LOCAL+endpoint,data=data,
        headers={"Content-Type":"application/json"},
        method="GET" if data is None else "POST",
    )
    with _opener().open(req,timeout=18) as response:
        if response.status!=200:
            raise ValueError("Local Ollama refused request")
        raw=response.read(65537)
        if len(raw)>65536:
            raise ValueError("Unexpectedly large localhost Ollama response")
        answer=json.loads(raw.decode("utf8"))
    if type(answer) is not dict:
        raise ValueError("Ollama response was not a JSON dictionary")
    return answer


def _gpu_capacity():
    r=subprocess.run(_GPU_QUERY,capture_output=True,text=True,
                     encoding="utf8",errors="replace",timeout=15,
                     check=False)
    if r.returncode!=0:
        raise ValueError("Local nvidia-smi failed")
    return parse_gpu_csv(r.stdout)


def _gpu_processes():
    try:
        r=subprocess.run(_PROCESS_QUERY,capture_output=True,text=True,
                         encoding="utf8",errors="replace",timeout=10,
                         check=False)
        return parse_gpu_process_rows(r.stdout if r.returncode==0 else "")
    except (FileNotFoundError,TimeoutError,subprocess.TimeoutExpired):
        return parse_gpu_process_rows("")


def _ollama_models():
    answer=_local_json("/api/ps")
    rows=answer.get("models")
    if type(rows) is not list:
        raise ValueError("Ollama /api/ps has no model list")
    chosen=select_ollama_targets(rows)
    amount=0
    for row in rows:
        mem=row.get("size_vram",0)
        if type(mem) is not int or mem<0 or mem>150_000_000_000:
            raise ValueError("Invalid model VRAM accounting")
        amount+=mem
    return rows,{"resident_count":len(rows),"resident_vram_mib":round(amount/(1024**2)),
                 "trusted_qwen_resident_count":len(chosen),
                 "other_models_present":len(chosen)!=len(rows)}


def run(mode,authorize):
    if mode not in ("inspect","unload_ollama"):
        raise ValueError("Only read-only inspect or trusted-model unload supported")
    expected=_INSPECT if mode=="inspect" else _AUTHORIZE
    if authorize!=expected:
        raise ValueError("Explicit exact user authorization missing")
    before=_gpu_capacity()
    processes=_gpu_processes()
    rows,models=_ollama_models()
    outcome={"mode":mode,"gpu_before":before,"gpu_processes":processes,
             "ollama_before":models,"no_unknown_processes_stopped":True,
             "no_model_weights_modified":True,"no_external_network":True}
    if mode=="unload_ollama":
        outcome.update(unload_known_models(
            rows,lambda tag:_local_json("/api/generate",{"model":tag,"keep_alive":0}),
            before))
        time.sleep(2)
        after=_gpu_capacity()
        remaining,after_models=_ollama_models()
        leftover=set(select_ollama_targets(remaining))
        if leftover&set(select_ollama_targets(rows)):
            raise ValueError("Ollama reported the model still resident after unload")
        outcome.update(gpu_after=after,ollama_after=after_models,
                       benchmark_ready=after["benchmark_ready"],
                       gpu_processes_after=_gpu_processes())
    return outcome


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--mode",choices=("inspect","unload_ollama"),required=True)
    parser.add_argument("--authorize",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.mode,args.authorize),sort_keys=True))


if __name__=="__main__":
    main()
