"""Local RTX/CUDA qualification for a sanitized Frontier 350 deployment bundle.

This is a runtime qualification only. It does not train, modify, promote, upload,
or publish model weights. Reports contain technical metrics and hashes, not model text.
"""
from __future__ import annotations

import argparse,hashlib,json,os,socket,statistics,subprocess,sys,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path

from frontier350_contract import FRONTIER350_RUNTIME
from frontier_sidecar import FrontierUnavailable,cancel,probe,stream_chat
from frontier_runtime import read_sha,verify_bundle

DEFAULT_MODEL_ID="neral-arctus-350"
PROMPTS=(
    "Antworte in einem kurzen Satz: Was ist ein Gravitationsfeld?",
    "Nenne drei Eigenschaften eines stabilen lokalen KI-Systems.",
    "Erkläre knapp den Unterschied zwischen Masse und Gewichtskraft.",
)

def free_port()->int:
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1",0))
        port=s.getsockname()[1]
    if port<1024:raise RuntimeError("no safe local port")
    return port

def sha_text(value:str)->str:
    return hashlib.sha256(value.encode("utf-8","replace")).hexdigest()

def nvidia_snapshot():
    cmd=["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"]
    try:
        raw=subprocess.check_output(cmd,text=True,encoding="utf-8",errors="replace",timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        return None
    if not raw:return None
    first=raw.splitlines()[0]
    parts=[x.strip() for x in first.split(",")]
    keys=("name","driver_version","memory_total_mib","memory_used_mib","temperature_c","power_w")
    return dict(zip(keys,parts))

def wait_health(row,proc,timeout=90.0):
    deadline=time.monotonic()+timeout;last=None
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Frontier runtime exited with code {proc.returncode}")
        try:return probe(row,timeout=0.5)
        except FrontierUnavailable as exc:last=exc;time.sleep(0.2)
    raise RuntimeError("Frontier runtime did not become ready") from last

def run_stream(row,prompt,budget):
    first_progress=None;progress_count=0
    start=time.perf_counter()
    def on_event(event):
        nonlocal first_progress,progress_count
        progress_count+=1
        if first_progress is None:first_progress=time.perf_counter()
    request_id=uuid.uuid4().hex
    result,health=stream_chat(row,[{"role":"user","content":prompt}],budget,request_id,on_event,timeout=180.0)
    wall=time.perf_counter()-start
    answer=result.get("answer","")
    return {
        "request_id":request_id,
        "generated_tokens":int(result.get("generated_tokens",0)),
        "input_tokens":result.get("input_tokens"),
        "finish_reason":result.get("finish_reason"),
        "runtime_seconds":float(result.get("seconds",0)),
        "wall_seconds":round(wall,4),
        "first_token_ms":None if first_progress is None else round((first_progress-start)*1000,2),
        "tokens_per_second":float(result.get("tokens_per_second",0)),
        "peak_vram_mib":float(result.get("peak_vram_mib",0)),
        "vram_allocated_mib":float(result.get("vram_allocated_mib",0)),
        "progress_events":progress_count,
        "answer_chars":len(answer),
        "answer_sha256":sha_text(answer),
        "device":health.get("device"),
    }

def run_cancel(row):
    holder={};started=threading.Event();done=threading.Event()
    request_id=uuid.uuid4().hex
    def worker():
        try:
            def progress(_event):started.set()
            holder["result"],holder["health"]=stream_chat(
                row,[{"role":"user","content":"Schreibe eine ausführliche fortlaufende Erklärung über Planetenbahnen."}],
                256,request_id,progress,timeout=180.0)
        except Exception as exc:holder["error"]=f"{type(exc).__name__}: {exc}"
        finally:done.set()
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    accepted=False;deadline=time.monotonic()+5
    while time.monotonic()<deadline and not done.is_set():
        if started.is_set() or time.monotonic()+4.7<deadline:
            try:
                if cancel(row,request_id,timeout=1.0):
                    accepted=True;break
            except FrontierUnavailable:pass
        time.sleep(0.03)
    thread.join(timeout=15)
    result=holder.get("result") or {}
    return {
        "request_id":request_id,
        "cancel_accepted":accepted,
        "stream_finished":done.is_set(),
        "finish_reason":result.get("finish_reason"),
        "generated_tokens":result.get("generated_tokens"),
        "error":holder.get("error"),
    }

def summary(runs):
    tps=[r["tokens_per_second"] for r in runs if r["tokens_per_second"]>0]
    ttft=[r["first_token_ms"] for r in runs if r["first_token_ms"] is not None]
    peaks=[r["peak_vram_mib"] for r in runs if r["peak_vram_mib"]>0]
    return {
        "median_decode_tokens_per_second":round(statistics.median(tps),2) if tps else None,
        "median_first_token_ms":round(statistics.median(ttft),2) if ttft else None,
        "max_peak_vram_mib":round(max(peaks),1) if peaks else None,
        "responses_with_tokens":sum(r["generated_tokens"]>0 for r in runs),
    }

def qualify(bundle:Path,runtime_script:Path,out_dir:Path,port:int,context_tokens:int):
    bundle=bundle.resolve(strict=True);runtime_script=runtime_script.resolve(strict=True)
    model,tok,model_sha,tok_sha=verify_bundle(bundle)
    row={"runtime_family":FRONTIER350_RUNTIME,"runtime_port":port,
         "model_sha256":model_sha,"tokenizer_sha256":tok_sha}
    out_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path=out_dir/f"frontier350-runtime-{stamp}.log"
    stderr_path=out_dir/f"frontier350-runtime-{stamp}.err.log"
    before=nvidia_snapshot()
    env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
    cmd=[sys.executable,"-X","utf8",str(runtime_script),"--bundle",str(bundle),
         "--port",str(port),"--device","cuda","--context-tokens",str(context_tokens)]
    with stdout_path.open("wb") as stdout,stderr_path.open("wb") as stderr:
        proc=subprocess.Popen(cmd,cwd=str(runtime_script.parent),env=env,stdin=subprocess.DEVNULL,
                              stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            health=wait_health(row,proc)
            if health.get("device")!="cuda":raise RuntimeError("qualification requires CUDA")
            if health.get("kv_cache") is not True:raise RuntimeError("KV cache is not active")
            if health.get("streaming") is not True or health.get("cancel_supported") is not True:
                raise RuntimeError("stream/cancel runtime contract missing")
            warmup=run_stream(row,"Antworte nur mit: bereit",16)
            runs=[run_stream(row,prompt,64) for prompt in PROMPTS]
            cancel_result=run_cancel(row)
            after=nvidia_snapshot()
            _m2,_t2,model_sha_after,tok_sha_after=verify_bundle(bundle)
            bundle_unchanged=(model_sha_after==model_sha and tok_sha_after==tok_sha)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait(timeout=2)
    metrics=summary(runs)
    gates={
        "cuda":health.get("device")=="cuda",
        "kv_cache":health.get("kv_cache") is True,
        "streaming":health.get("streaming") is True,
        "cancel_supported":health.get("cancel_supported") is True,
        "cancel_accepted":cancel_result["cancel_accepted"],
        "cancel_finished":cancel_result["stream_finished"],
        "cancel_effective":cancel_result["finish_reason"]=="cancelled",
        "bundle_unchanged":bundle_unchanged,
        "at_least_one_token_response":metrics["responses_with_tokens"]>=1,
    }
    report={
        "schema":"frontier350-runtime-qualification-v1",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "qualified":all(gates.values()),
        "quality_promotion":False,
        "training_started":False,
        "bundle_unchanged":bundle_unchanged,
        "python":sys.version.split()[0],
        "bundle":{"path":str(bundle),"deployment_sha256":model_sha,"tokenizer_sha256":tok_sha},
        "health":health,
        "nvidia_before":before,
        "warmup":warmup,
        "runs":runs,
        "cancel":cancel_result,
        "metrics":metrics,
        "gates":gates,
        "nvidia_after":after,
        "runtime_logs":{"stdout":str(stdout_path),"stderr":str(stderr_path)},
    }
    report_path=out_dir/f"frontier350-qualification-{stamp}.json"
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report_path,report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path("deployment/installed_models")/DEFAULT_MODEL_ID)
    p.add_argument("--runtime-script",type=Path,default=Path(__file__).with_name("frontier_runtime.py"))
    p.add_argument("--out-dir",type=Path,default=Path("local_state/qualifications"))
    p.add_argument("--port",type=int,default=0)
    p.add_argument("--context-tokens",type=int,default=512)
    a=p.parse_args(argv)
    port=a.port or free_port()
    if not 1024<=port<=65535:raise SystemExit("invalid port")
    report_path,report=qualify(a.bundle,a.runtime_script,a.out_dir,port,a.context_tokens)
    print(json.dumps({
        "event":"frontier350_runtime_qualification",
        "qualified":report["qualified"],
        "gpu":report["health"].get("gpu"),
        **report["metrics"],
        "cancel":report["cancel"],
        "report":str(report_path),
    },ensure_ascii=False,indent=2))
    raise SystemExit(0 if report["qualified"] else 2)

if __name__=="__main__":main()
