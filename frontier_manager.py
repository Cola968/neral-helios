"""Lifecycle manager for the local Frontier 350 sidecar.

Starts only a registered local bundle. The browser cannot provide executable paths,
bundle paths or ports; all values originate from the validated model catalog.
"""
from __future__ import annotations

import atexit,os,socket,subprocess,sys,threading,time
from pathlib import Path

from frontier_sidecar import FrontierUnavailable,probe

_lock=threading.Lock()
_processes={}
_logs=[]

def _python():
    override=os.environ.get("HELIOS_FRONTIER_PYTHON")
    if override:
        path=Path(override).expanduser().resolve()
        if not path.is_file():raise FrontierUnavailable("HELIOS_FRONTIER_PYTHON ist ungültig.")
        return str(path)
    return sys.executable

def _key(row):
    return (row.get("model_sha256"),row.get("runtime_port"))

def _stop_process(proc):
    if proc is None or proc.poll() is not None:return
    try:proc.terminate();proc.wait(timeout=3)
    except Exception:
        try:proc.kill();proc.wait(timeout=1)
        except Exception:pass

def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1",int(port)),timeout=0.2):return True
    except OSError:return False

def ensure_running(row,state_root,runtime_script,timeout=45.0):
    try:return probe(row,timeout=0.35)
    except FrontierUnavailable:pass
    paths=row.get("paths")
    if not isinstance(paths,tuple) or len(paths)!=2:raise FrontierUnavailable("Frontier-Bundle ist nicht registriert.")
    bundle=Path(paths[0]).resolve().parent
    runtime_script=Path(runtime_script).resolve()
    if not runtime_script.is_file():raise FrontierUnavailable("Frontier-Runtime-Datei fehlt.")
    port=row.get("runtime_port");context=row.get("context_tokens") or 512
    device=os.environ.get("HELIOS_FRONTIER_DEVICE","auto").lower()
    if device not in {"auto","cpu","cuda"}:device="auto"
    key=_key(row)
    with _lock:
        proc=_processes.get(key)
        if proc is None or proc.poll() is not None:
            # Only replace a sidecar that this Helios process itself owns.
            # Never terminate an arbitrary process just because it uses the port.
            for other_key,other_proc in list(_processes.items()):
                if other_key!=key and other_key[1]==port:
                    _stop_process(other_proc);_processes.pop(other_key,None)
            if _port_open(port):
                raise FrontierUnavailable("Frontier-Runtime-Port ist durch einen anderen Prozess belegt.")
            root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True)
            safe_id="".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in row.get("label","frontier"))[:48]
            out=(root/f"{safe_id}-runtime.log").open("ab",buffering=0)
            err=(root/f"{safe_id}-runtime-error.log").open("ab",buffering=0)
            _logs.extend((out,err))
            env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
            if device!="cpu" and env.get("CUDA_VISIBLE_DEVICES","")=="":
                env.pop("CUDA_VISIBLE_DEVICES",None)
            cmd=[_python(),"-X","utf8",str(runtime_script),"--bundle",str(bundle),
                 "--port",str(port),"--device",device,"--context-tokens",str(context)]
            try:
                proc=subprocess.Popen(cmd,cwd=str(runtime_script.parent),env=env,stdin=subprocess.DEVNULL,
                                      stdout=out,stderr=err,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            except OSError as exc:
                raise FrontierUnavailable("Frontier-Runtime konnte nicht gestartet werden.") from exc
            _processes[key]=proc
    deadline=time.monotonic()+timeout
    last=None
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            raise FrontierUnavailable("Frontier-Runtime wurde beim Start beendet. Prüfe das lokale Runtime-Log.")
        try:return probe(row,timeout=0.5)
        except FrontierUnavailable as exc:last=exc;time.sleep(0.2)
    raise FrontierUnavailable("Frontier-Runtime wurde nicht bereit.") from last

def stop_all():
    with _lock:
        processes=list(_processes.values());_processes.clear()
    for proc in processes:_stop_process(proc)
    for handle in _logs:
        try:handle.close()
        except Exception:pass
    _logs.clear()

atexit.register(stop_all)
