"""Lifecycle manager for the local Frontier 350 sidecar.

Starts only a registered local bundle. The browser cannot provide executable paths,
bundle paths or ports; all values originate from the validated model catalog.
"""
from __future__ import annotations

import atexit,os,subprocess,sys,threading,time
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
            root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True)
            safe_id="".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in row.get("label","frontier"))[:48]
            out=(root/f"{safe_id}-runtime.log").open("ab",buffering=0)
            err=(root/f"{safe_id}-runtime-error.log").open("ab",buffering=0)
            _logs.extend((out,err))
            env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
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
    for proc in processes:
        if proc.poll() is None:
            try:proc.terminate();proc.wait(timeout=3)
            except Exception:
                try:proc.kill()
                except Exception:pass
    for handle in _logs:
        try:handle.close()
        except Exception:pass
    _logs.clear()

atexit.register(stop_all)
