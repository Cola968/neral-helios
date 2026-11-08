"""Guarded local NERAL GPU runtime inspection and optional graceful stop.

This can only target one idle, older Python process whose *actual executable*
is precisely the user's pinned NERAL training Python, whose command line
belongs to NERAL, and whose PID was independently observed among CUDA GPU
processes. Never stop training/Actions/unrelated Python. No force-kill.
No private process paths/PIDs or command lines are uploaded to GitHub.
"""
from __future__ import annotations

import argparse
import json
import ntpath
import os
from pathlib import Path
import subprocess
import time

from scripts.neral_gpu_safe_release import (
    parse_gpu_csv, assert_idle, _gpu_capacity, _PROCESS_QUERY,
)

AUTH_INSPECT="INSPECT_NERAL_SCOPED_GPU_PYTHON_READONLY"
AUTH_STOP="STOP_NERAL_PINNED_IDLE_CUDA_PROCESS_ONLY"


def gpu_python_pids(raw:str)->list[int]:
    if not isinstance(raw,str):
        raise ValueError("Missing GPU process snapshot")
    found=set()
    for line in raw.strip().splitlines()[:256]:
        parts=[x.strip() for x in line.split(",")]
        if len(parts)<3 or not parts[0].isdigit():
            continue
        name=ntpath.basename(",".join(parts[1:-1]).strip().strip('"')).casefold()
        if name in ("python.exe","pythonw.exe"):
            pid=int(parts[0])
            if 1<=pid<2**31:
                found.add(pid)
    return sorted(found)


def _normalize(path):
    if not isinstance(path,str) or not path:
        return ""
    return ntpath.normcase(ntpath.normpath(path.strip().strip('"')))


def classify_neral_candidate(proc,expected:str,*,now:float)->dict:
    """Inspect process identity without returning identity details."""
    try:
        exe_path=proc.exe()
        same=_normalize(exe_path)==_normalize(expected)
        args=proc.cmdline()
        if type(args) is not list:
            args=[]
        pinned_launch=bool(args) and _normalize(args[0])==_normalize(expected)
        # Accept only the user's *standard local Python install* as the
        # resolved executable of the exact pinned NERAL venv launcher.
        # Do not trust arbitrary python.exe paths or a forged argv[0].
        home=_normalize(expected)
        for _ in range(4):
            home=ntpath.dirname(home)
        base_prefix=_normalize(ntpath.join(home,"AppData","Local","Programs","Python"))+"\\"
        normalized_exe=_normalize(exe_path)
        relative=(normalized_exe[len(base_prefix):]
                  if normalized_exe.startswith(base_prefix) else "")
        segments=relative.split("\\")
        compatible_python=bool(
            len(segments)==2 and segments[0].startswith("python")
            and segments[0][6:].isdigit()
            and segments[1] in ("python.exe","pythonw.exe")
        )
        trusted_interpreter=bool(same or (pinned_launch and compatible_python))
        tail=" ".join(args[1:]).casefold()
        from_neral=("neral" in tail or "helios" in tail)
        runner=any(fragment in tail for fragment in (
            "actions-runner","_work\\","runner.worker","runner.listener",
        ))
        training=any(fragment in tail for fragment in (
            "train-local-gpu","train_neral","selfverify_train",
            "qlora_train","trainer.py","--do_train",
        ))
        age=now-float(proc.create_time())
        old=300<=age<365*24*3600
        active_cpu=float(proc.cpu_percent(interval=0.15))>5.0
        alive=bool(proc.is_running()) and proc.status() not in ("zombie","dead")
        checked=bool(trusted_interpreter and from_neral and not runner and not training
                     and old and not active_cpu and alive)
    except (OSError,ValueError,TypeError,AttributeError):
        checked=False
        training=False
        active_cpu=True
        old=False
        same=False
        pinned_launch=False
        compatible_python=False
        from_neral=False
        runner=False
        alive=False
    return {
        "verified":checked,"training_process":bool(training),
        "older_than_5min":bool(old),"busy_cpu":bool(active_cpu),
        "pinned_executable":bool(same),
        "pinned_launch_argv0":bool(pinned_launch),
        "compatible_python_binary":bool(compatible_python),
        "project_command":bool(from_neral),
        "runner_process":bool(runner),
        "alive":bool(alive),
    }


def stop_verified(proc,expected:str,*,now:float)->bool:
    """Revalidate twice, then request graceful termination (never force-kill)."""
    first=classify_neral_candidate(proc,expected,now=now)
    if not first["verified"]:
        return False
    second=classify_neral_candidate(proc,expected,now=now)
    if not second["verified"]:
        return False
    proc.terminate()
    proc.wait(timeout=12)
    return True


def _gpu_python_processes()->list[int]:
    snapshot=subprocess.run(
        _PROCESS_QUERY,capture_output=True,text=True,encoding="utf8",
        errors="replace",timeout=12,check=False,
    )
    if snapshot.returncode!=0:
        raise ValueError("GPU process identities are unavailable; refuse stop")
    return gpu_python_pids(snapshot.stdout)


def run(mode:str,authorize:str)->dict:
    if mode not in ("inspect","stop_scoped"):
        raise ValueError("Only inspect or verified NERAL Python stop allowed")
    exact=AUTH_INSPECT if mode=="inspect" else AUTH_STOP
    if authorize!=exact:
        raise ValueError("Exact user authorization required")
    initial=_gpu_capacity()
    if mode=="stop_scoped":
        assert_idle(initial)
    pids=_gpu_python_processes()
    pinned=str(Path.home()/"NERAL-QLoRA-Research"/
               "training-env-py312"/"Scripts"/"python.exe")
    try:
        import psutil
    except ImportError as exc:
        raise ValueError("psutil required for independent process identity verification") from exc
    now=time.time()
    candidates=[]
    observations=[]
    for pid in pids:
        if pid==os.getpid() or pid==os.getppid():
            continue
        try:
            p=psutil.Process(pid)
            info=classify_neral_candidate(p,pinned,now=now)
            observations.append(info)
            if info["verified"]:
                candidates.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied,
                psutil.ZombieProcess):
            continue
    result={
        "mode":mode,
        "status":"NERAL_SCOPED_GPU_PYTHON_DIAGNOSTIC",
        "gpu_before":initial,
        "gpu_python_process_count":len(pids),
        "verified_idle_neral_runtime_count":len(candidates),
        "scope_evidence_counts":{
            key:sum(bool(item[key]) for item in observations)
            for key in ("pinned_executable","pinned_launch_argv0",
                        "compatible_python_binary","project_command",
                        "runner_process","training_process",
                        "older_than_5min","busy_cpu","alive")
        },
        "unrelated_processes_terminated":0,
        "model_weights_modified":False,
        "force_kill_used":False,
    }
    if mode=="stop_scoped":
        if len(candidates)!=1:
            raise ValueError("Must identify exactly one pinned idle NERAL CUDA process")
        assert_idle(_gpu_capacity())
        candidate=candidates[0]
        # No accidental termination of the current Actions Python.
        if candidate.pid in (os.getpid(),os.getppid()):
            raise ValueError("Refuse to terminate current process")
        ok=stop_verified(candidate,pinned,now=time.time())
        if not ok:
            raise ValueError("NERAL process changed or became active; refuse stop")
        time.sleep(3)
        result.update(
            terminated_verified_neral_process_count=1,
            gpu_after=_gpu_capacity(),
        )
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--mode",required=True,choices=("inspect","stop_scoped"))
    parser.add_argument("--authorize",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.mode,args.authorize),sort_keys=True))


if __name__=="__main__":
    main()
