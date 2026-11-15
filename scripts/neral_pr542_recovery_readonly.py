"""Read-only recovery of run 43709097701 private replies; CPU only.

Searches file *names* on the self-hosted Windows runner, validates candidate
content in memory against the published 2/12/10/0 scalar evidence, emits only
redacted counts and anonymous IDs. Does not invoke models or GPU.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
from typing import Iterable

from scripts.neral_pr542_offline_regrade_saved_raw import grade

RUN_ID="43709097701"
SEALED="74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
SKIP_DIRS=frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "site-packages",
    ".cache", "pip", "models", "model_cache", "checkpoints", "ollama",
    "Windows", "Program Files", "Program Files (x86)", "ProgramData",
    "$Recycle.Bin", "System Volume Information", ".vscode",
})

def is_link_or_junction(path:Path)->bool:
    try:
        return path.is_symlink() or (
            hasattr(os.path,"isjunction") and os.path.isjunction(path))
    except (PermissionError,OSError):
        return True

def scan_named_raw_files(roots:Iterable[Path], *, max_entries:int=350000,
                         deadline_seconds:float=150)->tuple[list[Path],dict]:
    """No raw file bytes are read; incomplete scans explicitly say so."""
    if max_entries<1 or deadline_seconds<=0:raise ValueError("Invalid read-only limits")
    starts=time.monotonic()
    found:list[Path]=[]
    seen_dirs=set()
    seen_files=set()
    # LIFO traversal must preserve caller priority (runner TEMP before broad drives).
    stack:list[tuple[Path,int]]=list(reversed([(Path(p),0) for p in roots]))
    visited=0
    denied=0
    timedout=False
    capped=False
    while stack:
        if time.monotonic()-starts>=deadline_seconds:
            timedout=True
            break
        current,depth=stack.pop()
        key=os.path.normcase(os.path.abspath(current))
        if key in seen_dirs:continue
        seen_dirs.add(key)
        if is_link_or_junction(current):continue
        try:
            with os.scandir(current) as listing:
                for entry in listing:
                    visited+=1
                    if visited>max_entries:
                        capped=True
                        break
                    if time.monotonic()-starts>=deadline_seconds:
                        timedout=True
                        break
                    try:
                        if entry.is_symlink() or (
                            hasattr(os.path,"isjunction") and
                            os.path.isjunction(entry.path)
                        ):continue
                        if entry.is_file(follow_symlinks=False):
                            if entry.name.lower()=="private-results.jsonl":
                                token=os.path.normcase(os.path.abspath(entry.path))
                                if token not in seen_files:
                                    seen_files.add(token)
                                    found.append(Path(entry.path))
                        elif (entry.is_dir(follow_symlinks=False)
                              and depth<18
                              and entry.name not in SKIP_DIRS
                              and not entry.name.startswith(".git")):
                            stack.append((Path(entry.path),depth+1))
                    except (OSError,PermissionError):
                        denied+=1
                if capped or timedout:break
        except (OSError,PermissionError):
            denied+=1
    return sorted(found),{
      "visited_entries":visited, "visited_directories":len(seen_dirs),
      "unreadable_entries":denied, "candidate_files":len(found),
      "incomplete":bool(timedout or capped),
      "time_limit_exceeded":timedout, "entry_limit_exceeded":capped,
    }

def local_runner_roots()->list[Path]:
    env=os.environ
    home=Path(env.get("USERPROFILE",str(Path.home())))
    roots=[]
    # Check the original ephemeral evidence directory before any large tree.
    if env.get("RUNNER_TEMP"):
        roots.append(Path(env["RUNNER_TEMP"]))
    roots += [
      Path(r"C:\actions-runner\_work\_temp"),
      Path(r"C:\actions-runner\_temp"),
      home/"NERAL-PR542-GPU-RAW",
      home/"NERAL-QLoRA-Research",
      home/"Documents"/"NERAL",
      home/"Desktop",
      home/"Downloads",
      Path(r"D:\neral-qlora"),
      Path(r"D:\NERAL"),
      Path(r"D:\neral"),
      Path(r"D:\actions-runner"),
      Path(r"D:\benchmarks"),
      Path(r"D:\datasets"),
    ]
    # Only after priority paths do we search broad runner/user/disk trees.
    roots += [Path(r"C:\actions-runner"),Path(r"C:\Users")]
    for name in ("RUNNER_WORKSPACE","GITHUB_WORKSPACE","TEMP","TMP"):
        if env.get(name):
            v=Path(env[name]);roots.extend([v,v.parent])
    roots.append(Path("D:\\"))

    unique={}
    for root in roots:
        unique.setdefault(os.path.normcase(os.path.abspath(root)),root)
    return list(unique.values())

def main()->None:
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",required=True,type=Path)
    cli.add_argument("--max-entries",type=int,default=350000)
    cli.add_argument("--deadline-seconds",type=float,default=150)
    a=cli.parse_args()
    if a.out.exists():raise FileExistsError("Never replace any result")
    found,scan=scan_named_raw_files(local_runner_roots(),
        max_entries=a.max_entries,deadline_seconds=a.deadline_seconds)
    candidate_errors=0
    validated=[]
    for path in found:
        try:
            if path.stat().st_size>32_000_000:raise ValueError("oversize")
            report=grade(path)  # Validates 100 cases and exact original strict scalars
            validated.append(report)
        except (OSError,UnicodeError,ValueError,TypeError,KeyError,
                json.JSONDecodeError):
            candidate_errors+=1
    if len(validated)==1:
        result=dict(validated[0])
        result["status"]="original_raw_recovered_readonly"
    elif len(validated)>1:
        result={"status":"multiple_verified_raw_copies_needs_manual_identity_check"}
    else:
        result={"status":"original_raw_not_found_in_scanned_scope"}
    result.update({
      "source_run":RUN_ID, "sealed_sha256":SEALED,
      "candidates_invalid":candidate_errors,
      "verified_original_raw_file_count":len(validated),
      "scan_summary":scan, "inference_calls":0,"gpu_calls":0,
      "raw_outputs_uploaded":False,"merged_to_main":False,
      "interpretation":"Not found is limited to scanned readable paths; no new inference was executed",
    })
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write("\n")
    print("READ_ONLY_RECOVERY_STATUS "+result["status"])
    print("READ_ONLY_RECOVERY_SCALARS "+json.dumps(result,sort_keys=True))
if __name__=="__main__":
    main()
