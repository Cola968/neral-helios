"""Opt-in Windows Ollama restart on proven 0-VRAM CUDA fallback.

One native, user-owned localhost Ollama listener only. No arbitrary process
kills, no cloud calls, no model downloads. Read-only diagnostics are recorded
before any mutation. The executable must be in the normal per-user install.
"""
from __future__ import annotations

from collections import Counter
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


CONSENT = "NERAL_RESTART_OLLAMA_ON_GPU_FALLBACK"


class RecoveryRefused(RuntimeError):
    """Failure codes are safe to report; raw local logs are never uploaded."""


def recovery_eligible(*, size_vram, free_mib, consent) -> bool:
    return (type(size_vram) is int and size_vram == 0
            and type(free_mib) is int and free_mib > 8192
            and consent == CONSENT)


def safe_log_signals(content: str) -> dict:
    """Read at most last 60 lines and summarize keywords, NEVER publish lines."""
    lines = content.splitlines()[-60:]
    patterns = {
        "inference_compute": re.compile(r"inference compute", re.I),
        "cuda": re.compile(r"\bcuda\b", re.I),
        "library_cpu": re.compile(r"library\s*=\s*[\"']?cpu", re.I),
        "library_cuda": re.compile(r"library\s*=\s*[\"']?cuda", re.I),
        "gpu": re.compile(r"\bgpu\b", re.I),
        "vram": re.compile(r"\bvram\b", re.I),
        "offload": re.compile(r"offload", re.I),
        "error": re.compile(r"\berror\b", re.I),
    }
    counts = Counter()
    seen = 0
    for line in lines:
        signals = [name for name, rx in patterns.items() if rx.search(line)]
        if signals:
            seen += 1
            counts.update(signals)
    return {
        "scanned_lines": len(lines),
        "technical_matching_lines": seen,
        "signal_types": sorted(counts),
        "signal_counts": dict(sorted(counts.items())),
        "raw_lines_uploaded": False,
    }


def environment_signals(env: dict) -> dict:
    """No environment variable values or secrets are exposed to CI."""
    device = env.get("CUDA_VISIBLE_DEVICES")
    library = env.get("OLLAMA_LLM_LIBRARY")
    if device is None:
        policy = "unset"
    elif device.strip() in ("-1", ""):
        policy = "disabled"
    elif re.fullmatch(r"\d{1,2}(?:,\d{1,2}){0,3}", device):
        policy = "specified"
    else:
        policy = "other_or_invalid"
    if library is None:
        lib = "unset"
    elif library.strip().lower() == "cpu":
        lib = "cpu"
    elif library.strip().lower().startswith("cuda"):
        lib = "cuda"
    else:
        lib = "other"
    return {
        "cuda_visible_devices_set": device is not None,
        "cuda_device_policy": policy,
        "llm_library_policy": lib,
        "raw_env_values_uploaded": False,
    }


def _normal_path(value: str) -> str:
    return value.replace("/", "\\").rstrip("\\").lower()


def validate_native_listener(data: dict, approved_install: str,
                             current_user: str) -> list[int]:
    """Only user-owned binary in %LOCALAPPDATA%/Programs/Ollama may be touched."""
    if not isinstance(data, dict) or not isinstance(data.get("processes"), list):
        raise RecoveryRefused("process_inventory_unavailable")
    processes, listener_pids = data["processes"], data.get("listener_pids")
    service_pids = data.get("service_pids")
    clients = data.get("established_clients")
    if not isinstance(listener_pids, list) or len(listener_pids) != 1:
        raise RecoveryRefused("ambiguous_ollama_listener")
    if not isinstance(service_pids, list):
        raise RecoveryRefused("process_inventory_unavailable")
    if not 1 <= len(processes) <= 3:
        raise RecoveryRefused("unexpected_ollama_process_count")
    # Explicit consent includes interruption of existing localhost chats.
    # It never permits interruption of remote users or a public listener.
    listener_addresses = data.get("listener_addresses")
    if (not isinstance(listener_addresses, list) or not listener_addresses
            or any(type(addr) is not str or addr not in ("127.0.0.1", "::1")
                   for addr in listener_addresses)):
        raise RecoveryRefused("ollama_listener_not_loopback")
    if data.get("all_clients_loopback") is not True:
        raise RecoveryRefused("nonlocal_ollama_client")
    if type(clients) is not int or clients < 0:
        raise RecoveryRefused("client_inventory_unavailable")
    if clients > 8:
        raise RecoveryRefused("too_many_connected_ollama_clients")
    owner = listener_pids[0]
    if type(owner) is not int or owner <= 0 or owner in service_pids:
        raise RecoveryRefused("foreign_or_service_listener")
    root = _normal_path(approved_install) + "\\"
    by_pid = {}
    for p in processes:
        if not isinstance(p, dict) or set(p) != {"pid", "name", "path", "user"}:
            raise RecoveryRefused("process_metadata_incomplete")
        pid, name, path, user = p["pid"], p["name"], p["path"], p["user"]
        if (type(pid) is not int or pid <= 0 or pid in by_pid
                or name not in ("ollama.exe", "ollama app.exe")
                or type(path) is not str or type(user) is not str
                or _normal_path(path) != root + name
                or user.casefold().split("\\")[-1] != current_user.casefold()):
            raise RecoveryRefused("unexpected_ollama_process")
        by_pid[pid] = p
    if owner not in by_pid or any(pid in service_pids for pid in by_pid):
        raise RecoveryRefused("ollama_listener_not_user_native")
    # Stop the tray application first to avoid its immediate server respawn,
    # then the local daemon owner. All PIDs have been user/path validated.
    app_pids = sorted(pid for pid,p in by_pid.items()
                      if p["name"] == "ollama app.exe" and pid != owner)
    other_pids = sorted(pid for pid,p in by_pid.items()
                        if p["name"] == "ollama.exe" and pid != owner)
    return app_pids + other_pids + [owner]


def _windows_inventory() -> dict:
    """PowerShell queries native Windows PID ownership without uploading PIDs."""
    ps = r"""
$ErrorActionPreference='Stop'
$names=@('ollama.exe','ollama app.exe')
$rows=@(Get-CimInstance Win32_Process | Where-Object { $names -contains $_.Name } |
  ForEach-Object {
    $own=Invoke-CimMethod -InputObject $_ -MethodName GetOwner
    [pscustomobject]@{
      pid=[int]$_.ProcessId;name=[string]$_.Name;
      path=[string]$_.ExecutablePath;user=[string]$own.User
    }
  })
$listen=@(Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction Stop)
$owners=@($listen | Select-Object -ExpandProperty OwningProcess -Unique)
$addresses=@($listen | Select-Object -ExpandProperty LocalAddress -Unique)
$est=@(Get-NetTCPConnection -LocalPort 11434 -State Established -ErrorAction Stop)
$nonlocal=@($est | Where-Object {
  $_.LocalAddress -notin @('127.0.0.1','::1') -or
  $_.RemoteAddress -notin @('127.0.0.1','::1')
})
$svc=@(Get-CimInstance Win32_Service |
  Where-Object { $_.ProcessId -gt 0 } | Select-Object -ExpandProperty ProcessId)
[pscustomobject]@{
  processes=$rows
  listener_pids=@($owners)
  listener_addresses=@($addresses)
  established_clients=@($est).Count
  all_clients_loopback=(@($nonlocal).Count -eq 0)
  service_pids=@($svc)
} | ConvertTo-Json -Depth 6 -Compress
"""
    try:
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=15, check=False)
        if p.returncode != 0 or len(p.stdout) > 16000:
            raise RecoveryRefused("windows_process_audit_failed")
        data = json.loads(p.stdout)
        if not isinstance(data, dict):
            raise ValueError("not dict")
        return data
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RecoveryRefused("windows_process_audit_failed") from exc


def _read_ollama_log() -> dict:
    local_app = os.environ.get("LOCALAPPDATA")
    if not local_app:
        return {"status": "local_appdata_unavailable"}
    logfile = Path(local_app) / "Ollama" / "server.log"
    if not logfile.is_file():
        return {"status": "server_log_unavailable"}
    # Read last 64 KiB, keep only 60 last lines. No arbitrary path or text logs.
    try:
        with logfile.open("rb") as stream:
            stream.seek(0, 2)
            length = stream.tell()
            stream.seek(max(0, length - 65536))
            data = stream.read(65536).decode("utf8", errors="replace")
        return {"status": "read", **safe_log_signals(data)}
    except OSError:
        return {"status": "server_log_read_error"}


def _check_tags() -> bool:
    """Only local HTTP. Caller must not use a proxy or external redirects."""
    from neral_agent_autonomy import local_only_opener
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags",
                                     method="GET")
        with local_only_opener().open(req, timeout=2.0) as response:
            packet = json.loads(response.read(65537))
        return isinstance(packet, dict) and isinstance(packet.get("models"), list)
    except (OSError, ValueError, TypeError):
        return False


def _terminate_native_owner(owner_pid: int, approved_install: str) -> dict:
    """CloseMainWindow first; force only same PID/path after 4 sec if necessary."""
    # Recheck PID/path immediately before action to avoid stale PID reuse.
    command = r"""
$ErrorActionPreference='Stop'
$pidToStop=[int]$env:NERAL_RECOVERY_OWNER_PID
$root=$env:NERAL_RECOVERY_INSTALL_DIR.TrimEnd('\')
$p=Get-CimInstance Win32_Process -Filter ("ProcessId="+$pidToStop)
if ($null -eq $p -or $p.Name -notin @('ollama.exe','ollama app.exe') -or
    -not $p.ExecutablePath.StartsWith(($root+'\'),[StringComparison]::OrdinalIgnoreCase)) {
  throw 'Native Ollama owner changed'
}
$own=Invoke-CimMethod -InputObject $p -MethodName GetOwner
if ($own.User -ne $env:USERNAME) { throw 'Ollama owner changed' }
$proc=Get-Process -Id $pidToStop -ErrorAction Stop
$graceful=$proc.CloseMainWindow()
if ($graceful) { $proc.WaitForExit(4000) | Out-Null }
$forced=$false
if (-not $proc.HasExited) {
  # Constrained to the already-verified per-user Ollama server owner only.
  Stop-Process -Id $pidToStop -Force -ErrorAction Stop
  $forced=$true
}
[pscustomobject]@{stopped=$true;forced=$forced} | ConvertTo-Json -Compress
"""
    try:
        safe_env = os.environ.copy()
        safe_env["NERAL_RECOVERY_OWNER_PID"] = str(owner_pid)
        safe_env["NERAL_RECOVERY_INSTALL_DIR"] = approved_install
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             command],
            env=safe_env, capture_output=True, text=True,
            timeout=12, check=False,
        )
        if result.returncode != 0:
            raise RecoveryRefused("native_ollama_stop_failed")
        result_obj = json.loads(result.stdout)
        if result_obj.get("stopped") is not True:
            raise RecoveryRefused("native_ollama_stop_unconfirmed")
        return {"stopped": True, "forced_after_graceful_attempt":
                result_obj.get("forced") is True}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise RecoveryRefused("native_ollama_stop_failed") from exc


def _spawn_new_ollama(approved_install: str) -> None:
    """Detached, local-only server; not killed by GitHub runner job cleanup."""
    binary = str(Path(approved_install) / "ollama.exe")
    if not Path(binary).is_file():
        raise RecoveryRefused("native_ollama_binary_missing")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env.pop("OLLAMA_LLM_LIBRARY", None)  # do not inherit forced CPU library
    env.pop("RUNNER_TRACKING_ID", None)
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen(
            [binary, "serve"], env=env,
            cwd=approved_install,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=flags,
            close_fds=True,
        )
    except OSError as exc:
        raise RecoveryRefused("native_ollama_relaunch_failed") from exc


def recover_if_authorized(*, size_vram: int, free_mib: int,
                          consent: str, report: dict) -> bool:
    """May stop the SINGLE user-owned localhost Ollama listener once only."""
    if not recovery_eligible(
            size_vram=size_vram, free_mib=free_mib, consent=consent):
        raise RecoveryRefused("restart_preconditions_not_met")
    if os.name != "nt":
        raise RecoveryRefused("windows_only")
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise RecoveryRefused("local_install_not_found")
    approved_install = str(Path(local) / "Programs" / "Ollama")
    report["ollama_restart_log_signals_before"] = _read_ollama_log()
    report["ollama_restart_environment_before"] = environment_signals(os.environ)

    audit = _windows_inventory()
    # Persist safe scalar evidence even if validation declines the restart.
    report["ollama_restart_process_audit"] = {
        "native_process_count": len(audit.get("processes", [])),
        "local_listener_count": len(audit.get("listener_pids", [])),
        "busy_client_count": audit.get("established_clients"),
        "listener_loopback_only": bool(
            audit.get("listener_addresses")) and all(
                a in ("127.0.0.1", "::1")
                for a in audit.get("listener_addresses", [])),
        "clients_loopback_only": audit.get("all_clients_loopback") is True,
        "service_host_detected": any(
            p in audit.get("service_pids", [])
            for p in audit.get("listener_pids", [])),
    }
    owners = validate_native_listener(audit, approved_install,
                                      getpass.getuser())
    report["ollama_restart_attempted"] = True
    report["ollama_restart_stop"] = {"terminated_processes": 0,
                                      "forced_after_graceful_attempt": False}
    for pid in owners:
        status = _terminate_native_owner(pid, approved_install)
        report["ollama_restart_stop"]["terminated_processes"] += 1
        report["ollama_restart_stop"]["forced_after_graceful_attempt"] |= (
            status["forced_after_graceful_attempt"])
    # The local endpoint must actually shut down; otherwise never start 2nd.
    for _ in range(15):
        if not _check_tags():
            break
        time.sleep(0.5)
    else:
        raise RecoveryRefused("old_ollama_api_still_listening")
    _spawn_new_ollama(approved_install)
    for _ in range(30):
        if _check_tags():
            report["ollama_restart_api_ready"] = True
            return True
        time.sleep(0.5)
    raise RecoveryRefused("restarted_ollama_api_not_ready")
