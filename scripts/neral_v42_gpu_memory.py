"""Read-only NVIDIA process/system VRAM accounting for the approved 3+3 probe.

No model/device initialization, no process enumeration in artifacts and no
process termination. Unknown WDDM per-process memory is None, never zero.
"""
from __future__ import annotations

import re
import subprocess

_CAP = 9500
_USAGE = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
_APPS = ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"]


def parse_compute_rows(raw: str) -> dict[int, int | None]:
    """Strict PID -> VRAM map; WDDM/N/A memory stays unknown."""
    if not isinstance(raw, str):
        raise ValueError("Invalid compute-app output")
    results = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        match = re.fullmatch(
            r"(?i)(\d+)\s*,\s*(\d+|N/A|Not Supported)\s*(?:MiB)?\s*", line)
        if not match:
            raise ValueError("Unparseable compute-app memory sample")
        pid = int(match[1])
        if pid <= 0 or pid in results:
            raise ValueError("Invalid or duplicated compute PID")
        results[pid] = int(match[2]) if match[2].isdigit() else None
    return results


def read_vram_snapshot(owner_pids, *, runner=subprocess.run):
    """Two independent nvidia-smi measurements; no global-used attribution."""
    if not isinstance(owner_pids, (set, tuple, list)) or not owner_pids or any(
            type(pid) is not int or pid <= 0 for pid in owner_pids):
        raise ValueError("Owner PID identity is mandatory")
    def request(command):
        try:
            response = runner(command, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True,
                              check=False, timeout=4)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return response.stdout if response.returncode == 0 else None

    total_raw = request(_USAGE)
    compute_raw = request(_APPS)
    total = None
    if isinstance(total_raw, str):
        lines = [line.strip() for line in total_raw.splitlines() if line.strip()]
        if len(lines) == 1:
            m = re.fullmatch(r"(?i)(\d+)\s*(?:MiB)?", lines[0])
            if m:
                total = int(m[1])
    compute = None
    if compute_raw is not None:
        try:
            compute = parse_compute_rows(compute_raw)
        except ValueError:
            compute = None
    owners = set(owner_pids)
    # A missing owner CUDA process is not a measured 0 MiB sample. If WDDM
    # suppresses per-process VRAM, keep the result explicitly unknown.
    observed = set(compute or {}) & owners
    mem = (sum(compute[pid] for pid in observed)
           if compute is not None and observed
           and all(compute[pid] is not None for pid in observed)
           else None)
    return {
        "system_total_mib":total,
        "neral_process_mib":mem,
        "neral_pid_count":len(owners),
        "neral_compute_pid_count":len(observed),
        "other_compute_process_count": (
            len(set(compute)-owners) if compute is not None else None),
        "process_memory_status":(
            "measured" if mem is not None else
            "wddm_memory_unreported" if compute is not None and observed
            else "owner_process_not_reported" if compute is not None
            else "nvidia_compute_query_unavailable"),
    }


def validate_owner_snapshot(snapshot, *, process_cap_mib=_CAP):
    """Only owner VRAM proves the 9,500 MiB gate; total VRAM is descriptive."""
    owner = snapshot.get("neral_process_mib") if isinstance(snapshot, dict) else None
    return (type(owner) is int and owner >= 0
            and type(process_cap_mib) is int and process_cap_mib > 0
            and owner < process_cap_mib)
