"""Read-only Windows GPU dedicated-memory attribution for NERAL diagnostics.

Windows NVIDIA process memory is often reported as N/A. This probe reads the
Windows GPU Process Memory performance counters and aggregates by category.
No PID, process name, paths, commands, prompts or private data are exported.
Counter totals are estimates, NOT authoritative total VRAM attribution.
"""
from __future__ import annotations

import json
import ntpath
import platform
import re
import subprocess


class ProbeError(ValueError):
    """Public-safe diagnostic error with no raw process data."""


_WINDOWS_CIM_COMMAND = r"""
$ErrorActionPreference='Stop'
try {
    $mem=@(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop | Select-Object Name,DedicatedUsage)
    $processes=@(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop | Select-Object ProcessId,Name)
    [pscustomobject]@{memory=$mem;processes=$processes} | ConvertTo-Json -Depth 4 -Compress
} catch {
    '{"error":"gpu_counters_unavailable"}'
}
"""

_PID = re.compile(r"^pid_([0-9]{1,10})_luid_", re.I)
_MIB = 1024 * 1024


def classify_process(name: str) -> str:
    if type(name) is not str:
        return "other"
    filename = ntpath.basename(name.strip()).casefold()
    if filename in {"python.exe", "pythonw.exe", "python"}:
        return "python"
    if filename in {"ollama.exe", "ollama_llama_server.exe", "ollama"}:
        return "ollama"
    return "other"


def parse_windows_gpu_snapshot(raw: str) -> dict:
    if type(raw) is not str or len(raw.encode("utf8")) > 1_000_000:
        raise ProbeError("Ungültige Größe der Windows-GPU-Diagnose.")
    try:
        snapshot = json.loads(raw)
    except (ValueError, TypeError, UnicodeError):
        raise ProbeError("Windows-GPU-Diagnose unlesbar.") from None
    if type(snapshot) is not dict:
        raise ProbeError("Windows-GPU-Diagnose ungültig.")
    if snapshot == {"error": "gpu_counters_unavailable"}:
        return {"status": "counters_unavailable", "read_only": True,
                "exposes_process_identity": False}
    if set(snapshot) != {"memory", "processes"}:
        raise ProbeError("Windows-GPU-Diagnose hat unerwartete Felder.")
    memory, processes = snapshot["memory"], snapshot["processes"]
    if (type(memory) is not list or type(processes) is not list or
            len(memory) > 4096 or len(processes) > 4096):
        raise ProbeError("Windows-GPU-Zählerdaten ungültig.")
    names: dict[int, str] = {}
    for row in processes:
        if type(row) is not dict:
            raise ProbeError("Windows-Prozesszähler ungültig.")
        pid, name = row.get("ProcessId"), row.get("Name")
        if (type(pid) is not int or pid < 0 or pid > 2**31 or
                type(name) is not str or len(name) > 260):
            raise ProbeError("Windows-Prozesszähler ungültig.")
        category = classify_process(name)
        if pid in names and names[pid] != category:
            raise ProbeError("Widersprüchlicher Windows-Prozesszähler.")
        names[pid] = category

    if not memory:
        return {"status": "no_gpu_process_counters", "read_only": True,
                "exposes_process_identity": False}

    # Some Windows counters expose the same PID through several GPU contexts.
    # The max per PID is a conservative approximation; never sum those rows.
    per_pid: dict[int, int] = {}
    for row in memory:
        if type(row) is not dict:
            raise ProbeError("Windows-GPU-Speicherzähler ungültig.")
        name, usage = row.get("Name"), row.get("DedicatedUsage")
        if (type(name) is not str or len(name) > 300 or
                type(usage) is not int or not 0 <= usage <= 150_000_000_000):
            raise ProbeError("Windows-GPU-Speicherzähler ungültig.")
        matched = _PID.match(name)
        if not matched:
            continue
        pid = int(matched.group(1))
        per_pid[pid] = max(per_pid.get(pid, 0), usage)
    if not per_pid:
        return {"status": "unrecognized_counter_names", "read_only": True,
                "exposes_process_identity": False}

    totals = {"python": 0, "ollama": 0, "other": 0}
    mapped = 0
    for pid, usage in per_pid.items():
        category = names.get(pid)
        if category is None:
            category = "other"
        else:
            mapped += 1
        totals[category] += usage
    return {
        "status": "measured_not_exact_vram_total",
        "read_only": True,
        "exposes_process_identity": False,
        "gpu_process_count": len(per_pid),
        "mapped_processes": mapped,
        "unmapped_processes": len(per_pid) - mapped,
        "python_dedicated_mib": round(totals["python"] / _MIB),
        "ollama_dedicated_mib": round(totals["ollama"] / _MIB),
        "other_dedicated_mib": round(totals["other"] / _MIB),
        "method": "windows_gpu_process_memory_counter_max_per_pid",
        "note": "Nicht mit nvidia-smi Gesamtbelegung gleichsetzen; Windows-Zähler können fehlen.",
    }


def run_probe(runner=subprocess.run) -> dict:
    if platform.system() != "Windows":
        raise ProbeError("Nur auf Windows ausführbar.")
    try:
        result = runner(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
             "-Command", _WINDOWS_CIM_COMMAND],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status":"counters_unavailable","read_only":True,
                "exposes_process_identity":False}
    if result.returncode != 0:
        return {"status":"counters_unavailable","read_only":True,
                "exposes_process_identity":False}
    return parse_windows_gpu_snapshot(result.stdout)


def main() -> None:
    # The only process output is aggregated, privacy-safe JSON.
    print(json.dumps(run_probe(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
