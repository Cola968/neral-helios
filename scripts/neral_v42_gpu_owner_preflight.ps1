# Read-only owner identity and foreign CUDA/Python process preflight.
# Return JSON on stdout for the single approved diag3 Python process.
# Never stop processes, start models, log command lines or expose PIDs in artifacts.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction Stop)
if($listeners.Count -ne 1 -or [string]$listeners[0].LocalAddress -ne '127.0.0.1'){
  throw 'Missing or ambiguous pinned 9690 owner'
}
$child=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
if($null -eq $child -or [int]$child.ParentProcessId -le 0){
  throw 'Pinned 9690 owner ancestry unavailable'
}
$parent=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$child.ParentProcessId) -ErrorAction Stop
if($null -eq $parent){throw 'Pinned 9690 parent unavailable'}
foreach($item in @($child,$parent)){
  if([string]$item.Name -notmatch '(?i)^pythonw?\.exe$' -or
     [string]$item.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
     [string]$item.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
     [string]$item.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
    throw 'Unknown owner identity; refuse to attribute GPU VRAM'
  }
}
$owners=@([int]$child.ProcessId,[int]$parent.ProcessId) | Select-Object -Unique
$rows=@(& nvidia-smi.exe --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null)
if($LASTEXITCODE -ne 0){throw 'Per-process GPU query unavailable; no safe final measurement'}
$foreignPython=0
$foreignHeavy=0
$ownerVisible=0
foreach($row in $rows){
  $line=([string]$row).Trim()
  if(-not $line -or $line -match '(?i)No running processes found'){continue}
  if($line -notmatch '(?i)^(\d+)\s*,\s*(\d+|N/A|Not Supported)\s*(?:MiB)?\s*$'){
    throw 'Unparseable GPU compute process line; refuse to guess'
  }
  $pidNumber=[int]$Matches[1]
  $rawMemory=[string]$Matches[2]
  $memory=$null
  if($rawMemory -match '^\d+$'){$memory=[int]$rawMemory}
  if($pidNumber -in $owners){$ownerVisible++;continue}
  $other=Get-CimInstance Win32_Process -Filter ('ProcessId = '+$pidNumber) -ErrorAction SilentlyContinue
  if($null -eq $other){continue} # PID exited before inspection
  $otherName=[string]$other.Name
  if($otherName -match '(?i)^pythonw?\.exe$'){
    $foreignPython++
  }elseif($null -ne $memory -and $memory -ge 512 -and
          $otherName -notmatch '(?i)^(?:dwm|chrome|msedge|firefox)\.exe$'){
    $foreignHeavy++
  }
}
if($foreignPython -gt 0 -or $foreignHeavy -gt 0){
  throw 'Foreign Python or high-VRAM CUDA process detected; no process termination'
}
$total=@(& nvidia-smi.exe --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
if($LASTEXITCODE -ne 0 -or $total.Count -ne 1 -or
   [string]$total[0] -notmatch '^\s*(\d+)\s*$'){
  throw 'GPU total baseline unavailable'
}
$baseline=[int]$Matches[1]
# This private machine-local JSON passes PIDs to the diagnostic process only.
# The caller prints ONLY safe counts and scalar memory values.
@{
  owner_pids=@($owners)
  foreign_python_gpu_process_count=$foreignPython
  foreign_heavy_gpu_process_count=$foreignHeavy
  owner_gpu_pid_count=$ownerVisible
  system_gpu_preflight_mib=$baseline
} | ConvertTo-Json -Compress -Depth 4
