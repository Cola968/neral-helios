# Read-only Windows GPU owner preflight. No PID, path or command-line in logs.
param([switch]$CpuTest)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

function Deny([string]$code){
  throw ('NERAL_GPU_PREFLIGHT_REASON:'+$code)
}
function Inspect-GpuApps {
  param([string[]]$Rows,[int[]]$Owners,[scriptblock]$ResolveName,[int]$FreeMiB)
  $ownerMap=@{}
  foreach($p in $Owners){$ownerMap[[int]$p]=$true}
  $seen=@{}
  $gui=0; $unknown=0; $wddmNA=0; $ownerVisible=0
  $foreignPython=0; $foreignLLM=0
  foreach($item in $Rows){
    $line=([string]$item).Trim()
    if(-not $line -or $line -match '(?i)^No running processes found'){continue}
    # WDDM uses bracketed [N/A] even for harmless window-manager processes.
    if($line -notmatch '(?i)^(\d+)\s*,\s*(\d+|\[?N/A\]?|\[?Not Supported\]?)\s*(?:MiB)?\s*$'){
      Deny 'compute_rows_unparseable'
    }
    $gpuPid=[int]$Matches[1]; $memory=[string]$Matches[2]
    if($gpuPid -le 0){Deny 'compute_pid_invalid'}
    if($seen.ContainsKey($gpuPid)){Deny 'compute_pid_duplicate'}
    $seen[$gpuPid]=$true
    if($memory -notmatch '^\d+$'){$wddmNA++}
    if($ownerMap.ContainsKey($gpuPid)){$ownerVisible++;continue}
    $name=[string](& $ResolveName $gpuPid)
    if(-not $name){$unknown++;continue} # The PID exited during inspection
    if($name -match '(?i)^(dwm|csrss|winlogon|explorer|ShellExperienceHost|SearchHost|SearchApp|sihost|RuntimeBroker|ApplicationFrameHost|StartMenuExperienceHost|TextInputHost|chrome|msedge|firefox|brave|opera|vivaldi)\.exe$'){
      $gui++;continue
    }
    if($name -match '(?i)^pythonw?\.exe$'){
      $foreignPython++;continue
    }
    if($name -match '(?i)^(ollama|llama-server|llama|koboldcpp|llamafile|lmstudio|text-generation-webui|text-generation-launcher|vllm|tritonserver)\.exe$'){
      $foreignLLM++;continue
    }
    $unknown++
  }
  if($foreignPython -gt 0){Deny 'foreign_python_compute_active'}
  if($foreignLLM -gt 0){Deny 'foreign_llm_compute_active'}
  if($FreeMiB -lt 8500){Deny 'driver_free_memory_below_8500_mib'}
  return @{
    foreign_python_gpu_process_count=0
    foreign_heavy_gpu_process_count=0
    foreign_llm_compute_process_count=0
    owner_gpu_pid_count=$ownerVisible
    gui_gpu_process_count=$gui
    unknown_gpu_process_count=$unknown
    wddm_na_memory_rows=$wddmNA
  }
}

if($CpuTest){
  # CPU-only synthetic tests: no driver query or process inspection.
  $names=@{10='python.exe';20='dwm.exe';21='chrome.exe';22='csrss.exe';30='python.exe';40='ollama.exe'}
  $resolve={param($p) $names[[int]$p]}
  $good=Inspect-GpuApps -Rows @('10, [N/A]','20, [N/A]','21, N/A','22, 142 MiB') -Owners @(10) -ResolveName $resolve -FreeMiB 9000
  if($good.gui_gpu_process_count -ne 3 -or $good.wddm_na_memory_rows -ne 3){
    throw 'CPU WDDM GUI/N/A test mismatch'
  }
  Write-Output 'NERAL_WDDM_CPU_GUI_NA_ALLOWED'
  $blocked=$false
  try {
    $null=Inspect-GpuApps -Rows @('10, [N/A]','30, [N/A]') -Owners @(10) -ResolveName $resolve -FreeMiB 9000
  } catch {
    $blocked=$_.Exception.Message -like '*foreign_python_compute_active*'
  }
  if(-not $blocked){throw 'CPU foreign Python block not enforced'}
  Write-Output 'NERAL_WDDM_CPU_FOREIGN_PYTHON_BLOCKED'
  $blocked=$false
  try {
    $null=Inspect-GpuApps -Rows @('10, [N/A]','40, [N/A]') -Owners @(10) -ResolveName $resolve -FreeMiB 9000
  } catch {
    $blocked=$_.Exception.Message -like '*foreign_llm_compute_active*'
  }
  if(-not $blocked){throw 'CPU foreign LLM block not enforced'}
  $blocked=$false
  try {
    $null=Inspect-GpuApps -Rows @('10, [N/A]','20, [N/A]') -Owners @(10) -ResolveName $resolve -FreeMiB 8499
  } catch {
    $blocked=$_.Exception.Message -like '*driver_free_memory_below_8500_mib*'
  }
  if(-not $blocked){throw 'CPU 8500-MiB driver-free gate not enforced'}
  Write-Output 'NERAL_WDDM_CPU_LOW_FREE_BLOCKED'
  exit 0
}

try {
  $listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction Stop)
  if($listeners.Count -ne 1 -or [string]$listeners[0].LocalAddress -ne '127.0.0.1'){
    Deny 'owner_listener_not_unique'
  }
  $child=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
  if($null -eq $child -or [int]$child.ParentProcessId -le 0){
    Deny 'owner_parent_unavailable'
  }
  $parent=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$child.ParentProcessId) -ErrorAction Stop
  if($null -eq $parent){Deny 'owner_parent_unavailable'}
  foreach($item in @($child,$parent)){
    if([string]$item.Name -notmatch '(?i)^pythonw?\.exe$' -or
       [string]$item.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
       [string]$item.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
       [string]$item.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
      Deny 'owner_identity_mismatch'
    }
  }
  $owners=@([int]$child.ProcessId,[int]$parent.ProcessId) | Select-Object -Unique
  $gpu=@(& nvidia-smi.exe --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits 2>$null)
  if($LASTEXITCODE -ne 0){Deny 'nvidia_smi_gpu_query_failed'}
  if($gpu.Count -ne 1 -or [string]$gpu[0] -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$'){
    Deny 'nvidia_smi_gpu_memory_unparseable'
  }
  $free=[int]$Matches[1];$used=[int]$Matches[2];$total=[int]$Matches[3]
  if($total -lt 10000 -or $free -lt 0 -or $free+$used -gt $total+256){
    Deny 'nvidia_smi_gpu_memory_inconsistent'
  }
  $rows=@(& nvidia-smi.exe --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null)
  if($LASTEXITCODE -ne 0){Deny 'nvidia_smi_compute_query_failed'}
  $resolver={
    param($pidValue)
    $process=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$pidValue) -ErrorAction SilentlyContinue
    if($null -eq $process){return ''}
    return [string]$process.Name
  }
  $report=Inspect-GpuApps -Rows $rows -Owners $owners -ResolveName $resolver -FreeMiB $free
  $report.owner_pids=@($owners)
  $report.system_gpu_preflight_mib=$used
  $report.system_gpu_free_preflight_mib=$free
  $report.system_gpu_total_preflight_mib=$total
  $report | ConvertTo-Json -Compress -Depth 4
} catch {
  $message=[string]$_.Exception.Message
  $reason='unexpected_owner_query_error'
  if($message -match 'NERAL_GPU_PREFLIGHT_REASON:([a-z0-9_]+)'){
    $reason=$Matches[1]
  }
  # Only an enum and safe hardware scalar, never PIDs, local paths,
  # process names or command lines. The unavailable sentinel is -1.
  $freeValue=-1
  $freeRecord=Get-Variable -Name free -Scope 0 -ErrorAction SilentlyContinue
  if($null -ne $freeRecord -and $freeRecord.Value -is [int]){
    $freeValue=[int]$freeRecord.Value
  }
  [Console]::Error.WriteLine('NERAL_GPU_PREFLIGHT_REASON:'+$reason+' free_mib='+$freeValue)
  exit 21
}
