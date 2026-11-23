# One-shot raw-9B benchmark with same-run original-owner restore.
# No unknown GPU or Windows process is terminated; no original service files
# or Blind200 paths are modified. Called ONLY from the approved W39 workflow.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$approved=($env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'push' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V43_REF50_APPROVED -ceq 'NERAL-V43-RAW9B-NERAL10-REF50' -and
  $env:NERAL_V43_REF50_PREFLIGHT -ceq 'success' -and
  $env:NERAL_V43_REF50_MERGE_TITLE -ceq 'NERAL V4.3 owner-approved raw9b vs NERAL10 reference50 20261123')
$recoveryApproved=($env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'workflow_dispatch' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V43_REF50_RECOVERY_APPROVED -ceq 'V43-RAW50-ONLY-45264946953' -and
  $env:NERAL_V43_REF50_PARENT_RUN -ceq '45264946953')
if(-not ($approved -or $recoveryApproved)){throw 'V4.3 raw baseline requires exact single-use owner approval'}
if(-not $env:NERAL_V43_REF50_PRIVATE){throw 'Owner-local private reference50 directory absent'}
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$python='D:/neral-qlora/venv/Scripts/python.exe'
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
if(-not (Test-Path -LiteralPath $python -PathType Leaf)){throw 'Pinned offline Python not found'}
if($recoveryApproved){
  $expectedPrivate=Join-Path $env:LOCALAPPDATA 'NERAL-private/v43-reference50/45264946953'
  if([IO.Path]::GetFullPath($env:NERAL_V43_REF50_PRIVATE) -cne
     [IO.Path]::GetFullPath($expectedPrivate)){
    throw 'Recovery private path differs from original sealed run'
  }
  & $python -m scripts.neral_v43_reference50_live --phase recovery-preflight --private-dir $expectedPrivate
  if($LASTEXITCODE -ne 0){throw 'Original NERAL 50 did not pass recovery preflight'}
}

$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if(@($task.Actions).Count -ne 1 -or [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
  throw 'Scheduled task identity mismatch'
}
function Get-Owner{
  $listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction Stop)
  if($listeners.Count -ne 1 -or [string]$listeners[0].LocalAddress -ne '127.0.0.1'){
    throw 'Pinned owner listener missing or ambiguous'
  }
  $child=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
  if($null -eq $child -or [int]$child.ParentProcessId -le 0){throw 'Owner process ancestry missing'}
  $parent=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$child.ParentProcessId) -ErrorAction Stop
  foreach($p in @($parent,$child)){
    if($null -eq $p -or
       [string]$p.Name -notmatch '(?i)^pythonw?\.exe$' -or
       [string]$p.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
       [string]$p.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
       [string]$p.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
      throw 'Unknown process identity at NERAL listener'
    }
  }
  return @{parent=$parent;child=$child}
}
function HttpReady {
  $result=@(& curl.exe --noproxy '*' --silent --connect-timeout 3 --max-time 8 --output NUL --write-out '%{http_code}' 'http://127.0.0.1:9690/' 2>$null)
  return ($LASTEXITCODE -eq 0 -and ($result -join '').Trim() -eq '200')
}
function Stop-ConfirmedOwner($owner) {
  $running=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  if([string]$running.State -ne 'Running'){throw 'Pinned owner task is not running'}
  # Record only owned descendants for safe same-process cleanup.
  $known=New-Object 'System.Collections.Generic.HashSet[int]'
  [void]$known.Add([int]$owner.parent.ProcessId)
  [void]$known.Add([int]$owner.child.ProcessId)
  $seen=@(Get-CimInstance Win32_Process -ErrorAction Stop)
  $descendants=New-Object System.Collections.ArrayList
  $growing=$true
  while($growing){
    $growing=$false
    foreach($item in $seen){
      if($known.Contains([int]$item.ProcessId)){continue}
      if($known.Contains([int]$item.ParentProcessId)){
        [void]$known.Add([int]$item.ProcessId)
        $growing=$true
        if([string]$item.Name -match '(?i)^pythonw?\.exe$'){
          [void]$descendants.Add(@{pid=[int]$item.ProcessId;created=[string]$item.CreationDate})
        }
      }
    }
  }
  Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
  for($i=0;$i -lt 30;$i++){
    if([string](Get-ScheduledTask -TaskName $taskName).State -ne 'Running'){break}
    Start-Sleep -Milliseconds 500
  }
  if([string](Get-ScheduledTask -TaskName $taskName).State -eq 'Running'){
    throw 'Scheduled owner did not stop'
  }
  foreach($p in @($owner.parent,$owner.child)){
    $live=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$p.ProcessId) -ErrorAction SilentlyContinue
    if($null -ne $live){
      if([string]$live.CreationDate -ne [string]$p.CreationDate -or
         [string]$live.CommandLine -notmatch '(?i)--experimental-three-tier-hf'){
        throw 'Owned Python identity changed; do not terminate'
      }
      Stop-Process -Id ([int]$p.ProcessId) -Force -ErrorAction Stop
    }
  }
  foreach($entry in $descendants){
    $live=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$entry.pid) -ErrorAction SilentlyContinue
    if($null -ne $live -and [string]$live.CreationDate -eq [string]$entry.created -and
       [string]$live.Name -match '(?i)^pythonw?\.exe$'){
      Stop-Process -Id ([int]$entry.pid) -Force -ErrorAction SilentlyContinue
    }
  }
  for($i=0;$i -lt 20;$i++){
    if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){return}
    Start-Sleep -Milliseconds 500
  }
  throw 'Pinned service port did not become free'
}
function Check-RawCapacityAndForeign {
  $gpu=@(& nvidia-smi.exe --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits 2>$null)
  if($LASTEXITCODE -ne 0 -or $gpu.Count -ne 1 -or [string]$gpu[0] -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$'){
    throw 'Raw GPU capacity sample unavailable'
  }
  $free=[int]$Matches[1];$used=[int]$Matches[2];$total=[int]$Matches[3]
  if($total -lt 10000 -or $free+$used -gt $total+256 -or $free -lt 8500){
    throw 'Raw model preload GPU capacity refused'
  }
  $rows=@(& nvidia-smi.exe --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null)
  if($LASTEXITCODE -ne 0){throw 'Raw foreign GPU process enumeration unavailable'}
  foreach($lineRaw in $rows){
    $line=([string]$lineRaw).Trim()
    if(-not $line -or $line -match '(?i)No running processes found'){continue}
    if($line -notmatch '(?i)^(\d+)\s*,\s*(\d+|\[?N/A\]?|\[?Not Supported\]?)\s*(?:MiB)?\s*$'){
      throw 'Raw GPU process output unparseable'
    }
    $otherPid=[int]$Matches[1]
    $process=Get-CimInstance Win32_Process -Filter ('ProcessId = '+$otherPid) -ErrorAction SilentlyContinue
    if($null -eq $process){continue}
    if([string]$process.Name -match '(?i)^(python|pythonw|ollama|llama-server|lmstudio|vllm|tritonserver)\.exe$'){
      throw 'Raw model foreign Python or LLM compute process active'
    }
  }
  Write-Host ('NERAL_V43_RAW_PRELOAD_OK used_mib='+$used+' free_mib='+$free)
}
function Restore-OriginalOwner {
  if([string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State -ne 'Ready' -or
     @(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -ne 0){
    throw 'Unexpected owner state before original restore'
  }
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
  for($i=0;$i -lt 150;$i++){
    Start-Sleep -Seconds 1
    if(HttpReady){
      $null=Get-Owner
      Write-Host 'NERAL_V43_REF50_ORIGINAL_OWNER_RESTORED_HTTP200'
      return
    }
  }
  throw 'Original owner failed HTTP200 restore after raw baseline'
}

if(-not(HttpReady)){throw 'Initial original owner not HTTP200'}
$owner=Get-Owner
$stopped=$false
try{
  $stopped=$true  # If StopOwned fails, attempt original-owner recovery.
  Stop-ConfirmedOwner $owner
  Check-RawCapacityAndForeign
  & $python -m scripts.neral_v43_reference50_live --phase raw --private-dir $env:NERAL_V43_REF50_PRIVATE
  if($LASTEXITCODE -ne 0){throw 'Raw 9B exactly 50 tasks failed; no rerun'}
}finally{
  if($stopped){
    Restore-OriginalOwner
  }
}
