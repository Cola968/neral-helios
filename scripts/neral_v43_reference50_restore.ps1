# Idempotent independent post-raw restore in case raw GPU job was interrupted.
# No process kills and no filesystem modifications. Fail closed on unknown
# listener, task, model or GPU compute owners.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$allowed=($env:GITHUB_REF -ceq 'refs/heads/main' -and
          $env:GITHUB_EVENT_NAME -ceq 'push' -and
          $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
          $env:NERAL_V43_REF50_APPROVED -ceq 'NERAL-V43-RAW9B-NERAL10-REF50' -and
          $env:NERAL_V43_REF50_MERGE_TITLE -ceq 'NERAL V4.3 owner-approved raw9b vs NERAL10 reference50 20261123')
$recoveryApproved=($env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'workflow_dispatch' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V43_REF50_RECOVERY_APPROVED -ceq 'V43-RAW50-ONLY-45264946953' -and
  $env:NERAL_V43_REF50_PARENT_RUN -ceq '45264946953')
if(-not ($allowed -or $recoveryApproved)){throw 'Unauthorized V4.3 independent owner rollback'}
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
function RequireTask {
  $t=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  if(@($t.Actions).Count -ne 1 -or [string]$t.Actions[0].Arguments -notlike ('*'+$service+'*')){
    throw 'Original NERAL scheduled task identity mismatch'
  }
  return $t
}
function Healthy {
  $code=@(& curl.exe --noproxy '*' --silent --max-time 8 --output NUL --write-out '%{http_code}' 'http://127.0.0.1:9690/' 2>$null)
  return ($LASTEXITCODE -eq 0 -and ($code -join '').Trim() -eq '200')
}
function ConfirmOwner {
  $listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction Stop)
  if($listeners.Count -ne 1 -or [string]$listeners[0].LocalAddress -ne '127.0.0.1'){
    throw 'Original owner listener missing or ambiguous'
  }
  $child=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
  if($null -eq $child -or [int]$child.ParentProcessId -le 0){
    throw 'Original owner child missing'
  }
  $parent=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$child.ParentProcessId) -ErrorAction Stop
  foreach($p in @($child,$parent)){
    if($null -eq $p -or
       [string]$p.Name -notmatch '(?i)^pythonw?\.exe$' -or
       [string]$p.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
       [string]$p.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
       [string]$p.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
      throw 'Unrecognized original owner process'
    }
  }
}
function NoForeignCompute {
  $rows=@(& nvidia-smi.exe --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null)
  if($LASTEXITCODE -ne 0){throw 'GPU process query unavailable before restore'}
  foreach($raw in $rows){
    $line=([string]$raw).Trim()
    if(-not $line -or $line -match '(?i)No running processes found'){continue}
    if($line -notmatch '(?i)^(\d+)\s*,\s*(\d+|\[?N/A\]?|\[?Not Supported\]?)\s*(?:MiB)?\s*$'){
      throw 'GPU process report unparseable before restore'
    }
    $pidNumber=[int]$Matches[1]
    $process=Get-CimInstance Win32_Process -Filter ('ProcessId = '+$pidNumber) -ErrorAction SilentlyContinue
    if($null -ne $process -and
       [string]$process.Name -match '(?i)^(python|pythonw|ollama|llama-server|lmstudio|vllm|tritonserver)\.exe$'){
      throw 'Competing compute process survived raw run; manual recovery needed'
    }
  }
}
$task=RequireTask
if(Healthy){
  ConfirmOwner
  Write-Host 'NERAL_V43_REF50_ORIGINAL_OWNER_VERIFIED_HTTP200'
  exit 0
}
if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -ne 0){
  throw 'Another process occupies NERAL port; refuse unverified restore'
}
if([string]$task.State -ne 'Ready'){
  throw 'Original task is not ready for safe restart'
}
NoForeignCompute
Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
for($attempt=0;$attempt -lt 150;$attempt++){
  Start-Sleep -Seconds 1
  if(Healthy){
    ConfirmOwner
    Write-Host 'NERAL_V43_REF50_ORIGINAL_OWNER_RESTORED_HTTP200'
    exit 0
  }
}
throw 'Original NERAL owner could not be restored'
