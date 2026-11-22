# NERAL V4.1: pinned owner, validated staged files and automatic rollback.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
# A manual, explicitly approved benchmark may also stage the updated
# no-capture API implementation from main, preserving the pinned service and
# the original rollback guarantees.
$approvedFeature = $env:GITHUB_REF -eq 'refs/heads/codex/neral-v41-context-memory-tiers-20261121'
$approvedMerge = $env:GITHUB_EVENT_NAME -ceq 'push' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V41_BLIND200_MERGE_TITLE -ceq 'NERAL V4.1 owner-approved Blind200 final launch 20261121'
$approvedBlind200 = $env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:NERAL_V41_BLIND200_APPROVED -ceq 'NERAL-V41-BLIND200-APPROVED' -and
  $env:NERAL_V41_BLIND200_PREFLIGHT -ceq 'success' -and
  ($env:GITHUB_EVENT_NAME -ceq 'workflow_dispatch' -or $approvedMerge)
$approvedV42AB50 = $env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'push' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V42_AB50_APPROVED -ceq 'NERAL-V42-AB50-APPROVED' -and
  $env:NERAL_V42_AB50_PREFLIGHT -ceq 'success' -and
  $env:NERAL_V42_AB50_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved A/B50 Quartz Obsidian 20261122'
$approvedV42Smoke5 = $env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'push' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V42_SMOKE5_APPROVED -ceq 'NERAL-V42-SMOKE5-APPROVED' -and
  $env:NERAL_V42_SMOKE5_PREFLIGHT -ceq 'success' -and
  $env:NERAL_V42_SMOKE5_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved deadline smoke5 20261122'
$approvedV42Diag3 = $env:GITHUB_REF -ceq 'refs/heads/main' -and
  $env:GITHUB_EVENT_NAME -ceq 'push' -and
  $env:GITHUB_RUN_ATTEMPT -ceq '1' -and
  $env:NERAL_V42_DIAG3_APPROVED -ceq 'NERAL-V42-DIAG3-APPROVED' -and
  $env:NERAL_V42_DIAG3_PREFLIGHT -ceq 'success' -and
  ($env:NERAL_V42_DIAG3_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved telemetry 3x2 diagnosis 20261122' -or
   $env:NERAL_V42_DIAG3_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved telemetry 3x2 bottleneck recheck 20261122' -or
    $env:NERAL_V42_DIAG3_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved telemetry 3x2 fastpath math evidence 20261122' -or
    $env:NERAL_V42_DIAG3_MERGE_TITLE -ceq 'NERAL V4.2 owner-approved telemetry 3x2 quartz year final 20261122')
if(-not ($approvedFeature -or $approvedBlind200 -or $approvedV42AB50 -or $approvedV42Smoke5 -or $approvedV42Diag3)){
  throw 'V4.1 owner deployment requires pinned feature branch or explicitly approved main benchmark'
}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$src=$env:GITHUB_WORKSPACE
$python='D:/neral-qlora/venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $service -PathType Leaf) -or
   -not(Test-Path -LiteralPath $python -PathType Leaf)){
  throw 'Pinned V4 owner service or isolated Python unavailable'
}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if(@($task.Actions).Count -ne 1 -or
   [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
   throw 'Scheduled task identity does not match expected local NERAL'
}
$startup=[IO.File]::ReadAllText($service)
if($startup -notmatch 'NERAL_V3_AGENT_ENABLED' -or
   $startup -notmatch '(?i)-m\s+neral_preview_server' -or
   $startup -notmatch '--experimental-three-tier-hf'){
   throw 'Pinned three-tier service has unexpected startup configuration'
}
function ReadOwnedListener {
  $listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
  if($listeners.Count -eq 0){return $null}
  if($listeners.Count -ne 1 -or [string]$listeners[0].LocalAddress -ne '127.0.0.1'){
    throw 'Port 9690 has an unknown or multiple listeners'
  }
  $child=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
  $parent=$null
  if($child -and [int]$child.ParentProcessId -gt 0){
    $parent=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$child.ParentProcessId) -ErrorAction SilentlyContinue
  }
  if($null -eq $child -or $null -eq $parent){throw 'Unknown NERAL process ancestry'}
  foreach($p in @($child,$parent)){
    if([string]$p.Name -notmatch '(?i)^pythonw?\.exe$' -or
       [string]$p.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
       [string]$p.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
       [string]$p.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
      throw 'Port owner not pinned NERAL HF worker'
    }
  }
  return @{child=$child;parent=$parent}
}
function StopOwned {
  $task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  if(@($task.Actions).Count -ne 1 -or
     [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
    throw 'Scheduled task identity changed'
  }
  $prior=ReadOwnedListener
  if([string]$task.State -eq 'Running'){
    Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
    for($i=0;$i -lt 20;$i++){
      Start-Sleep -Milliseconds 500
      if([string](Get-ScheduledTask -TaskName $taskName).State -ne 'Running'){break}
    }
  }
  if([string](Get-ScheduledTask -TaskName $taskName).State -eq 'Running'){throw 'Task stop failed'}
  $live=ReadOwnedListener
  if($null -ne $live){
    if($null -eq $prior -or
       [int]$prior.child.ProcessId -ne [int]$live.child.ProcessId -or
       [int]$prior.parent.ProcessId -ne [int]$live.parent.ProcessId -or
       [string]$prior.child.CreationDate -ne [string]$live.child.CreationDate -or
       [string]$prior.parent.CreationDate -ne [string]$live.parent.CreationDate){
      throw 'Listener identity changed; preserve unknown process'
    }
    Stop-Process -Id ([int]$live.parent.ProcessId) -Force -ErrorAction Stop
    $childNow=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$live.child.ProcessId) -ErrorAction SilentlyContinue
    if($childNow){
      if([string]$childNow.CreationDate -ne [string]$live.child.CreationDate -or
         [string]$childNow.CommandLine -notmatch '(?i)--experimental-three-tier-hf'){
        if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -gt 0){
          throw 'Child identity changed; do not terminate an unknown process'
        }
      }else{
        Stop-Process -Id ([int]$live.child.ProcessId) -Force -ErrorAction Stop
      }
    }
  }
  for($i=0;$i -lt 12;$i++){
    if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){return}
    Start-Sleep -Milliseconds 500
  }
  throw 'Pinned service port remained occupied'
}
function HttpReady {
  $code=@(& curl.exe --noproxy '*' --silent --connect-timeout 3 --max-time 8 --output NUL --write-out '%{http_code}' 'http://127.0.0.1:9690/' 2>$null)
  return ($LASTEXITCODE -eq 0 -and ($code -join '').Trim() -eq '200')
}
function StartOwned {
  $task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  if([string]$task.State -ne 'Ready' -or $null -ne (ReadOwnedListener)){
    throw 'Pinned NERAL task not safely restartable'
  }
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
  for($i=0;$i -lt 150;$i++){
    Start-Sleep -Seconds 1
    if(HttpReady){return}
  }
  throw 'Pinned NERAL did not answer HTTP200 after restart'
}

# Never mutate original dataset, 97-example note store, frozen private holdout,
# model checkpoint or production LoRA weights.
$files=@('neral_v41_context.py','neral_verified_notes.py',
         'neral_evidence_agent.py','neral_obsidian_deep.py','neral_local_wiki.py',
         'neral_v42_deadline.py','neral_v42_telemetry.py','neral_v42_quality.py',
         'neral_three_tier_runtime.py','neral_preview_server.py',
         'templates/neral_preview.html',
         'static/neral-preview.js','static/neral-reference-ui.js')
$backup=Join-Path $env:LOCALAPPDATA ('neral-v41-backup-'+$env:GITHUB_RUN_ID)
if(($approvedV42Smoke5 -or $approvedV42Diag3) -and $env:NERAL_V42_SMOKE5_REVERT -ceq '1'){
  # Same-run rollback after the experimental 5+5 measurement, regardless
  # of pass/fail. Never promote unverified changes to the owner's service.
  if(-not(Test-Path -LiteralPath $backup -PathType Container)){
    throw 'Pinned same-run V4.2 owner snapshot absent; refuse blind rollback'
  }
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Pinned owner service identity unhealthy before rollback'
  }
  StopOwned
  foreach($file in $files){
    $original=Join-Path $backup $file
    $installed=Join-Path $root $file
    if(Test-Path -LiteralPath $original -PathType Leaf){
      Copy-Item -LiteralPath $original -Destination $installed -Force -ErrorAction Stop
      if((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne
         (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash){
        throw ('Experiment rollback checksum mismatch: '+$file)
      }
    }elseif($file -in @('neral_v41_context.py','neral_obsidian_deep.py','neral_local_wiki.py','neral_v42_deadline.py','neral_v42_telemetry.py','neral_v42_quality.py')){
      Remove-Item -LiteralPath $installed -ErrorAction SilentlyContinue
    }else{
      throw ('Experiment rollback source missing: '+$file)
    }
  }
  StartOwned
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Pinned original service unavailable after experimental rollback'
  }
  Write-Host 'NERAL_V42_SMOKE5_ORIGINAL_OWNER_RESTORED_HTTP200'
  return
}
if(Test-Path -LiteralPath $backup){throw 'Backup directory already exists'}
New-Item -ItemType Directory -Path $backup -Force | Out-Null
foreach($file in $files){
  $source=Join-Path $src $file
  $installed=Join-Path $root $file
  if(-not(Test-Path -LiteralPath $source -PathType Leaf)){
    throw ('Candidate V4.1 file missing: '+$file)
  }
  if(Test-Path -LiteralPath $installed -PathType Leaf){
    $original=Join-Path $backup $file
    New-Item -ItemType Directory -Path (Split-Path -Parent $original) -Force | Out-Null
    Copy-Item -LiteralPath $installed -Destination $original -ErrorAction Stop
  }elseif($file -notin @('neral_v41_context.py','neral_obsidian_deep.py','neral_local_wiki.py','neral_v42_deadline.py','neral_v42_telemetry.py','neral_v42_quality.py')){
    throw ('Expected installed V4 file missing: '+$file)
  }
}
$state=[string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State
if($state -eq 'Ready'){
  if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -ne 0){
    throw 'Unknown listener on port 9690 while owner task is Ready'
  }
  StartOwned
}elseif($state -ne 'Running'){
  throw 'Owner three-tier service cannot be safely managed'
}
if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
  throw 'Unrecognized or unhealthy service on port 9690'
}
$changed=$false
$completed=$false
try{
  StopOwned
  $changed=$true
  foreach($file in $files){
    $source=Join-Path $src $file
    $installed=Join-Path $root $file
    New-Item -ItemType Directory -Path (Split-Path -Parent $installed) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $installed -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash){
      throw ('V4 installed module hash mismatch: '+$file)
    }
  }
  $env:PYTHONPATH=$root
  $py=@('neral_v41_context.py','neral_verified_notes.py',
        'neral_evidence_agent.py','neral_obsidian_deep.py','neral_local_wiki.py','neral_v42_deadline.py','neral_v42_telemetry.py','neral_v42_quality.py',
        'neral_three_tier_runtime.py','neral_preview_server.py')
  & $python -m py_compile @($py|ForEach-Object { Join-Path $root $_ })
  if($LASTEXITCODE -ne 0){throw 'Installed V4.1 Python modules failed syntax check'}
  StartOwned
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Updated V4.1 service failed owned HTTP200 check'
  }
  Write-Host 'NERAL_V41_STAGED_ON_OWNER_HTTP200'
  if($approvedV42Smoke5 -or $approvedV42Diag3){
    # Probe service identity and lock only. The EXACT 5+5 test downstream is
    # the first inference. A cold-model timeout must be measured, not hidden
    # behind an unrelated 7-category deployment prerequisite.
    $env:PYTHONPATH=$src
    & $python -m scripts.neral_v42_owner_availability
    if($LASTEXITCODE -ne 0){
      throw 'NERAL local-session service status or GPU-busy verification failed'
    }
    Write-Host 'NERAL_V42_SMOKE5_STAGED_ONLY_NO_INFERENCE'
  }else{
    # Full original owner acceptance remains required for all other deployments.
    $env:PYTHONPATH=$src
    & $python -m scripts.neral_v41_owner_smoke
    if($LASTEXITCODE -ne 0){throw 'V4.1 7-category HTTP acceptance failed'}
  }
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'V4.1 owner service not HTTP-ready after smoke'
  }
  $completed=$true
  Write-Host 'NERAL_V41_ACCEPTED_AND_LIVE_HTTP200'
}finally{
  if($changed -and -not $completed){
    # The benchmark cannot be used to auto-promote a failed source pipeline.
    if(HttpReady){StopOwned}
    foreach($file in $files){
      $original=Join-Path $backup $file
      $installed=Join-Path $root $file
      if(Test-Path -LiteralPath $original -PathType Leaf){
        Copy-Item -LiteralPath $original -Destination $installed -Force -ErrorAction Stop
        if((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne
           (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash){
          throw ('V4.1 rollback checksum mismatch: '+$file)
        }
      }elseif($file -in @('neral_v41_context.py','neral_obsidian_deep.py','neral_local_wiki.py','neral_v42_deadline.py','neral_v42_telemetry.py','neral_v42_quality.py')){
        Remove-Item -LiteralPath $installed -ErrorAction SilentlyContinue
      }else{
        throw ('Missing original during V4.1 rollback: '+$file)
      }
    }
    if(-not(HttpReady)){
      if([string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State -ne 'Ready'){
        throw 'Owner service cannot be restarted safely after rollback'
      }
      StartOwned
    }
    if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
       throw 'Original NERAL V4 not HTTP-ready after V4.1 rollback'
    }
    Write-Host 'NERAL_V41_REJECTED_OR_FAILED_ORIGINAL_V4_HTTP200_RESTORED'
  }
}
if(-not $completed){throw 'V4.1 not accepted; original service retained'}
