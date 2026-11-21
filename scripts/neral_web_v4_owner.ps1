# NERAL Web V4: exact pinned owner service, backup, no model replacement.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/codex/neral-web-v4-20261121'){
  throw 'Web V4 owner deployment requires approved isolated branch'
}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$src=$env:GITHUB_WORKSPACE
$python='D:/neral-qlora/venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $service -PathType Leaf) -or
   -not(Test-Path -LiteralPath $python -PathType Leaf)){
  throw 'Pinned V3.10 owner service or isolated Python unavailable'
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
$files=@('neral_page_reader.py','neral_evidence_agent.py',
         'neral_v32_verifier.py','neral_three_tier_runtime.py')
$backup=Join-Path $env:LOCALAPPDATA ('neral-web-v4-backup-'+$env:GITHUB_RUN_ID)
if(Test-Path -LiteralPath $backup){throw 'Backup directory already exists'}
New-Item -ItemType Directory -Path $backup -Force | Out-Null
foreach($file in $files){
  $source=Join-Path $src $file
  $installed=Join-Path $root $file
  if(-not(Test-Path -LiteralPath $source -PathType Leaf) -or
     -not(Test-Path -LiteralPath $installed -PathType Leaf)){
    throw ('Expected candidate or installed runtime module missing: '+$file)
  }
  Copy-Item -LiteralPath $installed -Destination (Join-Path $backup $file) -ErrorAction Stop
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
    Copy-Item -LiteralPath $source -Destination $installed -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash){
      throw ('V4 installed module hash mismatch: '+$file)
    }
  }
  $env:PYTHONPATH=$root
  & $python -m py_compile @($files|ForEach-Object { Join-Path $root $_ })
  if($LASTEXITCODE -ne 0){throw 'Installed Web V4 Python modules failed syntax check'}
  StartOwned
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Updated Web V4 service failed owned HTTP200 check'
  }
  Write-Host 'NERAL_WEB_V4_STAGED_ON_OWNER_HTTP200'
  # Every question uses ordinary consented search: same 6/min provider quota as
  # production. Cool-down is EXCLUDED from per-answer network/model latency.
  $env:PYTHONPATH=$src
  $reportDir=Join-Path 'D:/neral-web-v4-qa' ($env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
  if(Test-Path -LiteralPath $reportDir){throw 'Refusing to overwrite previous V4 run'}
  New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
  $bench=Join-Path $reportDir 'new-20-web-results.jsonl'
  & $python -m scripts.neral_web_v4_benchmark --out $bench --spacing-seconds 24
  $benchExit=$LASTEXITCODE
  $summary=[IO.Path]::ChangeExtension($bench,'.summary.json')
  if(Test-Path -LiteralPath $summary -PathType Leaf){
    $x=Get-Content -LiteralPath $summary -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ('NERAL_WEB_V4_TEST_PASSED '+[int]$x.passed+'/20')
    Write-Host ('NERAL_WEB_V4_MEAN_SECONDS '+[double]$x.mean_seconds)
    Write-Host ('NERAL_WEB_V4_MAX_SECONDS '+[double]$x.max_seconds)
    Write-Host ('NERAL_WEB_V4_ACCEPTANCE '+[bool]$x.acceptance)
  }
  if($benchExit -ne 0){throw 'Web V4 acceptance benchmark did not meet 19/20 and mean <6s'}
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Web V4 owner service not responding after 20-case benchmark'
  }
  $completed=$true
  Write-Host 'NERAL_WEB_V4_ACCEPTED_AND_LIVE_HTTP200'
}finally{
  if($changed -and -not $completed){
    # The benchmark cannot be used to auto-promote a failed source pipeline.
    if(HttpReady){StopOwned}
    foreach($file in $files){
      $original=Join-Path $backup $file
      $installed=Join-Path $root $file
      Copy-Item -LiteralPath $original -Destination $installed -Force -ErrorAction Stop
      if((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne
         (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash){
        throw ('V4 rollback checksum mismatch: '+$file)
      }
    }
    if(-not(HttpReady)){
      if([string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State -ne 'Ready'){
        throw 'Owner service cannot be restarted safely after rollback'
      }
      StartOwned
    }
    if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
       throw 'Original NERAL V3.10 not HTTP-ready after Web V4 rollback'
    }
    Write-Host 'NERAL_WEB_V4_REJECTED_OR_FAILED_ORIGINAL_V310_HTTP200_RESTORED'
  }
}
if(-not $completed){throw 'Web V4 not accepted; original service retained'}
