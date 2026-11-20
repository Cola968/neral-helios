# V3.5 one-shot paragraph comparison, 118-element lookup and source-math Docker retry test.
# V3.4 source-window anti-truncation and Helios proposal verification holdout.
# Approved independent RTX4070 benchmark: strict 90/100 gate with rollback.
# V3.3 regression: user-approved new 100 synthetic cases after V3.2 diagnosis.
# Owner-approved V3.2 candidate evaluation on the exact NERAL service, followed
# by unconditional restoration. No changes to main, model weights or task definition.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/codex/neral-v31-prospective100'){throw 'V3.2 branch required'}
if((& git log -1 --format=%s).Trim() -ne 'ops(v3.2): owner-approved independent 100 on RTX4070'){
  throw 'Explicit V3.2 benchmark authorization missing'
}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$src=$env:GITHUB_WORKSPACE
$python='D:/neral-qlora/venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $service -PathType Leaf) -or
   -not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'Pinned NERAL service missing'}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if(@($task.Actions).Count -ne 1 -or
   [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
  throw 'Approved service task identity mismatch'
}
$startup=[IO.File]::ReadAllText($service)
if($startup -notmatch 'NERAL_V3_AGENT_ENABLED' -or
   $startup -notmatch '(?i)-m\s+neral_preview_server' -or
   $startup -notmatch '--experimental-three-tier-hf'){
  throw 'Installed service is not pinned V3'
}
$paths=@('neral_evidence_agent.py','neral_v32_verifier.py','neral_page_reader.py','neral_knowledge_verifier.py')
$backup=Join-Path $env:RUNNER_TEMP ('neral-v32-rollback-'+$env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$had=@{}
foreach($name in $paths){
  if(-not(Test-Path -LiteralPath (Join-Path $src $name) -PathType Leaf)){
    throw ('Missing candidate module '+$name)
  }
  $original=Join-Path $root $name
  $had[$name]=Test-Path -LiteralPath $original -PathType Leaf
  if($had[$name]){
    Copy-Item -LiteralPath $original -Destination (Join-Path $backup $name) -ErrorAction Stop
  }
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
$mutated=$false
$completed=$false
try{
  if([string]$task.State -eq 'Ready'){StartOwned}
  elseif([string]$task.State -ne 'Running'){throw 'Unapproved task state'}
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Pinned V3 baseline listener unavailable'
  }
  StopOwned
  $mutated=$true
  foreach($name in $paths){
    $from=Join-Path $src $name
    $dest=Join-Path $root $name
    Copy-Item -LiteralPath $from -Destination $dest -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash){
      throw ('Candidate staging hash mismatch: '+$name)
    }
  }
  StartOwned
  $prefix=Join-Path $env:RUNNER_TEMP ('neral-v32-'+$env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
  New-Item -ItemType Directory -Path $prefix -Force | Out-Null
  $out=Join-Path $prefix 'summary.json'
  $sealed=Join-Path $prefix 'private-sealed-cases.json'
  $progress=Join-Path $prefix 'private-progress.jsonl'
  $details=Join-Path $prefix 'benchmark_details.jsonl'
  'V32_SUMMARY_PATH='+$out >> $env:GITHUB_ENV
  'V32_DETAILS_PATH='+$details >> $env:GITHUB_ENV
  Write-Host 'NERAL_V32_PINNED_CANDIDATE_STARTED_HTTP_200'
  Push-Location $src
  try{
    & $python -B -m scripts.neral_v31_prospective100 --authorize NERAL_V32_OWNER_APPROVED_NEW_100 --sealed $sealed --progress $progress --details $details --output $out
    if($LASTEXITCODE -ne 0){throw 'V3.2 100-case test did not complete'}
  }finally{Pop-Location}
  if(-not(Test-Path -LiteralPath $out -PathType Leaf)){throw 'Benchmark report missing'}
  $report=Get-Content -LiteralPath $out -Raw -Encoding UTF8 | ConvertFrom-Json
  if([int]$report.cases -ne 100 -or [int]$report.detail_count -ne 100){
    throw 'Incomplete 100-case evidence'
  }
  if([int]$report.strict_e2e_pass -ge 90){
    Write-Host ('NERAL_V32_ELIGIBLE_FOR_REVIEW_AND_MERGE score='+$report.strict_e2e_pass)
  }else{
    Write-Host ('NERAL_V32_PROMOTION_BLOCKED score='+$report.strict_e2e_pass)
  }
  $completed=$true
}finally{
  if($mutated){
    try{
      StopOwned
      foreach($name in $paths){
        $dest=Join-Path $root $name
        if($had[$name]){
          $original=Join-Path $backup $name
          Copy-Item -LiteralPath $original -Destination $dest -Force -ErrorAction Stop
          if((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne
             (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash){
            throw ('Rollback hash mismatch '+$name)
          }
        }elseif(Test-Path -LiteralPath $dest -PathType Leaf){
          Remove-Item -LiteralPath $dest -Force -ErrorAction Stop
        }
      }
      StartOwned
      Write-Host 'NERAL_V32_ORIGINAL_V3_RESTORED_HTTP_200'
    }catch{
      Write-Host ('NERAL_V32_ROLLBACK_NEEDS_REVIEW '+$_.Exception.GetType().Name)
      throw
    }
  }
}
if(-not $completed){throw 'V3.2 100-case test incomplete'}
