# Explicitly user-authorized staged V1 local 9690 acceptance: 2026-11-18
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=$env:GITHUB_WORKSPACE
$py='D:/neral-qlora/venv/Scripts/python.exe'
$oldName='NERAL-PR558-ThreeTier-909d1419c9ff'
$oldRoot=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$oldServer=Join-Path $oldRoot 'Start-Neral-Three-Tier-Service.ps1'
$short='77e124c98e61'
$newName='NERAL-PR561-V1-'+$short
$newParent=Join-Path $env:LOCALAPPDATA ('NERAL/pr561-v1-'+$short)
$newRoot=Join-Path $newParent 'NERAL-1.0-Local'
$newServer=Join-Path $newRoot 'Start-Neral-Three-Tier-Service.ps1'
$approved=[ordered]@{
  'neral_web_search.py'='08ae0fc8264fc9ace68f7e9a9a14a50a5b9c652c'
  'neral_web_api.py'='a4d621b318666d8521a758e7699ee19f408756ee'
  'neral_everyday_verified.py'='25bb4af5535035ebb4498d1b70d03fc78f5f3d06'
  'neral_three_tier_runtime.py'='5796223a47c1cec0362548715a775d7731bdaec7'
  'static/neral-preview.js'='42dcc9338bf5701ed3d982848c3db5ebe240b0b5'
  'static/neral-web-live.js'='a0e30ebf2dbca66a25556a07740b5ac93bb4c472'
}
$approvedPrevious=[ordered]@{
  'neral_web_search.py'='b8b9fd6ca3acc1286d69a5b758914631d009d16f'
  'neral_web_api.py'='3d9e8d779e514c31a1f243699d77d3a2565c18f6'
  'neral_everyday_verified.py'='6239a274a5049a32dde7408e1961244d2702dfbc'
  'neral_three_tier_runtime.py'='f3e5e2376999d821dd2a1aefc54be59a59ab0473'
  'static/neral-preview.js'='ce535f84c2e779d6619a968ca1e9b2582d20edbf'
  'static/neral-web-live.js'='d17ae1b0c7f45f46206181200e27f2a075a37da9'
}

function HashFor([string]$rel,[string]$root) {
  $file=Join-Path $root $rel
  if(-not(Test-Path -LiteralPath $file -PathType Leaf)){throw "Approved source missing: $rel"}
  $v=@(& git -c core.autocrlf=true hash-object --path=$rel $file)
  if($LASTEXITCODE -ne 0 -or @($v).Count -ne 1){throw "Cannot hash source: $rel"}
  return [string]$v[0].Trim()
}

function ReadOwnedListener {
  $listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
  if($listeners.Count -eq 0){return $null}
  if($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1'){
    throw 'Unknown or multiple listeners on port 9690'
  }
  $pidValue=[int]$listeners[0].OwningProcess
  $child=Get-CimInstance Win32_Process -Filter ("ProcessId = "+$pidValue) -ErrorAction Stop
  $parent=$null
  if($child -and [int]$child.ParentProcessId -gt 0){
    $parent=Get-CimInstance Win32_Process -Filter ("ProcessId = "+[int]$child.ParentProcessId) -ErrorAction SilentlyContinue
  }
  if($null -eq $child -or $null -eq $parent){throw 'Unknown NERAL process ancestry'}
  foreach($p in @($child,$parent)){
    if([string]$p.Name -notmatch '(?i)^pythonw?\.exe$' -or
       [string]$p.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
       [string]$p.CommandLine -notmatch '(?i)--experimental-three-tier-hf' -or
       [string]$p.CommandLine -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){
      throw 'Port owner is not expected NERAL HF worker; refuse to stop'
    }
  }
  return @{child=$child;parent=$parent}
}

function StopOwnedTask([string]$name,[string]$expectedService) {
  $task=Get-ScheduledTask -TaskName $name -ErrorAction Stop
  if([string]$task.Actions[0].Arguments -notlike ('*'+$expectedService+'*')){
    throw 'Scheduled task does not own the expected stable server script'
  }
  $prior=ReadOwnedListener
  if([string]$task.State -eq 'Running'){
    Stop-ScheduledTask -TaskName $name -ErrorAction Stop
    for($n=0;$n -lt 20;$n++){
      Start-Sleep -Milliseconds 500
      if([string](Get-ScheduledTask -TaskName $name).State -ne 'Running'){break}
    }
  }
  if([string](Get-ScheduledTask -TaskName $name).State -eq 'Running'){
    throw 'Owned task remains running; no process termination'
  }
  $live=ReadOwnedListener
  if($null -ne $live){
    if($null -eq $prior -or
       [int]$prior.child.ProcessId -ne [int]$live.child.ProcessId -or
       [string]$prior.child.CreationDate -ne [string]$live.child.CreationDate -or
       [int]$prior.parent.ProcessId -ne [int]$live.parent.ProcessId -or
       [string]$prior.parent.CreationDate -ne [string]$live.parent.CreationDate){
      throw 'NERAL port listener changed during stop; preserve processes'
    }
    # Stop only independently attested NERAL parent and original child.
    Stop-Process -Id ([int]$live.parent.ProcessId) -Force -ErrorAction Stop
    $childNow=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$live.child.ProcessId) -ErrorAction SilentlyContinue
    if($childNow){
      if([string]$childNow.CreationDate -ne [string]$live.child.CreationDate -or
         [string]$childNow.CommandLine -notmatch '(?i)--experimental-three-tier-hf'){
        throw 'NERAL worker changed after parent stop'
      }
      Stop-Process -Id ([int]$live.child.ProcessId) -Force -ErrorAction Stop
    }
  }
  for($i=0;$i -lt 12;$i++){
    if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){return}
    Start-Sleep -Milliseconds 500
  }
  throw 'Owned NERAL listener did not release port 9690'
}

function WaitForOwnApp {
  for($i=0;$i -lt 240;$i++){
    if($i -gt 0 -and $i % 30 -eq 0){Write-Host ('NERAL_PR561_WAITING_FOR_LOCAL_HTTP seconds='+$i)}
    Start-Sleep -Seconds 1
    try{
      $session=New-Object Microsoft.PowerShell.Commands.WebRequestSession
      $homeResponse=Invoke-WebRequest -UseBasicParsing -WebSession $session -Uri 'http://127.0.0.1:9690/' -TimeoutSec 4
      $apiResponse=Invoke-WebRequest -UseBasicParsing -WebSession $session -Uri 'http://127.0.0.1:9690/api/neral/preview/session' -TimeoutSec 7
      $receipt=$apiResponse.Content | ConvertFrom-Json
      if($homeResponse.StatusCode -eq 200 -and $apiResponse.StatusCode -eq 200 -and $receipt.three_tier_enabled -eq $true -and
         (@($receipt.three_tier_levels) -join ',') -eq 'Luna,Terra,Helios'){return}
    }catch{}
  }
  throw 'NERAL did not pass local HTTP/session readiness'
}


if(-not(Test-Path -LiteralPath $py -PathType Leaf)){throw 'Pinned HF Python absent'}
$existing=Get-ScheduledTask -TaskName $oldName -ErrorAction Stop
if([string]$existing.Actions[0].Arguments -notlike ('*'+$oldServer+'*')){
  throw 'Existing NERAL scheduled task identity mismatch'
}
$extra=Get-ScheduledTask -TaskName $newName -ErrorAction SilentlyContinue
if($extra -and [string]$extra.State -eq 'Running'){
  throw 'Unexpected alternate NERAL V1 task running; refuse to alter stable service'
}
foreach($rel in $approved.Keys){
  if((HashFor $rel $repo) -ne $approved[$rel]){throw "PR561 source changed: $rel"}
  if((HashFor $rel $oldRoot) -ne $approvedPrevious[$rel]){throw "Original source hash drift: $rel"}
}
if((HashFor 'neral_preview_server.py' $oldRoot) -ne 'b8c6bbef8e14ffae09aa8294125c54592490b383'){
  throw 'Original preview server identity changed'
}
$dockerReady=$false
try{& docker info --format '{{.ServerVersion}}' 2>$null|Out-Null
  $dockerReady=($LASTEXITCODE -eq 0)}catch{}
if(-not $dockerReady){throw 'Offline Docker engine unavailable'}
& docker image inspect python:3.11-alpine *> $null
if($LASTEXITCODE -ne 0){throw 'Pinned legacy Python sandbox image absent'}
$probe=@(& docker run --rm --pull never --network none --read-only --memory 128m --cpus 0.5 --pids-limit 32 python:3.11-alpine python -B -c 'print(340*24*365/1000)')
if($LASTEXITCODE -ne 0 -or ($probe -join '').Trim() -ne '2978.4'){
  throw 'Independent sandbox arithmetic cannot run'
}
Write-Host 'NERAL_PR561_HOTPATCH_REAL_DOCKER_READY'

# Inspect the staged source-only app without running a second server. A failed
# import explains the previous candidate launch but does not alter the prior
# working installation.
$stageImports=$false
if(Test-Path -LiteralPath $newRoot -PathType Container){
  $savedPythonPath=$env:PYTHONPATH
  $env:PYTHONPATH=(Join-Path $oldRoot 'local_state/three-tier-web-deps')+[IO.Path]::PathSeparator+$newRoot
  Push-Location -LiteralPath $newRoot
  try{
    $importResult=@(& $py -B -c 'import neral_preview_server, neral_three_tier_runtime, neral_everyday_verified, neral_web_search;print("IMPORTED")' 2>&1)
    $stageImports=($LASTEXITCODE -eq 0 -and ($importResult -join '') -match 'IMPORTED')
  }catch{$stageImports=$false}finally{Pop-Location;$env:PYTHONPATH=$savedPythonPath}
}
Write-Host ('NERAL_PR561_STAGED_IMPORT_HEALTH '+([string]$stageImports).ToLowerInvariant())
# Do not replace a currently unhealthy service; distinguish slow warmup from crash.
WaitForOwnApp
Write-Host 'NERAL_PR561_BASELINE_HTTP_READY_BEFORE_PATCH'

$backup=Join-Path $oldRoot 'local_state/rollback-before-pr561-v1'
if(-not(Test-Path -LiteralPath $backup -PathType Container)){
  New-Item -ItemType Directory -Path $backup -Force | Out-Null
  foreach($rel in $approvedPrevious.Keys){
    $target=Join-Path $backup $rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
    Copy-Item -LiteralPath (Join-Path $oldRoot $rel) -Destination $target -ErrorAction Stop
  }
}
foreach($rel in $approvedPrevious.Keys){
  if((HashFor $rel $backup) -ne $approvedPrevious[$rel]){
    throw "Rollback source has changed: $rel"
  }
}
$modified=$false
$passed=$false
try{
  StopOwnedTask $oldName $oldServer
  $modified=$true
  foreach($rel in $approved.Keys){
    Copy-Item -LiteralPath (Join-Path $repo $rel) -Destination (Join-Path $oldRoot $rel) -Force -ErrorAction Stop
    if((HashFor $rel $oldRoot) -ne $approved[$rel]){throw "Installed V1 file changed: $rel"}
  }
  Write-Host 'NERAL_PR561_INPLACE_V1_SOURCE_HASHES_PASS'
  Start-ScheduledTask -TaskName $oldName -ErrorAction Stop
  WaitForOwnApp
  $js=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:9690/static/neral-preview.js' -TimeoutSec 8
  if($js.StatusCode -ne 200 -or [string]$js.Content -notmatch 'neral-tool-status'){
    throw 'Running in-place NERAL did not serve PR561 UI'
  }
  $browser=@('C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    (Join-Path $env:LOCALAPPDATA 'Google/Chrome/Application/chrome.exe')) |
      Where-Object {Test-Path -LiteralPath $_ -PathType Leaf} | Select-Object -First 1
  if(-not $browser){throw 'No real Chrome or Edge browser'}
  $env:NERAL_CHROME_EXE=$browser
  & node scripts/neral_pr561_live_browser_e2e.cjs
  if($LASTEXITCODE -ne 0){throw 'Real V1 browser acceptance failed'}
  $passed=$true
  Write-Host 'NERAL_PR561_REAL_3_CASES_HOTPATCH_PASS'
}finally{
  if(-not $passed -and $modified){
    Write-Host 'NERAL_PR561_HOTPATCH_ROLLBACK_BEGIN'
    try{
      StopOwnedTask $oldName $oldServer
      foreach($rel in $approvedPrevious.Keys){
        Copy-Item -LiteralPath (Join-Path $backup $rel) -Destination (Join-Path $oldRoot $rel) -Force -ErrorAction Stop
        if((HashFor $rel $oldRoot) -ne $approvedPrevious[$rel]){throw 'Rollback verification failed'}
      }
      Start-ScheduledTask -TaskName $oldName -ErrorAction Stop
      WaitForOwnApp
      Write-Host 'NERAL_PR561_HOTPATCH_ORIGINAL_RESTORED_HTTP_200'
    }catch{
      Write-Host ('NERAL_PR561_HOTPATCH_ROLLBACK_CHECK_REQUIRED '+$_.Exception.GetType().Name)
    }
  }
}
