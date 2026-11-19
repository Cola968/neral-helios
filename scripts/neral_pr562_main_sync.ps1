# Owner-approved PR #562 main synchronization. Only the attested NERAL service
# and a fixed allowlist of runtime/UI files are touched; no trainer, model weights or user files.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=$env:GITHUB_WORKSPACE
if(-not $repo -or -not(Test-Path -LiteralPath $repo -PathType Container)){throw 'Windows runner workspace unavailable'}
Set-Location -LiteralPath $repo
$merged='bf06134566caf0b48932c3690d7a4c570b9bef6a'
& git merge-base --is-ancestor $merged HEAD
if($LASTEXITCODE -ne 0){throw 'Checked-out HEAD does not include merged PR562'}
if($env:GITHUB_REF -ne 'refs/heads/main'){throw 'Main ref not checked out'}
$head=(& git rev-parse HEAD).Trim()
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$paths=@(
  'neral_three_tier_runtime.py',
  'neral_z3_json_compiler.py',
  'neral_tir_experimental.py',
  'neral_docker_runtime.py',
  'neral_v21_attachment_data.py',
  'neral_preview_attachments.py',
  'neral_preview_server.py',
  'neral_everyday_verified.py',
  'neral_web_search.py',
  'neral_web_api.py',
  'templates/neral_preview.html',
  'static/neral-preview.js',
  'static/neral-clean-ui.css',
  'static/neral-web-live.js'
)
function HashFor([string]$rel,[string]$from){
  $file=Join-Path $from $rel
  if(-not(Test-Path -LiteralPath $file -PathType Leaf)){return $null}
  $actual=@(& git -c core.autocrlf=true hash-object --path=$rel $file)
  if($LASTEXITCODE -ne 0 -or @($actual).Count -ne 1){throw "Cannot hash approved runtime file: $rel"}
  return [string]$actual[0].Trim()
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
function StopOwnedTask([string]$name,[string]$expectedService){
  $task=Get-ScheduledTask -TaskName $name -ErrorAction Stop
  if([string]$task.Actions[0].Arguments -notlike ('*'+$expectedService+'*')){
    throw 'Scheduled task does not own the expected NERAL service script'
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
    throw 'Owned task remains running; cannot stop'
  }
  $live=ReadOwnedListener
  if($null -ne $live){
    if($null -eq $prior -or
       [int]$prior.child.ProcessId -ne [int]$live.child.ProcessId -or
       [string]$prior.child.CreationDate -ne [string]$live.child.CreationDate -or
       [int]$prior.parent.ProcessId -ne [int]$live.parent.ProcessId -or
       [string]$prior.parent.CreationDate -ne [string]$live.parent.CreationDate){
      throw 'Listener changed during stop: preserve unknown process'
    }
    Stop-Process -Id ([int]$live.parent.ProcessId) -Force -ErrorAction Stop
    $childNow=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$live.child.ProcessId) -ErrorAction SilentlyContinue
    if($childNow){
      if([string]$childNow.CreationDate -ne [string]$live.child.CreationDate -or
         [string]$childNow.CommandLine -notmatch '(?i)--experimental-three-tier-hf'){
        throw 'NERAL child identity changed after parent stop'
      }
      Stop-Process -Id ([int]$live.child.ProcessId) -Force -ErrorAction Stop
    }
  }
  for($i=0;$i -lt 12;$i++){
    if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){return}
    Start-Sleep -Milliseconds 500
  }
  throw 'Owned NERAL process did not release port 9690'
}
function WaitForOwnApp {
  for($i=0;$i -lt 180;$i++){
    Start-Sleep -Seconds 1
    try{
      $session=New-Object Microsoft.PowerShell.Commands.WebRequestSession
      $h=Invoke-WebRequest -UseBasicParsing -WebSession $session -Uri 'http://127.0.0.1:9690/' -TimeoutSec 5
      $a=Invoke-WebRequest -UseBasicParsing -WebSession $session -Uri 'http://127.0.0.1:9690/api/neral/preview/session' -TimeoutSec 8
      $receipt=$a.Content | ConvertFrom-Json
      if($h.StatusCode -eq 200 -and $a.StatusCode -eq 200 -and
         $receipt.three_tier_enabled -eq $true -and
         (@($receipt.three_tier_levels) -join ',') -eq 'Luna,Terra,Helios'){
        return
      }
    }catch{}
  }
  throw 'NERAL local homepage/API did not return HTTP 200 and expected modes'
}
if(-not(Test-Path -LiteralPath $service -PathType Leaf)){throw 'Managed NERAL service missing'}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if([string]$task.State -ne 'Running' -or
   [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
  throw 'Unexpected scheduled task; refuse to restart'
}
$startText=[IO.File]::ReadAllText($service)
if($startText -notmatch 'NERAL_CODE_SANDBOX_V2' -or
   $startText -notmatch '(?i)-m\s+neral_preview_server' -or
   $startText -notmatch '--experimental-three-tier-hf'){
  throw 'Running service is not the approved V2 service'
}
$before=ReadOwnedListener
if($null -eq $before){throw 'Expected owned NERAL HTTP listener absent'}
$changes=@()
foreach($rel in $paths){
  $repoHash=(HashFor $rel $repo)
  $expected=(& git rev-parse ('HEAD:'+$rel)).Trim()
  if($null -eq $repoHash -or $repoHash -ne $expected){
    throw "Checked-out main file failed integrity check: $rel"
  }
  $installedHash=(HashFor $rel $root)
  if($installedHash -ne $expected){
    $changes+=@{rel=$rel; existed=($null -ne $installedHash)}
  }
}
Write-Host ('NERAL_PR562_MAIN_SYNC_PREFLIGHT_PASS sha='+$head+' changed='+$changes.Count)
$backupRoot=Join-Path $root ('local_state/rollback-pr562-main-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
foreach($item in $changes){
  if($item.existed){
    $destination=Join-Path $backupRoot $item.rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root $item.rel) -Destination $destination -ErrorAction Stop
  }
}
$success=$false
$stopAttempted=$false
try{
  $stopAttempted=$true
  StopOwnedTask $taskName $service
  foreach($item in $changes){
    $destination=Join-Path $root $item.rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo $item.rel) -Destination $destination -Force -ErrorAction Stop
  }
  foreach($rel in $paths){
    if((HashFor $rel $root) -ne (HashFor $rel $repo)){
      throw "Installed main source mismatch: $rel"
    }
  }
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
  WaitForOwnApp
  $after=ReadOwnedListener
  if($null -eq $after -or
     ([int]$before.child.ProcessId -eq [int]$after.child.ProcessId -and
      [string]$before.child.CreationDate -eq [string]$after.child.CreationDate)){
    throw 'A new NERAL service process was not proved'
  }
  $success=$true
  Write-Host ('NERAL_PR562_MAIN_SYNC_RESTART_HTTP_200 sha='+$head+' files='+$paths.Count)
}finally{
  if(-not $success -and $stopAttempted){
    Write-Host 'NERAL_PR562_MAIN_SYNC_ROLLBACK_BEGIN'
    try {
      StopOwnedTask $taskName $service
      foreach($item in $changes){
        $destination=Join-Path $root $item.rel
        if($item.existed){
          Copy-Item -LiteralPath (Join-Path $backupRoot $item.rel) -Destination $destination -Force -ErrorAction Stop
        }elseif(Test-Path -LiteralPath $destination -PathType Leaf){
          Remove-Item -LiteralPath $destination -Force -ErrorAction Stop
        }
      }
      Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
      WaitForOwnApp
      Write-Host 'NERAL_PR562_MAIN_SYNC_ROLLBACK_HTTP_200'
    }catch{
      Write-Host ('NERAL_PR562_MAIN_SYNC_ROLLBACK_NEEDS_REVIEW '+$_.Exception.GetType().Name)
    }
  }
}
