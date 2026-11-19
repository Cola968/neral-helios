# NERAL V3 PR #567: owner-approved four-case Chrome retry after isolated pinned grammar dependency fix; never modify model weights; rollback required.
# Owner-approved RTX4070 V2.2 final acceptance following green full CPU regressions; strict rollback of 4 files on any failed Chrome case.
# Owner-approved numeric-only browser diagnosis for executed but incorrect hut amounts; rollback on any unproven case.
# V2.2 owner-approved GPU reacceptance after locale/person-night repair; real hut, real birthday, simulated UI fallback and rollback gate.
# Owner-authorized rerun for safe HTTP-only diagnosis of genuine hut sandbox failure; all live changes rollback unless both Chrome cases pass.
# Approval renewed for isolated V2.2 browser-gate concurrency; user PC stays protected by exact process checks and rollback.
# V2.2 RTX4070 real browser acceptance: hut+probability; rollback on any failure; owner-approved.
# Owner-authorized V2.2 RTX4070/Chrome local acceptance; changes only allowed NERAL files, rollback mandatory.
# Owner-approved real V2.2/RTX4070/Chrome retest after safe Docker Desktop cold-start parser gate; rollback remains mandatory.
# Corrected PowerShell 5.1 Docker Desktop cold-boot readiness; owner-approved V2.2 must still pass 3/3 + Chrome.
# Owner-approved cold-boot Docker Desktop recovery and pinned offline image preflight; require real 3/3 + Chrome or rollback.
# Owner-approved RTX4070 live retest with privacy-safe failure categories; 3/3 + Chrome and rollback still mandatory.
# User-approved final reboot-ready V2.2 RTX4070+Chrome reacceptance; merge only on 3/3 plus browser and rollback otherwise.
# Owner-approved reboot recovery retest: exact registered NERAL task only, 3/3 real HF/Docker, Chrome, rollback otherwise.
# Explicit owner approval: rerun V2.2 real RTX4070 and Chrome after exact browser response diagnostics; retain 3/3 + Chrome/rollback gates.
# Reapproved V2.2 real RTX4070 acceptance after web-currency source check and bounded repair; rollback on failure.
# Explicit V2.2 owner approval: run real RTX4070 live3 and Chrome with automatic rollback; main untouched until pass.
# Owner-approved V2.2 staged Windows upgrade. Only the attested NERAL service
# and eight fixed runtime files are touched; no trainer, model weights or user files.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=$env:GITHUB_WORKSPACE
if(-not $repo -or -not(Test-Path -LiteralPath $repo -PathType Container)){throw 'Windows runner workspace unavailable'}
Set-Location -LiteralPath $repo
$merged='c864e1d3537c6ccf48eb1421d8ceb4382e2125fb'
& git merge-base --is-ancestor $merged HEAD
if($LASTEXITCODE -ne 0){throw 'Candidate does not include approved V2.0 main base'}
$isMain=($env:GITHUB_REF -eq 'refs/heads/main')
$isPr=($env:GITHUB_HEAD_REF -eq 'codex/neral-v3-evidence-agent')
if(-not $isMain -and -not $isPr){throw 'Unapproved branch for V3 local upgrade'}
if($isMain){
  $title=(& git log -1 --format=%s).Trim()
  if($title -notmatch 'PR #567'){throw 'Main deployment requires the explicitly approved PR #567 merge'}
}
$head=(& git rev-parse HEAD).Trim()
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$paths=@(
  'neral_three_tier_runtime.py',
  'neral_tir_experimental.py',
  'neral_docker_runtime.py',
  'neral_preview_server.py',
  'neral_preview_attachments.py',
  'neral_page_reader.py',
  'neral_evidence_agent.py',
  'neral_quantity_plans.py',
  'neral_z3_json_compiler.py',
  'requirements-neral-v3.txt',
  'static/neral-preview.js'
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
function TestPinnedDockerImage([string]$docker) {
  # A stopped Docker engine is an expected negative readiness result.
  $ErrorActionPreference='Continue'
  $result=@(& $docker image inspect --format '{{.Id}}' neral-python-tools:v2 2>$null)
  return ($LASTEXITCODE -eq 0 -and $result.Count -ge 1)
}
function TestPinnedDockerProbe([string]$docker) {
  $ErrorActionPreference='Continue'
  $result=@(& $docker run --rm --pull never --network none --read-only --memory 256m --cpus 0.5 --pids-limit 32 neral-python-tools:v2 python -B -c 'import sympy,z3;print(7*8)' 2>$null)
  return ($LASTEXITCODE -eq 0 -and ($result -join '').Trim() -eq '56')
}
function EnsurePinnedDockerReady {
  # The Docker client and Desktop binary must belong to the SAME existing
  # user-scoped DockerDesktop install. No downloads, installs or system services.
  $expectedCli=Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
  $desktopExe=Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/Docker Desktop.exe'
  $command=Get-Command docker.exe -ErrorAction Stop
  $docker=[string]$command.Source
  if(([IO.Path]::GetFullPath($docker)) -ine
     ([IO.Path]::GetFullPath($expectedCli)) -or
     -not(Test-Path -LiteralPath $desktopExe -PathType Leaf)){
    throw 'Unrecognized Docker Desktop installation; no local changes'
  }
  if(-not(TestPinnedDockerImage $docker)){
    $existing=@(Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue)
    if($existing.Count -eq 0){
      Start-Process -FilePath $desktopExe -ErrorAction Stop
      Write-Host 'NERAL_V3_PR567_EXISTING_DOCKER_DESKTOP_START_REQUESTED'
    }
    $ready=$false
    for($i=0;$i -lt 75;$i++){
      Start-Sleep -Seconds 2
      if(TestPinnedDockerImage $docker){$ready=$true;break}
    }
    if(-not $ready){
      throw 'Docker Desktop engine or existing solver image unavailable; V2.0 preserved'
    }
  }
  if(-not(TestPinnedDockerProbe $docker)){
    throw 'Existing offline Docker solver did not execute; no V2.2 installation'
  }
  Write-Host 'NERAL_V3_PR567_DOCKER_PINNED_READY'
}

if(-not(Test-Path -LiteralPath $service -PathType Leaf)){throw 'Managed NERAL service missing'}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if([string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
  throw 'Unexpected scheduled task identity; refuse to restart'
}
$startText=[IO.File]::ReadAllText($service)
if($startText -notmatch 'NERAL_CODE_SANDBOX_V2' -or
   $startText -notmatch '(?i)-m\s+neral_preview_server' -or
   $startText -notmatch '--experimental-three-tier-hf'){
  throw 'Running service is not the approved V2 service'
}
Write-Host ('NERAL_V3_PR567_OWNED_TASK_INITIAL_STATE '+[string]$task.State)
if([string]$task.State -ne 'Running'){
  if([string]$task.State -ne 'Ready'){
    throw 'NERAL task neither Running nor Ready; refuse unknown service state'
  }
  # After a user PC reboot the signed task may be Ready, with no listener.
  # Start ONLY the exact previously registered NERAL service. Never touch
  # another process or replace a task definition.
  $otherListener=ReadOwnedListener
  if($null -ne $otherListener){throw 'Ready task already has a port owner; preserve it'}
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
  WaitForOwnApp
  $task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  if([string]$task.State -ne 'Running'){
    throw 'Owned NERAL task did not enter Running state after restart'
  }
  Write-Host 'NERAL_V3_PR567_RESTART_EXISTING_TASK_HTTP_200'
}
$before=ReadOwnedListener
if($null -eq $before){throw 'Expected owned NERAL HTTP listener absent'}
# Use only the already-installed local inference Python. Never mutate the
# pinned QLoRA training environment or download packages during a live gate.
$python='D:/neral-qlora/venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){
  throw 'Expected isolated NERAL inference environment not installed'
}
# Install only the two small pinned JSON grammar wheels into a separate
# LOCAL STATE directory, never the QLoRA/model environment. No torch updates.
$deps=Join-Path $root 'local_state/neral-v3-json-deps'
$existing=Join-Path $root 'local_state/three-tier-web-deps'
$previousPythonPath=$env:PYTHONPATH
$env:PYTHONPATH=$deps+[IO.Path]::PathSeparator+$existing+[IO.Path]::PathSeparator+$root
& $python -B -c 'from lmformatenforcer import JsonSchemaParser; import interegular; print("NERAL_V3_LMFE_AVAILABLE")' 2>$null
$hasEnforcer=($LASTEXITCODE -eq 0)
if(-not $hasEnforcer){
  New-Item -ItemType Directory -Force -Path $deps | Out-Null
  # Binary wheels only, no transitive dependency installation or model
  # downloads. Existing pinned venv must already have Pydantic/YAML.
  & $python -m pip install --disable-pip-version-check --no-input --only-binary=:all: --no-deps --target $deps 'lm-format-enforcer==0.11.3' 'interegular==0.3.3'
  if($LASTEXITCODE -ne 0){throw 'V3 isolated JSON enforcer wheel installation failed; service unchanged'}
  & $python -B -c 'from lmformatenforcer import JsonSchemaParser; import interegular; print("NERAL_V3_LMFE_AVAILABLE")'
  if($LASTEXITCODE -ne 0){throw 'V3 isolated dependency import failed; service unchanged'}
}
$env:PYTHONPATH=$previousPythonPath
# Verify local offline Docker BEFORE copying/stopping any V2.2 runtime files.
EnsurePinnedDockerReady
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
Write-Host ('NERAL_V3_PR567_MAIN_SYNC_PREFLIGHT_PASS sha='+$head+' changed='+$changes.Count)
$backupRoot=Join-Path $root ('local_state/rollback-v3-pr567-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$serviceBackup=Join-Path $backupRoot 'Start-Neral-Three-Tier-Service.ps1'
Copy-Item -LiteralPath $service -Destination $serviceBackup -ErrorAction Stop
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
      throw "Installed V3 source mismatch: $rel"
    }
  }
  # Opt in only this owned service. The exact original is stored for rollback.
  # A process-scoped env var avoids rewriting the trusted executable arguments.
  if($startText -notmatch 'NERAL_V3_AGENT_ENABLED'){
    if($startText -notmatch [regex]::Escape('Set-Location -LiteralPath $root')){
      throw 'Managed service script lacks audited V2.2 launch marker'
    }
    # Prepend explicit agent opt-in; add the isolated import folder AFTER
    # V2.2's original PYTHONPATH assignment and BEFORE the Python command.
    $header = '$env:NERAL_V3_AGENT_ENABLED="1"' + [Environment]::NewLine
    $addPath = '$env:PYTHONPATH=(Join-Path $PSScriptRoot ''local_state/neral-v3-json-deps'') + [IO.Path]::PathSeparator + $env:PYTHONPATH' + [Environment]::NewLine
    $newService=$header+$startText.Replace('Set-Location -LiteralPath $root',
                                           $addPath+'Set-Location -LiteralPath $root')
    [IO.File]::WriteAllText($service, $newService,
      ([System.Text.UTF8Encoding]::new($false)))
  }
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
  WaitForOwnApp
  $s=Invoke-RestMethod -Uri 'http://127.0.0.1:9690/api/neral/preview/session' -TimeoutSec 12
  if($s.evidence_agent_enabled -ne $true){throw 'Port 9690 did not enable V3 evidence agent'}
  $after=ReadOwnedListener
  if($null -eq $after -or
     ([int]$before.child.ProcessId -eq [int]$after.child.ProcessId -and
      [string]$before.child.CreationDate -eq [string]$after.child.CreationDate)){
    throw 'A new NERAL service process was not proved'
  }
  if(-not $isMain){
    # On PR branch: real Chrome at the requested live port, four distinct
    # tasks in a single chat, original citations and actual Docker/Z3.
    $browser=@('C:/Program Files/Google/Chrome/Application/chrome.exe',
      'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
      'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
      (Join-Path $env:LOCALAPPDATA 'Google/Chrome/Application/chrome.exe')) |
        Where-Object {Test-Path -LiteralPath $_ -PathType Leaf} | Select-Object -First 1
    if(-not $browser){throw 'Chrome/Edge executable unavailable'}
    $env:NERAL_CHROME_EXE=$browser
    & node scripts/neral_v3_pr567_browser_e2e.cjs
    if($LASTEXITCODE -ne 0){throw 'V3 real Chrome 4-of-4 gate not passed'}
    Write-Host 'NERAL_V3_PR567_REAL_CHROME_4OF4_PASS'
  } else {
    Write-Host 'NERAL_V3_PR567_MAIN_SERVICE_HTTP200'
  }
  $success=$true
  Write-Host ('NERAL_V3_PR567_RESTART_HTTP_200 sha='+$head+' files='+$paths.Count)
}finally{
  if(-not $success -and $stopAttempted){
    Write-Host 'NERAL_V3_PR567_MAIN_SYNC_ROLLBACK_BEGIN'
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
      Copy-Item -LiteralPath $serviceBackup -Destination $service -Force -ErrorAction Stop
      Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
      WaitForOwnApp
      Write-Host 'NERAL_V3_PR567_ROLLBACK_HTTP_200'
    }catch{
      Write-Host ('NERAL_V3_PR567_MAIN_SYNC_ROLLBACK_NEEDS_REVIEW '+$_.Exception.GetType().Name)
    }
  }
}
