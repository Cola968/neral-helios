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
  'neral_web_search.py'='d9a34976abe4824b2ef7c7ae65613b7f38ba5d40'
  'neral_everyday_verified.py'='8f04ea7eaf624c5debddc8955012649d6ea1a276'
  'neral_three_tier_runtime.py'='5796223a47c1cec0362548715a775d7731bdaec7'
  'static/neral-preview.js'='77e124c98e61c54f2427269e840a21de7feee03e'
}
$approvedPrevious=[ordered]@{
  'neral_web_search.py'='b8b9fd6ca3acc1286d69a5b758914631d009d16f'
  'neral_everyday_verified.py'='6239a274a5049a32dde7408e1961244d2702dfbc'
  'neral_three_tier_runtime.py'='f3e5e2376999d821dd2a1aefc54be59a59ab0473'
  'static/neral-preview.js'='ce535f84c2e779d6619a968ca1e9b2582d20edbf'
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
  for($i=0;$i -lt 90;$i++){
    Start-Sleep -Seconds 1
    try{
      $session=New-Object Microsoft.PowerShell.Commands.WebRequestSession
      $home=Invoke-WebRequest -UseBasicParsing -WebSession $session -Uri 'http://127.0.0.1:9690/' -TimeoutSec 4
      $receipt=Invoke-RestMethod -WebSession $session -Uri 'http://127.0.0.1:9690/api/neral/preview/session' -TimeoutSec 4
      if($home.StatusCode -eq 200 -and $receipt.three_tier_enabled -eq $true -and
         (@($receipt.three_tier_levels) -join ',') -eq 'Luna,Terra,Helios'){return}
    }catch{}
  }
  throw 'NERAL did not pass local HTTP/session readiness'
}

if(-not(Test-Path $py -PathType Leaf)){throw 'Preinstalled HF Python absent'}
foreach($rel in $approved.Keys) {
  if((HashFor $rel $repo) -ne $approved[$rel]){throw "PR561 target blob changed: $rel"}
  if((HashFor $rel $oldRoot) -ne $approvedPrevious[$rel]){throw "Previous stable blob not approved: $rel"}
}
$oldTask=Get-ScheduledTask -TaskName $oldName -ErrorAction Stop
if([string]$oldTask.Actions[0].Arguments -notlike ('*'+$oldServer+'*')){
  throw 'Prior scheduled task identity mismatch'
}
if((HashFor 'neral_preview_server.py' $oldRoot) -ne 'b8c6bbef8e14ffae09aa8294125c54592490b383'){
  throw 'Unapproved base web app'
}
$dockerOk=$false
try{
  & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
  $dockerOk=($LASTEXITCODE -eq 0)
}catch{}
if(-not $dockerOk){throw 'Docker engine not ready: block all file changes'}
& docker image inspect python:3.11-alpine *> $null
if($LASTEXITCODE -ne 0){throw 'Legacy Docker image missing: no live V1 verification possible'}
$dockerCheck=@(& docker run --rm --pull never --network none --read-only --memory 128m --cpus 0.5 --pids-limit 32 python:3.11-alpine python -B -c 'print(2+3)')
if($LASTEXITCODE -ne 0 -or ($dockerCheck -join '').Trim() -ne '5'){
  throw 'Actual local Docker isolation smoke failed'
}
Write-Host 'NERAL_PR561_PRECHECK_DOCKER_REAL_PASS'

$oldStopped=$false
$newStarted=$false
$complete=$false
try{
  if(-not(Test-Path -LiteralPath $newRoot -PathType Container)){
    if(Test-Path -LiteralPath $newParent){throw 'Partial pre-existing PR561 destination: preserve it'}
    $archive=Join-Path $env:RUNNER_TEMP ('neral-pr561-source-'+$env:GITHUB_RUN_ID+'.zip')
    if(Test-Path $archive){throw 'Release archive path already exists'}
    & $py -B -m scripts.build_neral_1_0_preview --root $repo --out $archive
    if($LASTEXITCODE -ne 0 -or -not(Test-Path -LiteralPath $archive -PathType Leaf)){
      throw 'Pinned NERAL source-only bundle failed'
    }
    New-Item -ItemType Directory -Path $newParent -Force | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $newParent -ErrorAction Stop
  }
  foreach($rel in $approved.Keys) {
    if((HashFor $rel $newRoot) -ne $approved[$rel]){
      throw "Staged release file hash differs from approved PR561: $rel"
    }
  }
  if((HashFor 'neral_preview_server.py' $newRoot) -ne 'b8c6bbef8e14ffae09aa8294125c54592490b383'){
    throw 'Preview server SHA drift in staged release'
  }
  $depRoot=Join-Path $oldRoot 'local_state/three-tier-web-deps'
  if(-not(Test-Path (Join-Path $depRoot 'waitress/__init__.py') -PathType Leaf)){
    throw 'Original isolated Waitress dependencies unavailable'
  }
  $dockerCli=(Get-Command docker.exe -ErrorAction Stop).Source
  $dockerDir=[IO.Path]::GetDirectoryName($dockerCli).Replace("'","''")
  $rootText=$newRoot.Replace("'","''")
  $depText=$depRoot.Replace("'","''")
  $serviceScript=@'
$ErrorActionPreference='Stop'
$root='ROOT_TOKEN'
$py='D:/neral-qlora/venv/Scripts/python.exe'
$env:HF_HOME='D:/neral-qlora-v2/hf-cache'
$env:HF_HUB_CACHE='D:/neral-qlora-v2/hf-cache/hub'
$env:HF_XET_CACHE='D:/neral-qlora-v2/hf-cache/xet'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PATH='DOCKER_TOKEN'+';'+$env:PATH
$env:PYTHONPATH='DEPS_TOKEN'+[IO.Path]::PathSeparator+$root
Set-Location -LiteralPath $root
& $py -B -m neral_preview_server --model qwen3:8b --experimental-three-tier-hf --luna-4b-hf 'D:/neral-qlora-v2/qwen35-4b-hf' --helios-9b-hf 'D:/neral-qlora/checkpoint' --helios-v2-full 'D:/neral-qlora-v2/v2full-43931510428-1-helios'
'@
  $serviceScript=$serviceScript.Replace('ROOT_TOKEN',$rootText).Replace('DEPS_TOKEN',$depText).Replace('DOCKER_TOKEN',$dockerDir)
  if(Test-Path -LiteralPath $newServer -PathType Leaf){
    if([IO.File]::ReadAllText($newServer) -ne $serviceScript){throw 'Existing V1 server script differs; preserve'}
  }else{
    [IO.File]::WriteAllText($newServer,$serviceScript,[Text.UTF8Encoding]::new($false))
  }
  $tokens=$null;$parseErrors=$null
  [Management.Automation.Language.Parser]::ParseFile($newServer,[ref]$tokens,[ref]$parseErrors)|Out-Null
  if(@($parseErrors).Count -ne 0){throw 'New service PS source parse failure'}
  $newTask=Get-ScheduledTask -TaskName $newName -ErrorAction SilentlyContinue
  if($null -eq $newTask){
    $exe=Join-Path $PSHOME 'powershell.exe'
    $args='-NoLogo -NoProfile -ExecutionPolicy Bypass -File "'+$newServer+'"'
    $action=New-ScheduledTaskAction -Execute $exe -Argument $args -WorkingDirectory $newRoot
    $trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddYears(5))
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $newName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'User-approved NERAL PR561 localhost candidate for three real browser cases' -ErrorAction Stop | Out-Null
  }elseif([string]$newTask.Actions[0].Arguments -notlike ('*'+$newServer+'*')){
    throw 'PR561 scheduled task identity mismatch'
  }
  Write-Host 'NERAL_PR561_STAGED_PINNED_REVIEWED_SOURCE_PASS'
  StopOwnedTask $oldName $oldServer
  $oldStopped=$true
  Start-ScheduledTask -TaskName $newName -ErrorAction Stop
  $newStarted=$true
  WaitForOwnApp
  $js=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:9690/static/neral-preview.js' -TimeoutSec 8
  if($js.StatusCode -ne 200 -or [string]$js.Content -notmatch 'neral-tool-status'){
    throw 'Running server is not serving PR561 status UI'
  }
  $browser=@('C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    (Join-Path $env:LOCALAPPDATA 'Google/Chrome/Application/chrome.exe')) |
      Where-Object {Test-Path -LiteralPath $_ -PathType Leaf} | Select-Object -First 1
  if(-not $browser){throw 'Real Chrome/Edge missing'}
  $env:NERAL_CHROME_EXE=$browser
  & node scripts/neral_pr561_live_browser_e2e.cjs
  if($LASTEXITCODE -ne 0){throw 'Real Chrome V1 acceptance failed'}
  $complete=$true
  Write-Host 'NERAL_PR561_THREE_REAL_BROWSER_CASES_PASS_SAFE_TO_MERGE'
}catch{
  Write-Host ('NERAL_PR561_PREMERGE_FAILURE '+$_.Exception.GetType().Name)
  throw
}finally{
  if(-not $complete -and $oldStopped){
    Write-Host 'NERAL_PR561_ROLLBACK_BEGIN'
    if($newStarted){
      try{StopOwnedTask $newName $newServer}catch{Write-Host 'NERAL_PR561_NEW_TASK_STOP_FAILED_CHECK_MANUALLY'}
    }
    if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){
      try{Start-ScheduledTask -TaskName $oldName -ErrorAction Stop
        WaitForOwnApp
        Write-Host 'NERAL_PR561_ORIGINAL_PREVIEW_RESTORED'
      }catch{Write-Host 'NERAL_PR561_ORIGINAL_PREVIEW_RESTART_FAILED'}
    }
  }
}
