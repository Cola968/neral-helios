# PR567: owner approves safe exact V2.2 port 9690 restoration with independent recovery gate; then staged V3 browser test.
# Only restore a verified public main blob or the saved owned service script.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=$env:GITHUB_WORKSPACE
$baseline=Join-Path $repo 'baseline'
if(-not(Test-Path -LiteralPath $baseline -PathType Container)){throw 'Pinned V2.2 baseline checkout missing'}
$expected='afe0c6970a35635cce655fd8ad9c67b81714f38b'
$head=(& git -C $baseline rev-parse HEAD).Trim()
if($head -ne $expected){throw 'Refuse to restore from non-V2.2 baseline'}
if($env:GITHUB_HEAD_REF -ne 'codex/neral-v3-evidence-agent'){throw 'Unexpected PR source'}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if([string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){throw 'Unexpected NERAL task owner'}
function GetV22Receipt {
  # PowerShell 5.1 WebRequestSession can fail to roundtrip the local Flask cookie.
  # Use curl.exe with a fresh cookie jar, and never log response bodies or tokens.
  $curl=(Get-Command curl.exe -ErrorAction Stop).Source
  $prefix=Join-Path $env:TEMP ('neral-pr567-v22-'+[guid]::NewGuid().ToString('N'))
  $jar=$prefix+'.cookie';$body=$prefix+'.json'
  $homeStatus='000';$api='000';$homeExit=-1;$apiExit=-1
  try{
    $ErrorActionPreference='Continue'
    $homeStatus=(@(& $curl --noproxy '*' --silent --connect-timeout 3 --max-time 8 --cookie-jar $jar --cookie $jar --output NUL --write-out '%{http_code}' 'http://127.0.0.1:9690/' 2>$null) -join '').Trim()
    $homeExit=$LASTEXITCODE
    $api=(@(& $curl --noproxy '*' --silent --connect-timeout 3 --max-time 8 --cookie-jar $jar --cookie $jar --output $body --write-out '%{http_code}' 'http://127.0.0.1:9690/api/neral/preview/session' 2>$null) -join '').Trim()
    $apiExit=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    $ok=$false
    if($homeExit -eq 0 -and $apiExit -eq 0 -and $homeStatus -eq '200' -and $api -eq '200' -and
       (Test-Path -LiteralPath $body -PathType Leaf)){
      $data=Get-Content -LiteralPath $body -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
      $evidenceProperty=$data.PSObject.Properties['evidence_agent_enabled']
      # Stable V2.2 predates this response field; absence means not V3.
      $ok=($data.three_tier_enabled -eq $true -and
           ($null -eq $evidenceProperty -or $evidenceProperty.Value -eq $false))
    }
    return @{ok=$ok;home=$homeStatus;api=$api}
  }finally{
    $ErrorActionPreference='Stop'
    foreach($item in @($jar,$body)){
      if(Test-Path -LiteralPath $item -PathType Leaf){
        Remove-Item -LiteralPath $item -Force -ErrorAction SilentlyContinue
      }
    }
  }
}
$listeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
if($listeners.Count -gt 0){
  # An existing owned listener must be identified before claiming recovery.
  if($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1'){
    throw 'Unexpected port 9690 listener; no service mutation'
  }
  $owner=Get-CimInstance Win32_Process -Filter ('ProcessId = '+[int]$listeners[0].OwningProcess) -ErrorAction Stop
  if(-not $owner -or [string]$owner.CommandLine -notmatch '(?i)-m\s+neral_preview_server' -or
     [string]$owner.CommandLine -notmatch '(?i)--experimental-three-tier-hf'){
    throw 'Port 9690 is not the registered NERAL Python service'
  }
  $receipt=GetV22Receipt
  Write-Host ('NERAL_PR567_V22_PREEXISTING_RECEIPT home='+$receipt.home+' session='+$receipt.api+' v22='+$receipt.ok)
  if($receipt.ok -eq $true){
    Write-Host 'NERAL_PR567_V22_ALREADY_HEALTHY_HTTP200'
    exit 0
  }
  throw 'NERAL listener present but authenticated V2.2 session not proven; no mutation'
}
if([string]$task.State -ne 'Ready'){
  throw 'Managed task not Ready: preserve unknown running state'
}
$baseState=Join-Path $root 'local_state'
$backups=@(Get-ChildItem -LiteralPath $baseState -Directory -ErrorAction Stop |
    Where-Object {$_.Name -like 'rollback-v3-pr567-*'} |
    Sort-Object LastWriteTime -Descending)
$original=$null
foreach($entry in $backups){
  $candidate=Join-Path $entry.FullName 'Start-Neral-Three-Tier-Service.ps1'
  if(-not(Test-Path -LiteralPath $candidate -PathType Leaf)){continue}
  $text=[IO.File]::ReadAllText($candidate)
  if($text -match 'NERAL_V3_AGENT_ENABLED' -or
     $text -notmatch 'NERAL_CODE_SANDBOX_V2' -or
     $text -notmatch '(?i)-m\s+neral_preview_server' -or
     $text -notmatch '--experimental-three-tier-hf' -or
     $text -notlike '*D:/neral-qlora-v2/v2full-43931510428-1-helios*'){continue}
  $original=$candidate
  break
}
if(-not $original){throw 'No validated original V2.2 service backup: no changes'}
$paths=@(
  'neral_three_tier_runtime.py','neral_tir_experimental.py',
  'neral_docker_runtime.py','neral_preview_server.py',
  'neral_preview_attachments.py','neral_page_reader.py',
  'neral_evidence_agent.py','neral_quantity_plans.py',
  'neral_z3_json_compiler.py','requirements-neral-v3.txt',
  'static/neral-preview.js'
)
$checkpoint=Join-Path $baseState ('pr567-failed-v3-recovery-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $checkpoint | Out-Null
foreach($rel in $paths){
  $old=Join-Path $root $rel
  if(Test-Path -LiteralPath $old -PathType Leaf){
    $dest=Join-Path $checkpoint $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    Copy-Item -LiteralPath $old -Destination $dest -Force -ErrorAction Stop
  }
}
Copy-Item -LiteralPath $service -Destination (Join-Path $checkpoint 'Start-Neral-Three-Tier-Service.ps1') -ErrorAction Stop
$copied=0
$removed=0
foreach($rel in $paths){
  $target=Join-Path $root $rel
  $source=Join-Path $baseline $rel
  if(Test-Path -LiteralPath $source -PathType Leaf){
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw 'Expected pinned main source absent'}
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force -ErrorAction Stop
    $expectedHash=(& git -C $baseline rev-parse ('HEAD:'+$rel)).Trim()
    $actual=(& git -C $baseline -c core.autocrlf=true hash-object --path=$rel $target).Trim()
    if($LASTEXITCODE -ne 0 -or $actual -ne $expectedHash){throw 'Pinned V2.2 file hash mismatch: '+$rel}
    $copied++
  }elseif(Test-Path -LiteralPath $target -PathType Leaf){
    Remove-Item -LiteralPath $target -Force -ErrorAction Stop
    $removed++
  }
}
Copy-Item -LiteralPath $original -Destination $service -Force -ErrorAction Stop
Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
# PR567 regression: a Task Scheduler start request is not proof of a durable service.
# Observe task, listener and HTTP status after five seconds before waiting for warmup.
Start-Sleep -Seconds 5
$fiveSecTask=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
$fiveSecInfo=Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction Stop
$fiveSecListeners=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
$fiveSecHttp=0
try{
  $probe=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:9690/' -TimeoutSec 3
  $fiveSecHttp=[int]$probe.StatusCode
}catch{}
Write-Host ('NERAL_PR567_RESTORE_AFTER_5S task='+[string]$fiveSecTask.State+
  ' listeners='+$fiveSecListeners.Count+' http='+$fiveSecHttp+
  ' last_result='+[int]$fiveSecInfo.LastTaskResult)
$healthy=$false
for($i=0;$i -lt 120;$i++){
  Start-Sleep -Seconds 1
  $receipt=GetV22Receipt
  if($receipt.ok -eq $true){
    $healthy=$true
    break
  }
}
if(-not $healthy){
  $failedTask=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  $failedInfo=Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction Stop
  Write-Host ('NERAL_PR567_RESTORE_FAILED task='+[string]$failedTask.State+
    ' last_result='+[int]$failedInfo.LastTaskResult)
  throw 'Restored V2.2 code but localhost did not start: manual review required'
}
Write-Host ('NERAL_PR567_V22_RECOVERY_HTTP200 sha='+$head+' files='+$copied+' removed_v3='+$removed)
