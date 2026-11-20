# Owner-gated post-merge NERAL V3.4 promotion: never replace non-owned listener.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/main'){throw 'Promotion requires main'}
if((& git log -1 --format=%s).Trim() -ne 'ops(v3.4): verified promote PR569'){
  throw 'Explicit owner-approved PR569 promotion missing'
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
$backup=Join-Path $env:LOCALAPPDATA ('neral-v34-pre569-'+$env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
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

# Entire score and source verification happens BEFORE any local mutation.
$expectedSource='388385a9e7686e2ac3f0cb536ca1d7cbc230be45'
$summaryPath=$env:V34_SUMMARY
$detailsPath=$env:V34_DETAILS
if([string]::IsNullOrWhiteSpace($summaryPath) -or [string]::IsNullOrWhiteSpace($detailsPath) -or
   -not(Test-Path -LiteralPath $summaryPath -PathType Leaf) -or
   -not(Test-Path -LiteralPath $detailsPath -PathType Leaf)){
  throw 'Sealed benchmark artifacts missing. Promotion is prohibited.'
}
$report=Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
if([int]$report.cases -ne 100 -or [int]$report.detail_count -ne 100 -or
   [int]$report.strict_e2e_pass -lt 90 -or $report.gate_90_of_100 -ne $true -or
   [string]$report.source_sha -ne $expectedSource -or
   $report.model_promotion_authorized -ne $false){
  throw 'Prospective score, complete receipt, source hash or promotion mode rejected'
}
$detailHash=(Get-FileHash -LiteralPath $detailsPath -Algorithm SHA256).Hash.ToLowerInvariant()
if($detailHash -ne [string]$report.details_sha256){
  throw 'Full original benchmark_details sha256 mismatch'
}
$rawDetails=@(Get-Content -LiteralPath $detailsPath -Encoding UTF8)
if($rawDetails.Count -ne 100){throw 'Incomplete source benchmark details'}
$strict=0
foreach($line in $rawDetails){
  $entry=$line | ConvertFrom-Json -ErrorAction Stop
  if($entry.grader.strict_e2e_pass -eq $true){$strict++}
}
if($strict -ne [int]$report.strict_e2e_pass){throw 'Grade summary does not match details'}
foreach($name in $paths){
  $tested=@(& git rev-parse ($expectedSource+':'+$name))
  $merged=@(& git rev-parse ('HEAD:'+ $name))
  if($LASTEXITCODE -ne 0 -or $tested.Count -ne 1 -or $merged.Count -ne 1 -or
     $tested[0].Trim() -ne $merged[0].Trim()){
    throw ('Merged runtime source deviates from measured GPU code: '+$name)
  }
}
Write-Host 'NERAL_V34_90_OF_100_COMPLETE_SOURCE_AND_ARTIFACT_VERIFIED'

$mutated=$false
$promoted=$false
try{
  if([string]$task.State -eq 'Ready'){StartOwned}
  elseif([string]$task.State -ne 'Running'){throw 'Pinned service is not safely restartable'}
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Original service is not ready with a proven local owner'
  }
  StopOwned
  $mutated=$true
  foreach($name in $paths){
    $from=Join-Path $src $name
    $dest=Join-Path $root $name
    Copy-Item -LiteralPath $from -Destination $dest -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash){
      throw ('Promoted runtime file does not match approved source: '+$name)
    }
  }
  StartOwned
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Promoted service failed pinned local HTTP 200 ownership check'
  }
  $promoted=$true
  Write-Host 'NERAL_V34_PROMOTED_MAIN_PR569_HTTP_200_OK'
}finally{
  if($mutated -and -not $promoted){
    StopOwned
    foreach($name in $paths){
      $target=Join-Path $root $name
      if($had[$name]){
        $copy=Join-Path $backup $name
        Copy-Item -LiteralPath $copy -Destination $target -Force -ErrorAction Stop
        if((Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash -ne
           (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash){
          throw ('Fail-closed rollback hash mismatch: '+$name)
        }
      }elseif(Test-Path -LiteralPath $target -PathType Leaf){
        Remove-Item -LiteralPath $target -Force -ErrorAction Stop
      }
    }
    StartOwned
    Write-Host 'NERAL_V34_PROMOTION_FAILED_ORIGINAL_SERVICE_RESTORED_HTTP_200'
  }
}
if(-not $promoted){throw 'Runtime source not promoted'}
