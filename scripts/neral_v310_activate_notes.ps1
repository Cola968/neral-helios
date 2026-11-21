# NERAL V3.10 - one-time, owner-pinned verified-notes activation, no GPU.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/codex/neral-v310-verified-data-cpu-only-20261120'){
  throw 'Wrong ref: owner V3.10 local notes activation refused'
}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$src=$env:GITHUB_WORKSPACE
$python='D:/neral-qlora/venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $service -PathType Leaf) -or
   -not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'Pinned owner service/Python absent'}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if(@($task.Actions).Count -ne 1 -or
   [string]$task.Actions[0].Arguments -notlike ('*'+$service+'*')){
  throw 'Unrecognized service task identity'
}
$startup=[IO.File]::ReadAllText($service)
if($startup -notmatch 'NERAL_V3_AGENT_ENABLED' -or
   $startup -notmatch '(?i)-m\s+neral_preview_server' -or
   $startup -notmatch '--experimental-three-tier-hf'){
  throw 'Unrecognized service startup code'
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

# Verify a REAL saved one-step LoRA update BEFORE any service mutation.
$training='D:/neral-qlora-v310-smoke/44889263214-1/one-step-adapter/manifest.json'
if(-not(Test-Path -LiteralPath $training -PathType Leaf)){
  throw 'Sealed one-step training manifest absent'
}
$report=Get-Content -LiteralPath $training -Raw -Encoding UTF8 | ConvertFrom-Json
if([int]$report.completed_optimizer_steps -ne 1 -or
   [bool]$report.adapter_weights_changed -ne $true -or
   [double]$report.measured_gradient_norm -le 0 -or
   [double]$report.measured_peak_reserved_gib -ge 10.75 -or
   [int]$report.measured_system_gpu_peak_mib -ge 11000 -or
   [int]$report.lora_rank -ne 16 -or
   [int]$report.max_context -ne 1024 -or
   $report.gradient_checkpointing -ne $true -or
   $report.status -ne 'adapter_saved_not_evaluated'){
  throw 'Missing one-step proof or unsafe memory/configuration'
}
$trainFile=Join-Path $src 'datasets/neral_v39_seed97.jsonl'
if(-not(Test-Path -LiteralPath $trainFile -PathType Leaf) -or
    (Get-FileHash -LiteralPath $trainFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    [string]$report.source_dataset_sha256){
  throw 'Owner test used a different dataset from this source commit'
}
Write-Host ('NERAL_V310_INDEPENDENT_GPU_PROOF_RECHECKED_GRAD '+[double]$report.measured_gradient_norm)
Write-Host ('NERAL_V310_INDEPENDENT_GPU_PROOF_RECHECKED_NVIDIA_MIB '+[int]$report.measured_system_gpu_peak_mib)

# Freeze the prospective holdout offline, privately and without model access.
$env:PYTHONPATH=$src
$holdout='D:/neral-qlora-v310-smoke/v310-private-holdout100.jsonl'
if(-not(Test-Path -LiteralPath $holdout -PathType Leaf)){
  & $python -m scripts.neral_v310_freeze_holdout100 --train $trainFile --out $holdout
  if($LASTEXITCODE -ne 0){throw 'Independent holdout freeze failed'}
}
$holdoutManifest=[IO.Path]::ChangeExtension($holdout,'.manifest.json')
# Path.ChangeExtension replaces only .jsonl -> .manifest.json.
if(-not(Test-Path -LiteralPath $holdoutManifest -PathType Leaf)){
  throw 'Private holdout manifest missing'
}
$freeze=Get-Content -LiteralPath $holdoutManifest -Raw -Encoding UTF8 | ConvertFrom-Json
if([int]$freeze.rows -ne 100 -or $freeze.no_train_question_overlap -ne $true -or
   (Get-FileHash -LiteralPath $holdout -Algorithm SHA256).Hash.ToLowerInvariant() -ne
   [string]$freeze.sha256){
  throw 'Prospective holdout count/digest/disjointness invalid'
}
Write-Host ('NERAL_V310_PRIVATE_HOLDOUT_FROZEN_SHA256 '+$freeze.sha256)

$paths=@('neral_verified_training.py','neral_verified_notes.py','neral_three_tier_runtime.py')
$localDataset=Join-Path $root 'data/verified_training_dataset.jsonl'
$backup=Join-Path $env:LOCALAPPDATA ('neral-v310-pre572-'+$env:GITHUB_RUN_ID)
if(Test-Path -LiteralPath $backup){throw 'Existing backup path: refuse overwrite'}
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$had=@{}
foreach($name in $paths){
  if(-not(Test-Path -LiteralPath (Join-Path $src $name) -PathType Leaf)){
    throw ('Candidate runtime module missing: '+$name)
  }
  $dest=Join-Path $root $name
  $had[$name]=Test-Path -LiteralPath $dest -PathType Leaf
  if($had[$name]){
    Copy-Item -LiteralPath $dest -Destination (Join-Path $backup $name) -ErrorAction Stop
  }
}
$hadDataset=Test-Path -LiteralPath $localDataset -PathType Leaf
if($hadDataset){
  Copy-Item -LiteralPath $localDataset -Destination (Join-Path $backup 'verified_training_dataset.jsonl') -ErrorAction Stop
}
$taskState=[string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State
if($taskState -eq 'Ready' -and @(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -eq 0){
  StartOwned
}elseif($taskState -ne 'Running'){
  throw 'Original NERAL task state prevents safe owner deployment'
}
if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
  throw 'Original NERAL service ownership or HTTP status invalid'
}
$stopping=$false
$mutated=$false
$activated=$false
try{
  $stopping=$true
  StopOwned
  $mutated=$true
  foreach($name in $paths){
    $srcFile=Join-Path $src $name
    $dstFile=Join-Path $root $name
    Copy-Item -LiteralPath $srcFile -Destination $dstFile -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $srcFile -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $dstFile -Algorithm SHA256).Hash){
      throw ('Installed runtime module SHA256 differs: '+$name)
    }
  }
  & $python -m scripts.neral_v39_prepare_local_dataset --seed $trainFile --out $localDataset
  if($LASTEXITCODE -ne 0){throw 'Local verified training examples merge failed'}
  & $python -m scripts.neral_v310_verify_notes_local --installed-root $root --data $localDataset
  if($LASTEXITCODE -ne 0){throw 'Installed exact local note lookup check failed'}
  StartOwned
  if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
    throw 'Updated NERAL service did not respond with owned HTTP200'
  }
  & $python -m scripts.neral_v310_http_notes_smoke --data $localDataset
  if($LASTEXITCODE -ne 0){throw 'Actual local note HTTP/CSRF response failed'}
  $activated=$true
  Write-Host 'NERAL_V310_LOCAL_VERIFIED_NOTES_ACTIVE_AND_HTTP200'
}finally{
  if($stopping -and -not $activated){
    if($mutated){
      StopOwned
      foreach($name in $paths){
        $dstFile=Join-Path $root $name
        if($had[$name]){
          Copy-Item -LiteralPath (Join-Path $backup $name) -Destination $dstFile -Force -ErrorAction Stop
        }elseif(Test-Path -LiteralPath $dstFile -PathType Leaf){
          Remove-Item -LiteralPath $dstFile -Force -ErrorAction Stop
        }
      }
      if($hadDataset){
        Copy-Item -LiteralPath (Join-Path $backup 'verified_training_dataset.jsonl') -Destination $localDataset -Force -ErrorAction Stop
      }elseif(Test-Path -LiteralPath $localDataset -PathType Leaf){
        Remove-Item -LiteralPath $localDataset -Force -ErrorAction Stop
      }
    }
    if(-not(HttpReady)){
      StartOwned
    }
    if(-not(HttpReady)){
      throw 'Rollback failed: pinned NERAL still not HTTP-ready'
    }
    Write-Host 'NERAL_V310_NOTE_ACTIVATION_FAILED_ORIGINAL_SERVICE_RESTORED'
  }
}
if(-not $activated){throw 'Verified note deployment not completed'}
