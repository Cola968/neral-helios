# NERAL V3.11: pinned frozen-base versus candidate A/B. No model promotion; no public raw holdout logs.
# Stop only the pinned owner service and ALWAYS restore it in finally.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/codex/neral-v311-holdout-ab-20261121'){
  throw 'Wrong ref: no owner GPU test'
}
$root=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local'
$service=Join-Path $root 'Start-Neral-Three-Tier-Service.ps1'
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
$src=$env:GITHUB_WORKSPACE
$python='D:/neral-qlora/venv/Scripts/python.exe'
$checkpoint='D:/neral-qlora/checkpoint'
if(-not(Test-Path -LiteralPath $service -PathType Leaf) -or
   -not(Test-Path -LiteralPath $python -PathType Leaf) -or
   -not(Test-Path -LiteralPath (Join-Path $checkpoint 'config.json') -PathType Leaf)){
  throw 'Pinned service, Python or offline model checkpoint missing'
}
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

# Run against the SAME private, immutable 100-question holdout and 97 seed.
$holdout='D:/neral-qlora-v310-smoke/v310-private-holdout100.jsonl'
$holdoutSHA='6b2e7d2412c950deb8fda3f8ad843a7adab2a3e0f428c580b5d4d909234f0e04'
$holdoutManifest=[IO.Path]::ChangeExtension($holdout,'.manifest.json')
$dataset=Join-Path $src 'datasets/neral_v39_seed97.jsonl'
$checkpoint='D:/neral-qlora/checkpoint'
$python='D:/neral-qlora/venv/Scripts/python.exe'
$trainedManifest='D:/neral-qlora-v311-candidates/44890902637-1/two-epoch-candidate/manifest.json'
$adapter='D:/neral-qlora-v311-candidates/44890902637-1/two-epoch-candidate/adapter'
if(-not(Test-Path -LiteralPath $holdout -PathType Leaf) -or
   -not(Test-Path -LiteralPath $holdoutManifest -PathType Leaf) -or
   -not(Test-Path -LiteralPath $trainedManifest -PathType Leaf) -or
   -not(Test-Path -LiteralPath (Join-Path $adapter 'adapter_config.json') -PathType Leaf) -or
   -not(Test-Path -LiteralPath (Join-Path $checkpoint 'config.json') -PathType Leaf)){
  throw 'Sealed holdout or completed candidate adapter missing'
}
$hs=(Get-FileHash -LiteralPath $holdout -Algorithm SHA256).Hash.ToLowerInvariant()
$hm=Get-Content -LiteralPath $holdoutManifest -Raw -Encoding UTF8 | ConvertFrom-Json
$cm=Get-Content -LiteralPath $trainedManifest -Raw -Encoding UTF8 | ConvertFrom-Json
if($hs -ne $holdoutSHA -or [string]$hm.sha256 -ne $holdoutSHA -or
   [int]$hm.rows -ne 100 -or $hm.no_train_question_overlap -ne $true -or
   [string]$hm.training_sha256 -ne
      (Get-FileHash -LiteralPath $dataset -Algorithm SHA256).Hash.ToLowerInvariant() -or
   [int]$cm.completed_optimizer_steps -ne 26 -or
   [int]$cm.num_train_epochs -ne 2 -or
   [int]$cm.max_seq_length -ne 1024 -or
   [string]$cm.optimizer -ne 'paged_adamw_8bit' -or
   [string]$cm.source_dataset_sha256 -ne [string]$hm.training_sha256 -or
   $cm.adapter_weights_changed -ne $true -or
   [double]$cm.measured_gradient_norm -le 0 -or
   [double]$cm.measured_peak_reserved_gib -ge 10.75 -or
   [int]$cm.measured_system_gpu_peak_mib -ge 11000){
  throw 'Candidate training proof and holdout/source hashes are inconsistent'
}
Write-Host 'NERAL_V311_AB_SEALED_INPUTS_AND_TWO_EPOCH_CANDIDATE_VERIFIED'
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
$state=[string]$task.State
if($state -eq 'Ready'){
  if(@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue).Count -ne 0){
    throw 'Unknown listener while owner task is Ready'
  }
  StartOwned
}elseif($state -ne 'Running'){
  throw 'Owner service cannot be safely resumed'
}
if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
  throw 'Owner NERAL service not verified before A/B'
}
$gpuRows=@(& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if($LASTEXITCODE -ne 0 -or $gpuRows.Count -ne 1 -or $gpuRows[0] -notmatch 'RTX 4070'){
  throw 'Exact owner RTX4070 unavailable'
}
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:WANDB_DISABLED='true'
$env:PYTHONPATH=$src
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$abRoot=Join-Path 'D:/neral-qlora-v311-ab' ($env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
if(Test-Path -LiteralPath $abRoot){throw 'A/B result path already exists; do not overwrite'}
New-Item -ItemType Directory -Path $abRoot -Force | Out-Null
$paused=$false
$finished=$false
$worker=$null
try{
  $paused=$true
  StopOwned
  $mem=@(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if($LASTEXITCODE -ne 0 -or $mem.Count -ne 1 -or
      ($mem[0] -join '') -notmatch '^\s*([0-9]+)\s*$'){
    throw 'GPU VRAM not measurable after owner NERAL service stop'
  }
  $baseline=[int]$Matches[1]
  if($baseline -ge 2048){throw 'Unrelated VRAM consumer after owner service stop'}
  Write-Host ('NERAL_V311_AB_IDLE_VRAM_MIB '+$baseline)
  foreach($variant in @('base','candidate')){
    $target=Join-Path $abRoot ($variant+'-outcomes.jsonl')
    $stdout=Join-Path $abRoot ($variant+'-stdout.txt')
    $stderr=Join-Path $abRoot ($variant+'-stderr.txt')
    $args='-m scripts.neral_v311_ab_holdout --mode evaluate --variant '+$variant+
      ' --train "'+$dataset+'" --holdout "'+$holdout+'" --checkpoint "'+$checkpoint+'"'+
      ' --out "'+$target+'"'
    if($variant -eq 'candidate'){$args+=' --adapter "'+$adapter+'"'}
    $worker=Start-Process -FilePath $python -ArgumentList $args -PassThru -NoNewWindow `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $peak=$baseline
    Write-Host ('NERAL_V311_AB_MODEL_STARTED '+$variant)
    while($true){
      $worker.Refresh()
      if($worker.HasExited){break}
      $read=@(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if($LASTEXITCODE -ne 0 -or $read.Count -ne 1 -or
        ($read[0] -join '') -notmatch '^\s*([0-9]+)\s*$'){
        if(-not $worker.HasExited){Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue}
        throw 'Failed to monitor VRAM; inference aborted'
      }
      $used=[int]$Matches[1]
      if($used -gt $peak){$peak=$used}
      if($used -ge 11000){
        if(-not $worker.HasExited){Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue}
        throw 'A/B VRAM exceeded 11000 MiB safety cutoff'
      }
      Start-Sleep -Milliseconds 400
    }
    $worker.WaitForExit()
    if($null -ne $worker.ExitCode -and [int]$worker.ExitCode -ne 0){
      throw ('A/B '+$variant+' exited with error code '+$worker.ExitCode)
    }
    $summary=[IO.Path]::ChangeExtension($target,'.summary.json')
    if(-not(Test-Path -LiteralPath $summary -PathType Leaf) -or
       -not(Test-Path -LiteralPath $target -PathType Leaf)){
       throw ('A/B '+$variant+' did not write completed manifest')
    }
    $manifest=Get-Content -LiteralPath $summary -Raw -Encoding UTF8 | ConvertFrom-Json
    if([int]$manifest.count -ne 100 -or
        [string]$manifest.holdout_sha256 -ne $holdoutSHA -or
        [string]$manifest.variant -ne $variant -or
        [int](@(Get-Content -LiteralPath $target -Encoding UTF8)).Count -ne 100){
      throw ('A/B '+$variant+' produced incomplete or mismatched cases')
    }
    Write-Host ('NERAL_V311_AB_MODEL_FINISHED '+$variant+' '+[int]$manifest.total+'/100')
    Write-Host ('NERAL_V311_AB_MAX_GPU_MIB '+$variant+' '+$peak)
    $worker=$null
  }
  $report=Join-Path $abRoot 'comparison-private.json'
  & $python -m scripts.neral_v311_ab_holdout --mode compare `
    --base-summary (Join-Path $abRoot 'base-outcomes.summary.json') `
    --candidate-summary (Join-Path $abRoot 'candidate-outcomes.summary.json') `
    --report $report
  if($LASTEXITCODE -ne 0){throw 'Failed to compare both genuine 100-case results'}
  $comparison=Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json
  Write-Host ('NERAL_V311_AB_BASE_TOTAL '+[int]$comparison.baseline_total)
  Write-Host ('NERAL_V311_AB_CANDIDATE_TOTAL '+[int]$comparison.candidate_total)
  foreach($category in @('mathematics','logic','programming','knowledge','web')){
    $line=$comparison.categories.$category
    Write-Host ('NERAL_V311_AB_CATEGORY '+$category+' '+$line.base+' '+$line.candidate)
  }
  if($comparison.a_b_gate_pass){
    Write-Host 'NERAL_V311_MODEL_AB_GATE_PASSED_REQUIRES_SEPARATE_LIVE_E2E'
  }else{
    Write-Host ('NERAL_V311_MODEL_AB_NOT_PROMOTED '+$comparison.failure_reason)
  }
  $finished=$true
}finally{
  if($null -ne $worker){
    $worker.Refresh()
    if(-not $worker.HasExited){Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue}
  }
  if($paused){
    if(-not(HttpReady)){
      if([string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State -ne 'Ready'){
         throw 'Pinned task not safely restartable after A/B'
      }
      StartOwned
    }
    if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
      throw 'Original NERAL port 9690 was not safely restored after A/B'
    }
    Write-Host 'NERAL_V311_AB_ORIGINAL_LIVE_SERVICE_RESTORED_HTTP200'
  }
}
if(-not $finished){throw 'Both A/B variants did not complete'}
