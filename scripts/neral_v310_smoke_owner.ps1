# NERAL V3.10: one optimizer step only; no model promotion or service mutation.
# Stop only the pinned owner service and ALWAYS restore it in finally.
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($env:GITHUB_REF -ne 'refs/heads/codex/neral-v310-verified-data-cpu-only-20261120'){
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

# GPU and local-service preflight: never stop an unidentified process.
# The owner PC can be online while the approved scheduled task is Ready
# (e.g. a previous safety test left the app stopped). Only that exact
# pinned task is allowed to be started; no unknown port/process is touched.
$state=[string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State
$connections=@(Get-NetTCPConnection -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
Write-Host ('NERAL_V310_OWNER_SERVICE_TASK_STATE '+$state)
Write-Host ('NERAL_V310_LOCAL_9690_LISTENER_COUNT '+$connections.Count)
if($state -eq 'Ready'){
  if($connections.Count -ne 0){throw 'Port 9690 already occupied while owner task stopped'}
  StartOwned
  Write-Host 'NERAL_V310_PINNED_TASK_SAFELY_STARTED_FROM_READY'
}elseif($state -eq 'Running'){
  $ready=$false
  for($n=0;$n -lt 15;$n++){
    if(HttpReady){$ready=$true;break}
    Start-Sleep -Seconds 2
  }
  if(-not $ready){throw 'Owner service still not HTTP-ready; preserve running task'}
}else{
  throw 'Pinned owner task neither Ready nor Running'
}
if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
  throw 'Pinned NERAL service cannot be proven as owned and healthy'
}
$gpuRows=@(& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if($LASTEXITCODE -ne 0 -or $gpuRows.Count -ne 1 -or
   $gpuRows[0] -notmatch 'RTX 4070'){
  throw 'Expected single RTX 4070 GPU not detected'
}
if($gpuRows[0] -match ',\s*([0-9]+)'){$gpuTotal=[int]$Matches[1]}
else{throw 'Cannot read total VRAM'}
if($gpuTotal -lt 11000){throw 'GPU total VRAM below RTX4070 12GB expected'}
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:WANDB_DISABLED='true'
$env:PYTHONPATH=$src
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$dataset=Join-Path $src 'datasets/neral_v39_seed97.jsonl'
$rows=@(Get-Content -LiteralPath $dataset -Encoding UTF8)
if($rows.Count -ne 97){throw 'Sealed training data has wrong number of rows'}
$smokeRoot=Join-Path 'D:/neral-qlora-v310-smoke' ($env:GITHUB_RUN_ID+'-'+$env:GITHUB_RUN_ATTEMPT)
if(Test-Path -LiteralPath $smokeRoot){throw 'Refuse to overwrite owner training output'}
# Separate namespace, outside live service, previous QLoRA and existing checkpoints.
New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
$adapterOut=Join-Path $smokeRoot 'one-step-adapter'
$peakFile=Join-Path $smokeRoot 'peak-nvidia-mib.txt'
$stdout=Join-Path $smokeRoot 'trainer-stdout.txt'
$stderr=Join-Path $smokeRoot 'trainer-stderr.txt'
$worker=$null
$needRestore=$false
$trainingComplete=$false
try{
  Write-Host 'NERAL_V310_OWNER_PINNED_SERVICE_PRECHECK_OK'
  $needRestore=$true
  StopOwned
  $initial=@(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if($LASTEXITCODE -ne 0 -or $initial.Count -ne 1){
    throw 'GPU memory usage is not measurable; refuse smoke'
  }
  if(($initial[0] -join '') -match '^\s*([0-9]+)\s*$'){$baseline=[int]$Matches[1]}
  else{throw 'GPU memory usage invalid; refuse smoke'}
  if($baseline -ge 2048){throw 'Other GPU load remained after stopping the service'}
  $peak=$baseline
  [IO.File]::WriteAllText($peakFile,[string]$peak)
  Write-Host ('NERAL_V310_IDLE_GPU_USED_MIB '+$baseline)
  # Exactly ONE optimizer update, 8 training microbatches from the 97-row set.
  # Quoted paths prevent accidental splitting at spaces.
  $arguments='-m scripts.neral_qlora_verified_v39 --run-gpu --max-steps 1 --context 1024 '+
     '--train "'+$dataset+'" --checkpoint "'+$checkpoint+'" '+
     '--out "'+$adapterOut+'" --nvidia-peak-file "'+$peakFile+'"'
  $worker=Start-Process -FilePath $python -ArgumentList $arguments -PassThru -NoNewWindow `
     -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  Write-Host 'NERAL_V310_R16_ONE_STEP_STARTED'
  $violated=$false
  while($true){
    $worker.Refresh()
    if($worker.HasExited){break}
    $sample=@(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if($LASTEXITCODE -ne 0 -or $sample.Count -ne 1){$violated=$true;break}
    if(($sample[0] -join '') -match '^\s*([0-9]+)\s*$'){$used=[int]$Matches[1]}
    else{$violated=$true;break}
    if($used -gt $peak){
      $peak=$used
      [IO.File]::WriteAllText($peakFile,[string]$peak)
    }
    # Abort early with 264 MiB of margin below 11 GiB (11264 MiB).
    if($used -ge 11000){$violated=$true;break}
    Start-Sleep -Milliseconds 250
  }
  if($violated){
    if(-not $worker.HasExited){Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue}
    throw 'GPU monitoring unavailable or total VRAM exceeded 11000 MiB cutoff'
  }
  $worker.WaitForExit()
  # On Windows PowerShell Start-Process -PassThru may report a null
  # ExitCode even after WaitForExit; never interpret null as a failure.
  # A missing or invalid unique-run manifest below still fails closed.
  if($null -ne $worker.ExitCode -and [int]$worker.ExitCode -ne 0){
    throw ('One-step QLoRA child failed exit '+$worker.ExitCode+'; local logs retained')
  }
  $manifestPath=Join-Path $adapterOut 'manifest.json'
  if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){
    throw 'Verified one-step adapter manifest not written'
  }
  $report=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if([int]$report.completed_optimizer_steps -ne 1 -or
     [bool]$report.adapter_weights_changed -ne $true -or
     [double]$report.measured_gradient_norm -le 0 -or
     [double]$report.measured_peak_reserved_gib -ge 10.75 -or
     [int]$report.measured_system_gpu_peak_mib -ge 11000){
    throw 'Manifest lacks exact positive-gradient/adapter/VRAM proof'
  }
  $trainingComplete=$true
  Write-Host ('NERAL_V310_R16_1STEP_OK_GRAD_NORM '+[double]$report.measured_gradient_norm)
  Write-Host ('NERAL_V310_R16_1STEP_OK_PEAK_RESERVED_GIB '+[double]$report.measured_peak_reserved_gib)
  Write-Host ('NERAL_V310_R16_1STEP_OK_NVIDIA_PEAK_MIB '+$peak)
  Write-Host 'NERAL_V310_ONE_STEP_ADAPTER_CHANGED_NOT_DEPLOYED'
}finally{
  if($null -ne $worker){
    $worker.Refresh()
    if(-not $worker.HasExited){
      Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue
    }
  }
  if($needRestore){
    $taskNow=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    if(-not(HttpReady)){
      if([string]$taskNow.State -ne 'Ready'){
        throw 'Service not safely restartable after one-step smoke'
      }
      StartOwned
    }
    if(-not(HttpReady) -or $null -eq (ReadOwnedListener)){
      throw 'NERAL service HTTP200/owner restore failed'
    }
    Write-Host 'NERAL_V310_SERVICE_RESTORED_HTTP200'
  }
}
if(-not $trainingComplete){throw 'One-step training incomplete'}
