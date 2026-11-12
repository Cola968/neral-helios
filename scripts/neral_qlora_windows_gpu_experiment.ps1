# Local Windows RTX4070 one-step Qwen3.5 9B QLoRA smoke, then optional full.
# No cloud inference, no overwrite of production Ollama models, no deletion.
param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('smoke','full')]
  [string]$Phase
)
$ErrorActionPreference='Stop'
if($env:GITHUB_REPOSITORY -ne 'Cola968/neral-helios' -or
   $env:NERAL_QLORA_MANUAL_AUTH -ne 'NERAL_QLORA_GPU_TRAIN_EXPLICIT'){
  throw 'No explicit user-authorized local GPU training'
}
$repo=Resolve-Path .
$expected=$env:PINNED_TRAIN_SHA
if($expected -notmatch '^[0-9a-f]{40}
$actual=(& git rev-parse HEAD).Trim()
if($LASTEXITCODE -ne 0 -or $actual -ne $expected){
  throw 'Wrong immutable training candidate'
}
$root='D:\NERAL-QLoRA-Experiments\Qwen35-9B-20261112'
$drive=Get-PSDrive -Name D -ErrorAction Stop
if($drive.Free -lt 70GB){throw 'Need at least 70 GiB available on D before local 9B package/model staging'}
$osDrive=Get-PSDrive -Name C -ErrorAction Stop
if($osDrive.Free -lt 10GB){throw 'Need at least 10 GiB remaining on C for Windows'}
$base=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $base -PathType Leaf)){throw 'Local Python 3.12 not installed'}
$env:HF_HOME=Join-Path $root 'hf-cache'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'
$env:WANDB_DISABLED='true'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PIP_CACHE_DIR=Join-Path $root 'pip-cache'
$env:TMP=Join-Path $root 'temp'
$env:TEMP=$env:TMP
$env:TMPDIR=$env:TMP
foreach($dir in @($root,$env:HF_HOME,$env:HF_HUB_CACHE,$env:PIP_CACHE_DIR,$env:TMP)){
  New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
}
$venv=Join-Path $root 'venv'
$python=Join-Path $venv 'Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){
  if(Test-Path -LiteralPath $venv){throw 'Partial venv found; refuse automatic repair'}
  & $base -B -m venv --system-site-packages $venv
  if($LASTEXITCODE -ne 0){throw 'Unable to create isolated D drive environment'}
}
& $python -B -c "import torch;print('LOCAL_TORCH_CUDA=' + str(torch.cuda.is_available()))"
if($LASTEXITCODE -ne 0){throw 'Training torch is broken'}
$installed=& $python -B -c "import importlib.util; print('yes' if importlib.util.find_spec('unsloth') else 'no')"
if($LASTEXITCODE -ne 0){throw 'Unable to inspect local Unsloth installation'}
if(($installed | Out-String).Trim() -ne 'yes'){
  Write-Host 'UNSLOTH_INSTALL_BEGIN_ISOLATED_D_DRIVE'
  & $python -B -m pip install --disable-pip-version-check --no-input --timeout 60 --retries 1 unsloth
  if($LASTEXITCODE -ne 0){throw 'Unsloth installation failed (existing env untouched)'}
}
& $python -B -c "from unsloth import FastModel;import torch;print('UNSLOTH_IMPORT_OK='+str(torch.cuda.is_available()))"
if($LASTEXITCODE -ne 0){throw 'Unsloth incompatible with this Windows Python/Torch stack'}
$train=Join-Path $root 'train.jsonl'
$holdout=Join-Path $repo 'datasets\neral_qwen35_holdout_v1.jsonl'
if(-not(Test-Path -LiteralPath $train -PathType Leaf)){
  Push-Location $repo
  try {
    & $python -B -m scripts.neral_qlora_dataset --out $train --holdout $holdout --pairs 360
    if($LASTEXITCODE -ne 0){throw 'Verified dataset generation failed'}
  } finally {Pop-Location}
}
$digest=(Get-FileHash -Algorithm SHA256 -LiteralPath $train).Hash.ToLowerInvariant()
if($digest -ne '2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79'){
  throw 'Training data does not match frozen 720-row hash'
}
Push-Location $repo
try {
  & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out (Join-Path $root 'data_validation_only') --check-data-only
  if($LASTEXITCODE -ne 0){throw 'Holdout disjointness or training structure failed'}
} finally {Pop-Location}
# Running original local chat model must be unloaded to free VRAM.
$gpu=& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
if($LASTEXITCODE -ne 0){throw 'Cannot read RTX GPU'}
$values=($gpu|Select-Object -First 1) -split ','
if($values.Count -ne 2){throw 'Malformed GPU status'}
if([int]$values[0].Trim() -lt 9216){
  Write-Host 'UNLOADING_OLLAMA_MODEL_FOR_LOCAL_TRAINING'
  & ollama stop qwen3.5:9b
  if($LASTEXITCODE -ne 0){throw 'Ollama model did not unload cleanly'}
  Start-Sleep -Seconds 3
}
$gpu=& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
$values=($gpu|Select-Object -First 1) -split ','
Write-Host "GPU_FREE_MIB_AFTER_UNLOAD=$($values[0].Trim())"
if([int]$values[0].Trim() -lt 9216 -or [int]$values[1].Trim() -gt 25){
  throw 'GPU is not sufficiently free for 9B QLoRA'
}
$runOut=Join-Path $root ('adapter_'+$Phase)
if(Test-Path -LiteralPath $runOut){throw 'Refuse overwrite previous adapter'}
Write-Host "QLORA_PHASE_BEGIN=$Phase"
Push-Location $repo
try {
  if($Phase -eq 'smoke'){
    & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out $runOut --model unsloth/Qwen3.5-9B --epochs 1 --max-steps 1
  }else{
    if(-not(Test-Path -LiteralPath (Join-Path $root 'adapter_smoke\adapters\adapter_config.json'))){
      throw 'One-step GPU smoke has not passed'
    }
    $env:HF_HUB_OFFLINE='1'
    & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out $runOut --model unsloth/Qwen3.5-9B --epochs 1
  }
  if($LASTEXITCODE -ne 0){throw 'Local QLoRA GPU training failed'}
} finally {Pop-Location}
if(-not(Test-Path -LiteralPath (Join-Path $runOut 'adapters\adapter_config.json') -PathType Leaf)){
  throw 'No valid PEFT adapter was saved'
}
Write-Host "QLORA_PHASE_SUCCESS=$Phase"
# Never upload model checkpoint, tokenizer, train text, or HuggingFace credentials.
){throw 'Missing immutable training commit pin'}
$actual=(& git rev-parse HEAD).Trim()
if($LASTEXITCODE -ne 0 -or $actual -ne $expected){
  throw 'Wrong immutable training candidate'
}
$root='D:\NERAL-QLoRA-Experiments\Qwen35-9B-20261112'
$drive=Get-PSDrive -Name D -ErrorAction Stop
if($drive.Free -lt 70GB){throw 'Need at least 70 GiB available on D before local 9B package/model staging'}
$osDrive=Get-PSDrive -Name C -ErrorAction Stop
if($osDrive.Free -lt 10GB){throw 'Need at least 10 GiB remaining on C for Windows'}
$base=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $base -PathType Leaf)){throw 'Local Python 3.12 not installed'}
$env:HF_HOME=Join-Path $root 'hf-cache'
$env:HF_HUB_CACHE=Join-Path $env:HF_HOME 'hub'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:WANDB_DISABLED='true'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PIP_CACHE_DIR=Join-Path $root 'pip-cache'
$env:TMP=Join-Path $root 'temp'
$env:TEMP=$env:TMP
$env:TMPDIR=$env:TMP
foreach($dir in @($root,$env:HF_HOME,$env:HF_HUB_CACHE,$env:PIP_CACHE_DIR,$env:TMP)){
  New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
}
$venv=Join-Path $root 'venv'
$python=Join-Path $venv 'Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){
  if(Test-Path -LiteralPath $venv){throw 'Partial venv found; refuse automatic repair'}
  & $base -B -m venv --system-site-packages $venv
  if($LASTEXITCODE -ne 0){throw 'Unable to create isolated D drive environment'}
}
& $python -B -c "import torch;print('LOCAL_TORCH_CUDA=' + str(torch.cuda.is_available()))"
if($LASTEXITCODE -ne 0){throw 'Training torch is broken'}
$installed=& $python -B -c "import importlib.util; print('yes' if importlib.util.find_spec('unsloth') else 'no')"
if($LASTEXITCODE -ne 0){throw 'Unable to inspect local Unsloth installation'}
if(($installed | Out-String).Trim() -ne 'yes'){
  Write-Host 'UNSLOTH_INSTALL_BEGIN_ISOLATED_D_DRIVE'
  & $python -B -m pip install --disable-pip-version-check --no-input --timeout 60 --retries 1 unsloth
  if($LASTEXITCODE -ne 0){throw 'Unsloth installation failed (existing env untouched)'}
}
& $python -B -c "from unsloth import FastModel;import torch;print('UNSLOTH_IMPORT_OK='+str(torch.cuda.is_available()))"
if($LASTEXITCODE -ne 0){throw 'Unsloth incompatible with this Windows Python/Torch stack'}
$train=Join-Path $root 'train.jsonl'
$holdout=Join-Path $repo 'datasets\neral_qwen35_holdout_v1.jsonl'
if(-not(Test-Path -LiteralPath $train -PathType Leaf)){
  Push-Location $repo
  try {
    & $python -B -m scripts.neral_qlora_dataset --out $train --holdout $holdout --pairs 360
    if($LASTEXITCODE -ne 0){throw 'Verified dataset generation failed'}
  } finally {Pop-Location}
}
$digest=(Get-FileHash -Algorithm SHA256 -LiteralPath $train).Hash.ToLowerInvariant()
if($digest -ne '2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79'){
  throw 'Training data does not match frozen 720-row hash'
}
Push-Location $repo
try {
  & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out (Join-Path $root 'data_validation_only') --check-data-only
  if($LASTEXITCODE -ne 0){throw 'Holdout disjointness or training structure failed'}
} finally {Pop-Location}
# Running original local chat model must be unloaded to free VRAM.
$gpu=& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
if($LASTEXITCODE -ne 0){throw 'Cannot read RTX GPU'}
$values=($gpu|Select-Object -First 1) -split ','
if($values.Count -ne 2){throw 'Malformed GPU status'}
if([int]$values[0].Trim() -lt 9216){
  Write-Host 'UNLOADING_OLLAMA_MODEL_FOR_LOCAL_TRAINING'
  & ollama stop qwen3.5:9b
  if($LASTEXITCODE -ne 0){throw 'Ollama model did not unload cleanly'}
  Start-Sleep -Seconds 3
}
$gpu=& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
$values=($gpu|Select-Object -First 1) -split ','
Write-Host "GPU_FREE_MIB_AFTER_UNLOAD=$($values[0].Trim())"
if([int]$values[0].Trim() -lt 9216 -or [int]$values[1].Trim() -gt 25){
  throw 'GPU is not sufficiently free for 9B QLoRA'
}
$runOut=Join-Path $root ('adapter_'+$Phase)
if(Test-Path -LiteralPath $runOut){throw 'Refuse overwrite previous adapter'}
Write-Host "QLORA_PHASE_BEGIN=$Phase"
Push-Location $repo
try {
  if($Phase -eq 'smoke'){
    & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out $runOut --model unsloth/Qwen3.5-9B --epochs 1 --max-steps 1
  }else{
    if(-not(Test-Path -LiteralPath (Join-Path $root 'adapter_smoke\adapters\adapter_config.json'))){
      throw 'One-step GPU smoke has not passed'
    }
    $env:HF_HUB_OFFLINE='1'
    & $python -B -m scripts.neral_qlora_train --train $train --holdout $holdout --out $runOut --model unsloth/Qwen3.5-9B --epochs 1
  }
  if($LASTEXITCODE -ne 0){throw 'Local QLoRA GPU training failed'}
} finally {Pop-Location}
if(-not(Test-Path -LiteralPath (Join-Path $runOut 'adapters\adapter_config.json') -PathType Leaf)){
  throw 'No valid PEFT adapter was saved'
}
Write-Host "QLORA_PHASE_SUCCESS=$Phase"
# Never upload model checkpoint, tokenizer, train text, or HuggingFace credentials.
