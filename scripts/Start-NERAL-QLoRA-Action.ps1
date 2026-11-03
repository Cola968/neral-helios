# Restricted GitHub Actions entrypoint for three optimizer steps of rank-32 NERAL QLoRA.
# Runs only on a trusted Windows self-hosted runner, never on unreviewed PR code.
[CmdletBinding()]
param(
    [ValidateSet('preflight', 'train-three-steps')][string]$Mode = 'preflight',
    [switch]$InstallPinnedEnvironment
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$expectedSource = 'a082738f9145722dec443141e5d7801a3ead93af'
$expectedTrainerBlob = 'f4b0f72440ce40653ca0ce2b171bec6721505e18'
$work = [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE)
$control = Join-Path $work 'control'
$src = Join-Path $work 'trainer'
$python = 'C:\actions-runner\_work\_tool\Python\3.12.10\x64\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = (& py -3.12 -c 'import sys; print(sys.executable)').Trim()
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'No Python 3.12 interpreter' }
if (-not (Test-Path -LiteralPath $src -PathType Container) -or -not (Test-Path -LiteralPath $control -PathType Container)) {
    throw 'Pinned checkout or workflow control checkout missing'
}
$head = (& git -C $src rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $expectedSource) { throw 'Trainer source SHA differs from trusted pin' }
$blob = (& git -C $src hash-object 'scripts/neral_qlora_pilot.py').Trim()
if ($LASTEXITCODE -ne 0 -or $blob -ne $expectedTrainerBlob) { throw 'QLoRA trainer blob differs from trusted pin' }

# Always place outputs outside both GitHub checkout worktrees. Never reuse a run directory.
$external = Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research'
$runId = $env:GITHUB_RUN_ID
$attempt = $env:GITHUB_RUN_ATTEMPT
if ($runId -notmatch '^[0-9]{6,24}$' -or $attempt -notmatch '^[0-9]{1,3}$') { throw 'Invalid GitHub run identity' }
$runRoot = Join-Path $external "actions-r32-$runId-$attempt"
if (Test-Path -LiteralPath $runRoot) { throw 'Experiment already exists; no second run or overwrite' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$reportRoot = Join-Path $control 'neral_qlora_reports'
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$step = 'preflight'
try {
    # Prefer a previously sealed, flat HF snapshot. Do not recursively crawl C:\ or use raw HF cache.
    $candidates = @()
    if ($env:NERAL_QWEN25_HF_SNAPSHOT) { $candidates += $env:NERAL_QWEN25_HF_SNAPSHOT }
    if ($env:NERAL_ASSETS_ROOT) { $candidates += (Join-Path $env:NERAL_ASSETS_ROOT 'installed_models\hf-qwen25-coder-7b') }
    $candidates += (Join-Path $env:USERPROFILE 'Documents\Codex\2026-09-05\lo\outputs\neral\installed_models\hf-qwen25-coder-7b')
    $candidates += (Join-Path $env:USERPROFILE 'NERAL\models\qwen25-verified')
    $candidates += (Join-Path $env:USERPROFILE 'NERAL\models\qwen25-coder-7b')
    $found = @($candidates | Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ 'helios-manifest.json') -PathType Leaf) } | Select-Object -Unique)
    if ($found.Count -ne 1) { throw 'Expected exactly one known, sealed Qwen HF snapshot; no unverified directory selected' }
    $model = [IO.Path]::GetFullPath($found[0])
    $env:PYTHONPATH = $src
    $check = 'import sys;from model_catalog import validate_hf_snapshot; v=validate_hf_snapshot(sys.argv[1],verify_hashes=True);assert v["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"; print("Offline model hash validation PASSED")'
    & $python -c $check $model
    if ($LASTEXITCODE -ne 0) { throw 'Local HF snapshot hash check failed' }
    $step = 'snapshot_verified'

    $rankCopy = Join-Path $src 'scripts\neral_qlora_rank32_pilot.py'
    & $python (Join-Path $control 'scripts\neral_rank32_copy.py') --trainer (Join-Path $src 'scripts\neral_qlora_pilot.py') --output $rankCopy
    if ($LASTEXITCODE -ne 0) { throw 'Rank32 transformer refused source' }
    $dataset = Join-Path $runRoot 'reviewed_dataset'
    & $python (Join-Path $src 'scripts\neral_capability_train_pack.py') --root $src --out-dir $dataset
    if ($LASTEXITCODE -ne 0) { throw 'Dataset review or 86-row pack rejected' }
    $step = 'data_verified'

    $venv = Join-Path $external 'training-env'
    $trainPython = if ($env:NERAL_QLORA_PYTHON) { $env:NERAL_QLORA_PYTHON } else { Join-Path $venv 'Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $trainPython -PathType Leaf) -and $InstallPinnedEnvironment) {
        if (Test-Path -LiteralPath $venv) { throw 'Environment exists but lacks interpreter, refuse repairs in-place' }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $src 'scripts\setup_neral_training.ps1') -VenvDir $venv
        if ($LASTEXITCODE -ne 0) { throw 'Pinned isolated dependency installation failed' }
    }
    if (-not (Test-Path -LiteralPath $trainPython -PathType Leaf)) { throw 'Isolated training environment unavailable' }
    if ($trainPython -like "${src}*") { throw 'Training interpreter may not reside within source checkout' }
    & $trainPython (Join-Path $src 'scripts\neral_training_preflight.py') --check-cuda
    if ($LASTEXITCODE -ne 0) { throw 'CUDA BF16/NF4 training preflight failed' }
    $step = 'cuda_verified'
    for ($i=0; $i -lt 2; $i++) {
        $raw = & nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
        $parts = ($raw | Select-Object -First 1) -split ','
        if ($parts.Count -ne 2) { throw 'Invalid nvidia-smi result' }
        $free = [int]$parts[0].Trim()
        $usage = [int]$parts[1].Trim()
        if ($free -lt 10000 -or $usage -gt 15) { throw 'GPU is busy or below 10000 MiB free' }
        if ($i -eq 0) { Start-Sleep -Seconds 2 }
    }
    if ($Mode -eq 'preflight') {
        [ordered]@{schema=1;status='PREFLIGHT_PASSED';training_executed=$false;rank=32;steps=3;weight_upload=$false} |
            ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $reportRoot 'status.json') -Encoding UTF8
        Write-Host 'Verified preflight only. No training performed.'
        exit 0
    }
    if ($env:NERAL_R32_TRAIN_AUTHORIZATION -ne 'EXACTLY_THREE_OPTIMIZER_STEPS_R32') {
        throw 'Explicit one-shot workflow training authorization absent'
    }
    $step = 'training_started'
    $env:HF_HUB_OFFLINE='1'
    $env:TRANSFORMERS_OFFLINE='1'
    $env:WANDB_DISABLED='true'
    $adapter = Join-Path $runRoot 'adapter'
    & $trainPython $rankCopy --dataset (Join-Path $dataset 'neral_train_86.jsonl') `
      --review-manifest (Join-Path $dataset 'neral_train_86.review.json') `
      --model-dir $model --output-dir $adapter --adapter-version '0.1.0-pilot.1' `
      --steps 3 --lora-rank 32 --max-length 512 --run-local-gpu
    if ($LASTEXITCODE -ne 0) { throw 'QLoRA 3-step pilot failed (see local logs)' }
    & $python (Join-Path $control 'scripts\neral_qlora_action_report.py') `
      --trainer-report (Join-Path $adapter 'training-report.json') `
      --adapter (Join-Path $adapter 'adapter_model.safetensors') `
      --report-out (Join-Path $reportRoot 'summary.json')
    if ($LASTEXITCODE -ne 0) { throw 'Completed run failed scalar-only report audit' }
    $step = 'completed'
    Write-Host 'Completed: rank32, three optimizer steps. Adapter remains LOCAL, not deployed.'
} catch {
    [ordered]@{schema=1;status='REFUSED';last_phase=$step;training_verified=$false;weight_upload=$false} |
        ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $reportRoot 'status.json') -Encoding UTF8
    throw
}
