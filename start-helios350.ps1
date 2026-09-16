# Start with: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-helios350.ps1
# Optional: -GPU (only when no training is using the RTX 4070); -Smoke (one-token test).
# This script never edits any checkpoint, modifies the user PATH or installs software.
param([switch]$GPU, [switch]$Smoke)
$ErrorActionPreference = 'Stop'

function To-WslPath([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') { throw 'An absolute Windows drive path is required.' }
    if ($full.Contains("'")) { throw 'Paths containing a single quotation mark are not supported by this launcher.' }
    return '/mnt/' + $matches[1].ToLowerInvariant() + '/' + $matches[2].Replace('\','/')
}

$source = Split-Path -Parent $PSCommandPath
$program = Join-Path $source 'frontier350\nextgen_v1\helios350_preview.py'
$assets = if ($env:NERAL_ASSETS_ROOT) { $env:NERAL_ASSETS_ROOT } else { Join-Path $env:USERPROFILE 'Documents\Codex\2026-09-05\lo\outputs\neral' }
$safe = Join-Path $assets 'runs\neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected\resume.pt'
$tokenizer = Join-Path $assets 'frontier350_assets\tokenizer.json'
foreach ($file in @($program, $safe, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing local Neral file. Set NERAL_ASSETS_ROOT to the existing private assets folder: $file"
    }
}

$device = 'cpu'
if ($GPU) {
    # Inference and training must not fight for the same 12GB GPU. Do not kill apps.
    $samples = @()
    for ($i=0; $i -lt 3; $i++) {
        $readings = @(nvidia-smi --query-gpu=name,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0 -or $readings.Count -ne 1) { throw 'Cannot verify a single idle NVIDIA GPU; use CPU mode.' }
        $parts = @($readings[0] -split ',' | ForEach-Object {$_.Trim()})
        if ($parts.Count -ne 4 -or -not $parts[0].Contains('RTX 4070')) { throw 'Unexpected GPU; use CPU mode.' }
        $free = [int]$parts[1]; $temp = [int]$parts[2]; $util = [int]$parts[3]
        $samples += $util
        if ($free -lt 9000 -or $temp -gt 78 -or $util -gt 15) {
            throw "GPU is busy/not ready (free $free MiB, $temp C, $util% utilization); stop training or use CPU mode."
        }
        if ($i -lt 2) { Start-Sleep -Seconds 2 }
    }
    $device = 'cuda'
}

$flag = if ($Smoke) { ' --smoke' } else { '' }
$command = "set -euo pipefail; cd '$(To-WslPath $source)'; ~/.venvs/neral-triton-py312/bin/python -u frontier350/nextgen_v1/helios350_preview.py --assets-root '$(To-WslPath $assets)' --model safe --device $device$flag"
if (-not $Smoke) {
    Write-Host 'Helios 350M SAFE, LOCAL, read-only. Open http://127.0.0.1:9681/ in your browser after the server prints its ready message.'
    Write-Host 'Press Ctrl+C here to stop. This is an experimental foundation model, not a released assistant.'
}
& wsl.exe -- bash -lc $command
if ($LASTEXITCODE -ne 0) { throw 'Helios preview did not start successfully. Check the message above; no checkpoint was changed.' }
