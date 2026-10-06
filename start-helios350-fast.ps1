# Opt-in faster CPU preview. No installs, checkpoint writes, GPU or training.
# Start with: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-helios350-fast.ps1
param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
function W([string]$p) {
    $full = [IO.Path]::GetFullPath($p)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') { throw 'Absolute Windows drive path required' }
    if ($full.Contains("'")) { throw 'Single quote in paths not supported' }
    return '/mnt/' + $matches[1].ToLowerInvariant() + '/' + $matches[2].Replace('\','/')
}
$source = Split-Path -Parent $PSCommandPath
$program = Join-Path $source 'frontier350\nextgen_v1\helios350_fast_preview.py'
$assets = if ($env:NERAL_ASSETS_ROOT) { $env:NERAL_ASSETS_ROOT } else { Join-Path $env:USERPROFILE 'Documents\Codex\2026-09-05\lo\outputs\neral' }
$safe = Join-Path $assets 'runs\neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected\resume.pt'
$tokenizer = Join-Path $assets 'frontier350_assets\tokenizer.json'
foreach ($item in @($program, $safe, $tokenizer)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw 'Missing source or own SAFE assets; set NERAL_ASSETS_ROOT if needed' }
}
$flag = if ($Smoke) { ' --smoke' } else { '' }
$cmd = "set -euo pipefail; cd '$(W $source)'; ~/.venvs/neral-triton-py312/bin/python -u frontier350/nextgen_v1/helios350_fast_preview.py --assets-root '$(W $assets)' --model safe --device cpu$flag"
if (-not $Smoke) { Write-Host 'Fast experimental 350M CPU preview: http://127.0.0.1:9681/ ; Ctrl+C stops it.' }
& wsl.exe -- bash -lc $cmd
if ($LASTEXITCODE -ne 0) { throw 'Fast preview failed; the original preview and checkpoint remain unchanged.' }
