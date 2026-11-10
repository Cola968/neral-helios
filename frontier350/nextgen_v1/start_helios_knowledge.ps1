# Opt-in Helios Knowledge CPU preview. Wikipedia is contacted only when Knowledge Mode is checked in the UI.
param([ValidateRange(1024,65535)][int]$Port = 9683)
$ErrorActionPreference = 'Stop'
function To-Wsl([string]$InputPath) {
    $full = [IO.Path]::GetFullPath($InputPath)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') { throw 'Only absolute local Windows drive paths supported' }
    return '/mnt/' + $matches[1].ToLowerInvariant() + '/' + $matches[2].Replace('\','/')
}
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$assets = if ($env:NERAL_ASSETS_ROOT) { $env:NERAL_ASSETS_ROOT } else { Join-Path $env:USERPROFILE 'Documents\Codex\2026-09-05\lo\outputs\neral' }
$checkpoint = 'D:\NeralScaleV81\_continuity_v5\continuity_002073161216_f7a73e283421.pt'
$code = Join-Path $repo 'frontier350\nextgen_v1\helios_knowledge_preview.py'
if (-not (Test-Path -LiteralPath $code -PathType Leaf)) { throw 'Knowledge preview source not found in this checkout' }
if (-not (Test-Path -LiteralPath $assets -PathType Container)) { throw 'Local Neral tokenizer asset root missing' }
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw 'Pinned 2.073B checkpoint missing' }
if ((Get-PSDrive -Name C).Free -lt 30GB -or (Get-PSDrive -Name D).Free -lt 200GB) { throw 'Disk reserve not satisfied' }
$sha = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sha -ne 'f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7') { throw 'Own checkpoint SHA-256 mismatch' }
foreach ($p in @($repo, $assets, $checkpoint)) { if ($p.Contains("'")) { throw 'Unsupported apostrophe in shell path' } }
$cmd = "set -euo pipefail; export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1; cd '$(To-Wsl $repo)'; ~/.venvs/neral-triton-py312/bin/python -u frontier350/nextgen_v1/helios_knowledge_preview.py --execute --trusted-local-checkpoint --assets-root '$(To-Wsl $assets)' --checkpoint '$(To-Wsl $checkpoint)' --port $Port"
Write-Host "Starting Helios Knowledge on CPU. Open http://127.0.0.1:$Port/ when ready. Knowledge Mode sends ONLY the entered question to Wikipedia when enabled. Ctrl+C stops it."
& wsl.exe -- bash -lc $cmd
if ($LASTEXITCODE -ne 0) { throw 'Helios Knowledge preview stopped with an error' }
