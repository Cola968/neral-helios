# NERAL Mini: local-only Windows start, using the existing GPU Python environment.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$program = Join-Path $root 'mini\neral_mini.py'
if (-not (Test-Path -LiteralPath $program -PathType Leaf)) {
    throw 'mini\neral_mini.py fehlt. Bitte das gesamte NERAL-Mini-ZIP entpacken.'
}
$python = $env:NERAL_MINI_PYTHON
if (-not $python) {
    $python = Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Die lokale Python-GPU-Umgebung fehlt. Setze NERAL_MINI_PYTHON auf die vorhandene python.exe mit torch, transformers, peft und bitsandbytes. Es werden keine Pakete heimlich installiert.'
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:WANDB_DISABLED = 'true'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Host ''
Write-Host 'NERAL Mini – lokaler LoRA-Chat' -ForegroundColor Cyan
Write-Host 'Qwen2.5-Coder-7B ist nur die Foundation. Aktiv ist ausschliesslich ein lokal verifizierter NERAL-LoRA-Adapter.'
Write-Host 'Keine externen KI-APIs, keine Modell-Downloads und keine Zahlung.' -ForegroundColor Gray
Write-Host 'Start nur auf diesem PC unter http://127.0.0.1:9691' -ForegroundColor Gray
Write-Host 'Beenden: in diesem Fenster Strg+C.'
Write-Host ''
& $python -B $program
if ($LASTEXITCODE -ne 0) {
    throw "NERAL Mini beendet mit Fehlercode $LASTEXITCODE. Details stehen oben."
}
