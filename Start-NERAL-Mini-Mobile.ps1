# NERAL Mini Mobile - an encrypted, paired WLAN companion to the local GPU model.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'mini\mobile_gateway.py') -PathType Leaf)) {
    throw 'Bitte das komplette NERAL-Mini-Mobile-ZIP entpacken.'
}
$python = $env:NERAL_MINI_PYTHON
if (-not $python) {
    $python = Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Lokale NERAL-GPU-Python-Umgebung fehlt. NERAL_MINI_PYTHON auf python.exe mit torch, peft, transformers, bitsandbytes und cryptography setzen.'
}
& $python -B -c 'import cryptography, torch, transformers, peft, bitsandbytes; print("GPU-Python-Abhaengigkeiten vorhanden.")'
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Wenn nur cryptography fehlt, im PowerShell-Fenster ausfuehren:' -ForegroundColor Yellow
    Write-Host ('& "{0}" -m pip install cryptography' -f $python)
    throw 'Lokale Python-Abhaengigkeit fehlt. Keine automatische Installation erfolgt.'
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:WANDB_DISABLED = 'true'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Host ''
Write-Host 'NERAL Mini Mobile - privater HTTPS-Zugang' -ForegroundColor Cyan
Write-Host 'Das Modell laeuft auf diesem PC. Handy und PC muessen im selben privaten WLAN sein.'
Write-Host 'Das Zertifikat und den Kopplungscode findest du in local_state\mini_mobile\' -ForegroundColor Yellow
Write-Host 'Android: Neral-Android-CA.crt vertrauenswuerdig uebertragen, in Android installieren, HTTPS-Adresse aus HANDY-ZUGANG.txt oeffnen.'
Write-Host 'Keine Router-Portfreigabe; Firewallfreigabe nur fuer das private WLAN und TCP 9444.' -ForegroundColor Yellow
Write-Host 'Wenn sich das Terminal schliesst, endet der Handyzugang. Beenden mit Strg+C.'
Write-Host ''
& $python -B -m mini.mobile_gateway
if ($LASTEXITCODE -ne 0) {throw "NERAL Mini Mobile wurde mit Fehlercode $LASTEXITCODE beendet."}
