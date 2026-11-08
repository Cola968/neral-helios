param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot

# No installation at normal NERAL startup and no changes to the GPU venv.
$py=Join-Path $PSScriptRoot 'local_state\preview-venv\Scripts\python.exe'
$installer=Join-Path $PSScriptRoot 'scripts\neral_local_voice_install.py'
if(-not(Test-Path -LiteralPath $py -PathType Leaf)){
    throw 'Zuerst Start NERAL 1.0 Local.cmd starten, damit die isolierte Preview-Pythonumgebung angelegt wird.'
}
if(-not(Test-Path -LiteralPath $installer -PathType Leaf)){
    throw 'Unvollstaendige NERAL Sprachinstallation: scripts/neral_local_voice_install.py fehlt.'
}
if($CheckOnly){
    & $py -B -c 'from neral_local_voice import LocalVoiceService; print(LocalVoiceService().status())'
    return
}
Write-Host 'NERAL Lokaler Sprachmodus: CPU-Spracherkennung + deutsche Offline-Stimme' -ForegroundColor Cyan
Write-Host 'Download NUR nach deiner Zustimmung: Faster-Whisper Basis ca. 150 MB, Piper Deutsch ca. 63 MB; zusaetzlich Python-Pakete.'
Write-Host 'Datenquellen: Hugging Face Systran/faster-whisper-base, rhasspy/piper-voices, PyPI.'
Write-Host 'Es werden KEINE Trainingsgewichte ersetzt; keine kostenpflichtige API und kein Cloud-Sprachdienst waehrend des Betriebs.'
$consent=Read-Host 'Lokale Sprachpakete installieren und Modell-Dateien herunterladen? (J/N)'
if($consent -notin @('J','j','Ja','ja')){
    Write-Host 'Nichts installiert.'
    return
}
& $py -B -m pip install --disable-pip-version-check --no-input -r (Join-Path $PSScriptRoot 'requirements-neral-local-voice.txt')
if($LASTEXITCODE -ne 0){throw 'Installation der lokalen Sprachpakete fehlgeschlagen.'}
$voiceRoot=Join-Path $PSScriptRoot 'local_state\voice'
& $py -B $installer --root $voiceRoot
if($LASTEXITCODE -ne 0){throw 'Sprachmodelle konnten nicht sicher heruntergeladen/verifiziert werden.'}
# Check actual native CPU engines, not just whether their model files exist.
# This can catch Piper phonemizer incompatibility and Whisper DLL failures
# that the superficial status endpoint could not detect.
$selftest=Join-Path $PSScriptRoot 'scripts\neral_local_voice_selftest.py'
if(-not(Test-Path -LiteralPath $selftest -PathType Leaf)){
    throw 'Offline-Sprachinstallation unvollstaendig: Sprach-Selbsttest fehlt.'
}
Write-Host 'Pruefe jetzt ECHTE lokale Whisper- und Piper-Audioverarbeitung ...'
# Run as module from the NERAL root so neral_local_voice can import.
& $py -B -m scripts.neral_local_voice_selftest
if($LASTEXITCODE -ne 0){
    throw 'Offline-Sprachmodus ist nicht funktionsfaehig. Python-Paket/Modell/Sprachstimme pruefen; kein Erfolg vorgetaeuscht.'
}
Write-Host 'NERAL-Sprachmodus geprueft. NERAL vollstaendig schliessen und neu starten; dann Lokaler Sprachmodus anklicken.' -ForegroundColor Green
