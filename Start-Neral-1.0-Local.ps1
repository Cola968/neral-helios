param(
    [ValidateSet('qwen3:8b','qwen3.5:9b')]
    [string]$Model='qwen3:8b',
    [switch]$ExperimentalSampling,
    [switch]$ExperimentalHighQwen35,
    [switch]$CheckOnly,
    [switch]$PrepareOnly,
    [switch]$NoBrowser
)
$ErrorActionPreference='Stop'

# Standalone 1.0 Local Preview; no dependency on historic 50M deployment.
$server=Join-Path $PSScriptRoot 'neral_preview_server.py'
if(-not(Test-Path -LiteralPath $server -PathType Leaf)){
    throw 'NERAL 1.0 Local Preview unvollstaendig: neral_preview_server.py fehlt.'
}
$ollama=Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue
if(-not $ollama){throw 'Ollama fehlt. Bitte eine vorhandene Ollama-Installation verwenden.'}
$listing=@(& $ollama.Source list)
if($LASTEXITCODE -ne 0 -or $listing.Count -lt 2){
    throw 'Lokaler Ollama-Dienst nicht bereit. Ollama starten. Keine Modelldownloads.'
}
$installed=@()
foreach($line in $listing | Select-Object -Skip 1){
    if(([string]$line) -match '^\s*([^\s]+)\s+'){
        $installed += $Matches[1]
    }
}
if($Model -notin $installed){
    throw "Lokales Modell $Model ist nicht installiert. ollama list pruefen; kein automatischer Download."
}
if($ExperimentalHighQwen35){
    if($Model -ne 'qwen3:8b'){
        throw 'Experimentelles Qwen3.5-HIGH erfordert qwen3:8b als LOW. Kein stiller Modellwechsel.'
    }
    if('qwen3.5:9b' -notin $installed){
        throw 'Qwen3.5:9b nicht installiert. Experimentelles HIGH bleibt aus; kein automatischer Download.'
    }
}
if($CheckOnly){
    Write-Host "NERAL 1.0: $Model verfuegbar, lokale Vorpruefung bestanden. Kein Release-Gate."
    return
}
$port=9690
$socket=New-Object System.Net.Sockets.TcpClient
try {
    $socket.Connect('127.0.0.1',$port)
    throw "Port $port ist bereits belegt. Bestehende NERAL-Instanz kontrolliert beenden; kein stilles Wiederverwenden."
} catch [System.Net.Sockets.SocketException] {
    # No listener; safe to start our new instance.
} finally {
    $socket.Dispose()
}
# Isolation: use a dedicated preview venv. Never install Flask/Waitress
# into the local QLoRA training environment or the system Python.
$state=Join-Path $PSScriptRoot 'local_state'
New-Item -ItemType Directory -Path $state -Force | Out-Null
$venv=Join-Path $state 'preview-venv'
$py=Join-Path $venv 'Scripts\python.exe'
if(-not(Test-Path -LiteralPath $py -PathType Leaf)){
    if(Test-Path -LiteralPath $venv){
        throw 'Die lokale Preview-Python-Umgebung ist unvollstaendig. Ordner local_state/preview-venv nach Backup pruefen.'
    }
    $base=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
    if(-not(Test-Path -LiteralPath $base -PathType Leaf)){
        $candidate=Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if(-not $candidate){throw 'Python 3.11+ fehlt. Bitte Python installieren.'}
        $base=$candidate.Source
    }
    if($env:NERAL_PREVIEW_INSTALL_CONSENT -ne '1'){
        Write-Host 'NERAL benoetigt einmalig Flask und Waitress aus PyPI (Internet). Keine Modelle werden heruntergeladen.' -ForegroundColor Yellow
        $consent=Read-Host 'Isolierte Python-Webserverumgebung anlegen und Pakete installieren? (J/N)'
        if($consent -notin @('J','j','Ja','ja')){
            throw 'Installation nicht freigegeben; vorhandene Trainingsumgebung bleibt unveraendert.'
        }
    }
    & $base -B -m venv $venv
    if($LASTEXITCODE -ne 0){throw 'Isolierte Preview-Python-Umgebung konnte nicht erstellt werden.'}
    & $py -B -m pip install --disable-pip-version-check --no-input --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements-neral-preview.txt')
    if($LASTEXITCODE -ne 0){throw 'Installation der Webserverpakete fehlgeschlagen. Keine Aenderung an Trainingsgewichten.'}
}
# Existing preview-venv installations may predate image/PDF attachments.
# Check the FULL declared runtime, never modify training or system Python.
& $py -B -c 'import sys,flask,waitress,pypdf; assert sys.version_info >= (3,11)'
if($LASTEXITCODE -ne 0){
    if($env:NERAL_PREVIEW_INSTALL_CONSENT -ne '1'){
        Write-Host 'NERAL Preview benoetigt aktualisierte Python-Pakete (inklusive pypdf) in der isolierten preview-venv.' -ForegroundColor Yellow
        $repairConsent=Read-Host 'Fehlende Preview-Pakete nach Einwilligung in dieser Umgebung installieren? (J/N)'
        if($repairConsent -notin @('J','j','Ja','ja')){
            throw 'Keine Aenderung ohne Zustimmung. Bestehende Preview- und Trainingsumgebungen bleiben unveraendert.'
        }
    }
    & $py -B -m pip install --disable-pip-version-check --no-input --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements-neral-preview.txt')
    if($LASTEXITCODE -ne 0){throw 'Aktualisierung der isolierten Preview-Pakete fehlgeschlagen.'}
    & $py -B -c 'import sys,flask,waitress,pypdf; assert sys.version_info >= (3,11)'
    if($LASTEXITCODE -ne 0){throw 'Preview-Python bleibt unvollstaendig. Nur local_state/preview-venv pruefen.'}
}
if($PrepareOnly){
    Write-Host 'NERAL Preview: isolierte Python-Abhaengigkeiten sind installiert. Kein Server gestartet.'
    return
}
$err=Join-Path $state 'neral-preview-error.log'
$out=Join-Path $state 'neral-preview.log'
$profile=if($ExperimentalSampling){'measured_nonthinking_2048'}else{'legacy'}
$highLabel=if($ExperimentalHighQwen35){'qwen3.5:9b (EXPERIMENTELL)'}else{$Model}
Write-Host "NERAL 1.0 Local Preview: LOW=$Model HIGH=$highLabel, Profil $profile (local_foundation, kein freigegebener NERAL-LoRA-Adapter)." -ForegroundColor Cyan
$startArgs=@('-B','-m','neral_preview_server','--model',$Model,'--inference-profile',$profile)
if($ExperimentalHighQwen35){$startArgs+=@('--experimental-high-qwen35')}
$proc=Start-Process -FilePath $py -ArgumentList $startArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardError $err -RedirectStandardOutput $out -PassThru
$ready=$false
for($attempt=0;$attempt -lt 60;$attempt++){
    Start-Sleep -Milliseconds 300
    if($proc.HasExited){throw "NERAL Preview ist fehlgeschlagen. Details: $err"}
    $probe=New-Object System.Net.Sockets.TcpClient
    try{
        $probe.Connect('127.0.0.1',$port)
        $ready=$true
        break
    }catch [System.Net.Sockets.SocketException] {
    }finally{
        $probe.Dispose()
    }
}
if(-not $ready){
    try{$proc.Kill()}catch{}
    throw "NERAL reagiert auf Port $port nicht. Details: $err"
}
$url="http://127.0.0.1:$port/"
Write-Host "NERAL ist lokal erreichbar: $url" -ForegroundColor Green
Write-Host 'Eigenes Ollama-Modell ohne Cloud; Modellqualitaet noch nicht releasevalidiert.'
if(-not $NoBrowser){Start-Process -FilePath $url}
