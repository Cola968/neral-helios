param(
    [switch]$CheckOnly,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
# Explicit frozen HF mode. No training, downloads, model installation,
# process termination or silent changes to the established NERAL preview.
$root = $PSScriptRoot
$py = 'D:/neral-qlora/venv/Scripts/python.exe'
$base4 = 'D:/neral-qlora-v2/qwen35-4b-hf'
$base9 = 'D:/neral-qlora/checkpoint'
$full = 'D:/neral-qlora-v2/v2full-43931510428-1-helios'
foreach($file in @(
    $py,
    (Join-Path $root 'neral_preview_server.py'),
    (Join-Path $root 'neral_three_tier_runtime.py'),
    (Join-Path $base4 'model.safetensors.index.json'),
    (Join-Path $base9 'config.json'),
    (Join-Path $full 'manifest.json'),
    (Join-Path $full 'adapter/adapter_model.safetensors')
)){
    if(-not (Test-Path -LiteralPath $file -PathType Leaf)){
        throw ('Lokale Voraussetzung fehlt: ' + $file)
    }
}
$env:HF_HOME = 'D:/neral-qlora-v2/hf-cache'
$env:HF_HUB_CACHE = 'D:/neral-qlora-v2/hf-cache/hub'
$env:HF_XET_CACHE = 'D:/neral-qlora-v2/hf-cache/xet'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = $root
# Do not install Flask/Waitress in the frozen QLoRA research venv.
$check = @'
import sys
from pathlib import Path
import torch, transformers, peft, bitsandbytes, z3, flask
from scripts.neral_three_tier_runtime_inventory import frozen_v2_full
from neral_three_tier_runtime import FrozenQwen35HF
assert sys.version_info >= (3,11)
assert frozen_v2_full(Path(sys.argv[3]))
FrozenQwen35HF(Path(sys.argv[1]))
FrozenQwen35HF(Path(sys.argv[2]),full_root=Path(sys.argv[3]))
print(1)
'@
Push-Location -LiteralPath $root
try{
    & $py -B -c $check $base4 $base9 $full
    if($LASTEXITCODE -ne 0){
        throw 'HF-GPU-Umgebung oder Modellpfade unvollstaendig. Keine Installation erfolgt.'
    }
}finally{ Pop-Location }

# Production HTTP server dependency is intentionally NOT installed in the
# frozen QLoRA / HF Python environment. Waitress has no third-party dependencies,
# so install only its standalone files into an isolated app-local directory,
# and only after explicit user consent. Never download or alter model weights.
$state = Join-Path $root 'local_state'
$webDeps = Join-Path $state 'three-tier-web-deps'
$installedPkgs = @(& $py -B -m pip list --format=freeze)
if($LASTEXITCODE -ne 0){throw 'Isolierte Python-Paketinventur fehlgeschlagen.'}
$nativeWaitress = (@($installedPkgs | Where-Object {$_ -match '^waitress==[0-9]' }).Count -gt 0)
if(-not $nativeWaitress){
    $waitressFile = Join-Path $webDeps 'waitress/__init__.py'
    if(-not (Test-Path -LiteralPath $waitressFile -PathType Leaf)){
        if($CheckOnly){
            throw 'Waitress fehlt in der HF-Umgebung. Ohne Installation starten mit Zustimmung; CheckOnly bleibt schreibgeschuetzt.'
        }
        if($env:NERAL_THREE_TIER_INSTALL_CONSENT -ne '1'){
            Write-Host 'NERAL benoetigt einmalig Waitress 3.0.2 aus PyPI (kleines Webserver-Paket). Die QLoRA-Umgebung bleibt unveraendert.' -ForegroundColor Yellow
            $consent = Read-Host 'Nur Waitress separat in local_state installieren? (J/N)'
            if($consent -notin @('J','j','Ja','ja')){
                throw 'Waitress-Installation nicht freigegeben. Keine Aenderungen vorgenommen.'
            }
        }
        if(Test-Path -LiteralPath $webDeps){
            throw 'Isolierte Waitress-Installation unvollstaendig; vorhandenen Ordner manuell pruefen.'
        }
        if(-not $CheckOnly){ New-Item -ItemType Directory -Path $state -Force | Out-Null }
        & $py -B -m pip install --disable-pip-version-check --no-input --no-cache-dir --no-deps --target $webDeps waitress==3.0.2
        if($LASTEXITCODE -ne 0){
            throw 'Isolierte Waitress-Installation fehlgeschlagen. QLoRA-Umgebung unveraendert.'
        }
    }
    $env:NERAL_THREE_TIER_WEB_DEPS = $webDeps
    $env:PYTHONPATH = $webDeps + [IO.Path]::PathSeparator + $root
    & $py -B -c 'import waitress,sys;from pathlib import Path;assert Path(waitress.__file__).resolve().is_relative_to(Path(sys.path[1]).resolve())'
    if($LASTEXITCODE -ne 0){ throw 'Isoliertes Waitress konnte nicht sicher geladen werden.' }
}
$ollama = Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue
if(-not $ollama){throw 'Bestehendes Ollama für den Legacy-Rückwechsel fehlt.'}
$names = @(& $ollama.Source list)
if($LASTEXITCODE -ne 0 -or -not (@($names | Where-Object {$_ -match '^\s*qwen3:8b\s+'}).Count -gt 0)){
    throw 'qwen3:8b nicht installiert; Legacy-Rückwechsel bleibt gesperrt. Kein Download.'
}
if($CheckOnly){
    Write-Host 'Luna 4B und Helios v2-full sind lokal verifiziert. Keine GPU-Inferenz gestartet.' -ForegroundColor Green
    return
}
$port = 9690
$socket = New-Object System.Net.Sockets.TcpClient
try{
    $socket.Connect('127.0.0.1',$port)
    throw "Port $port bereits belegt. Bestehenden NERAL-Dienst kontrolliert beenden."
}catch [System.Net.Sockets.SocketException] {
}finally{ $socket.Dispose() }
$state = Join-Path $root 'local_state'
New-Item -ItemType Directory -Path $state -Force | Out-Null
$stdout = Join-Path $state 'neral-three-tier.log'
$stderr = Join-Path $state 'neral-three-tier-error.log'
$startArgs = @('-B','-m','neral_preview_server','--model','qwen3:8b','--experimental-three-tier-hf','--luna-4b-hf',$base4,'--helios-9b-hf',$base9,'--helios-v2-full',$full)
$proc = Start-Process -FilePath $py -ArgumentList $startArgs -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$ready = $false
for($attempt=0; $attempt -lt 60; $attempt++){
    Start-Sleep -Milliseconds 300
    if($proc.HasExited){throw ('NERAL wurde beendet. Lokales Fehlerprotokoll: '+$stderr)}
    $probe = New-Object System.Net.Sockets.TcpClient
    try{ $probe.Connect('127.0.0.1',$port);$ready=$true;break }
    catch [System.Net.Sockets.SocketException] {}
    finally{ $probe.Dispose() }
}
if(-not $ready){
    try{$proc.Kill()}catch{}
    throw 'NERAL hat den lokalen Port 9690 nicht geöffnet. Fehlerprotokoll prüfen.'
}
$url = 'http://127.0.0.1:9690/'
Write-Host ("NERAL 3 Modi ist lokal erreichbar: "+$url) -ForegroundColor Green
Write-Host 'Die PEFT-Gewichte werden erst auf Anforderung geladen. Der Modus bleibt experimentell.'
if(-not $NoBrowser){Start-Process -FilePath $url}
