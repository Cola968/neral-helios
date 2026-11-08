param(
    [ValidateSet('qwen3:8b','qwen3.5:9b')]
    [string]$Model='qwen3:8b',
    [switch]$CheckOnly,
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
$preferred=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
if(Test-Path -LiteralPath $preferred -PathType Leaf){
    $py=$preferred
}else{
    $candidate=Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if(-not $candidate){throw 'Python fehlt. Python 3.12 mit Flask und Waitress wird benoetigt.'}
    $py=$candidate.Source
}
& $py -B -c 'import sys,flask,waitress; assert sys.version_info >= (3,11)'
if($LASTEXITCODE -ne 0){
    throw 'Python-Abhaengigkeiten fehlen. Einmalig bewusst installieren: python -m pip install -r requirements-neral-preview.txt'
}
$state=Join-Path $PSScriptRoot 'local_state'
New-Item -ItemType Directory -Path $state -Force | Out-Null
$err=Join-Path $state 'neral-preview-error.log'
$out=Join-Path $state 'neral-preview.log'
Write-Host "NERAL 1.0 Local Preview: $Model (local_foundation, kein freigegebener NERAL-LoRA-Adapter)." -ForegroundColor Cyan
$proc=Start-Process -FilePath $py -ArgumentList @('-B','-m','neral_preview_server','--model',$Model) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardError $err -RedirectStandardOutput $out -PassThru
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
