param(
    [ValidateSet('qwen3:8b','qwen3.5:9b')]
    [string]$Model='qwen3:8b',
    [switch]$CheckOnly,
    [switch]$NoBrowser
)
$ErrorActionPreference='Stop'

# Release preview: deliberately use a declared local_foundation, never claim
# an experimental adapter passed the still-pending independent release gate.
$main=Join-Path $PSScriptRoot 'Start-Neral.ps1'
if(-not(Test-Path -LiteralPath $main -PathType Leaf)){
    throw 'Komplette Helios-App fehlt: Start-Neral.ps1 im Projektordner erforderlich.'
}
$ollama=Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue
if(-not $ollama){
    throw 'Ollama ist nicht installiert oder nicht in PATH. Keine Modell-Downloads gestartet.'
}
$listing=@(& $ollama.Source list)
if($LASTEXITCODE -ne 0 -or $listing.Count -lt 2){
    throw 'Der lokale Ollama-Dienst ist nicht bereit. Ollama starten und erneut versuchen.'
}
$installed=@()
foreach($line in $listing|Select-Object -Skip 1){
    $text=[string]$line
    if($text -match '^\s*([^\s]+)\s+'){
        $installed+= $Matches[1]
    }
}
if($Model -notin $installed){
    throw "Das bereits installierte lokale Modell '$Model' wurde nicht gefunden. Bitte 'ollama list' prüfen. Kein automatischer Download."
}
if($CheckOnly){
    Write-Host "NERAL 1.0 Local Preview: lokales Modell $Model installiert; Start nicht ausgeführt."
    return
}

# The existing Helios app may have been launched with a different backend.
# Never reuse it silently while claiming Qwen3/3.5 is selected.
$socket=New-Object System.Net.Sockets.TcpClient
try {
    $socket.Connect('127.0.0.1',9686)
    throw 'Helios läuft bereits auf Port 9686. Bitte die bestehende App vollständig beenden und erneut starten.'
} catch [System.Net.Sockets.SocketException] {
    # Connection refused: port is not occupied; proceed.
} finally {
    $socket.Dispose()
}

$env:NERAL_AGENT_BACKEND='ollama'
$env:NERAL_AGENT_OLLAMA_MODEL=$Model
Remove-Item Env:NERAL_AGENT_ADAPTER_CHANNEL -ErrorAction SilentlyContinue

Write-Host "NERAL 1.0 Local Preview: $Model (local_foundation, kein freigegebener NERAL-LoRA-Adapter)." -ForegroundColor Cyan
Write-Host 'Nach dem Start oben auf NERAL AI klicken. Der alte 50M-Hauptchat bleibt separat.' -ForegroundColor Cyan
& $main -NoBrowser:$NoBrowser
if(-not $?){
    throw 'Der lokale NERAL/Helios-Start ist fehlgeschlagen. Bitte local_state/app-error.log prüfen.'
}
