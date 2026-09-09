$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot

$state=Join-Path $PSScriptRoot 'local_state'
New-Item -ItemType Directory -Force -Path $state | Out-Null

function Find-SystemPython {
    $candidates=@(
        @{Command='py';Prefix=@('-3')},
        @{Command='python';Prefix=@()},
        @{Command='python3';Prefix=@()}
    )

    foreach($candidate in $candidates){
        $cmd=Get-Command $candidate.Command -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if(-not $cmd){continue}

        $prefix=@($candidate.Prefix)
        try{
            & $cmd.Source @prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
            if($LASTEXITCODE -eq 0){
                return [pscustomobject]@{
                    Exe=$cmd.Source
                    Prefix=$prefix
                }
            }
        }catch{}
    }

    return $null
}

$pythonCandidates=@(
    $env:NERAL_PYTHON,
    (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    (Join-Path $PSScriptRoot '../neral/.venv/Scripts/python.exe'),
    (Join-Path $state 'python/Scripts/python.exe')
)
$appPython=$pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1

if(-not $appPython){
    $systemPython=Find-SystemPython
    if(-not $systemPython){
        throw 'Python 3.11 oder neuer wurde nicht gefunden. Installiere Python von python.org mit aktivierter PATH-Option und starte Neral erneut.'
    }

    $venvPath=Join-Path $state 'python'
    $venvArgs=@($systemPython.Prefix)+@('-m','venv',$venvPath)
    & $systemPython.Exe @venvArgs
    if($LASTEXITCODE -ne 0){throw 'Die lokale Python-Umgebung fuer Neral konnte nicht erstellt werden.'}

    $appPython=Join-Path $venvPath 'Scripts/python.exe'
    if(-not(Test-Path -LiteralPath $appPython -PathType Leaf)){
        throw 'Die lokale Python-Umgebung wurde erstellt, aber python.exe fehlt.'
    }

    & $appPython -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
    if($LASTEXITCODE -ne 0){throw 'CPU-PyTorch konnte nicht installiert werden.'}

    & $appPython -m pip install --disable-pip-version-check flask waitress numpy cryptography
    if($LASTEXITCODE -ne 0){throw 'App-Abhaengigkeiten konnten nicht installiert werden.'}
}

$env:PYTHONDONTWRITEBYTECODE='1'
$env:HELIOS_DEVICE='cpu'
$env:HELIOS_PREPARATION_MODE='0'
$env:HELIOS_CHECKPOINT='neral50m_i1'
$env:HELIOS_PORT='9684'
$env:HELIOS_ASSETS_ROOT=Join-Path $PSScriptRoot 'deployment'

$url='http://127.0.0.1:9684/'
$ready=$false
try{
    $info=Invoke-RestMethod ($url+'api/info') -TimeoutSec 2
    $ready=($info.active_checkpoint -eq 'neral50m_i1' -and $info.checkpoint_loaded -and $info.device -eq 'cpu' -and -not $info.preparation_mode)
}catch{}

if(-not $ready){
    Start-Process -FilePath $appPython -ArgumentList @('-X','utf8','app.py') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $state 'app.log') -RedirectStandardError (Join-Path $state 'app-error.log') | Out-Null
    for($attempt=0;$attempt -lt 30;$attempt++){
        Start-Sleep -Seconds 1
        try{
            $info=Invoke-RestMethod ($url+'api/info') -TimeoutSec 2
            if($info.checkpoint_loaded -and $info.device -eq 'cpu' -and $info.active_checkpoint -eq 'neral50m_i1' -and -not $info.preparation_mode){
                $ready=$true
                break
            }
        }catch{}
    }
}

if(-not $ready){throw 'Neral konnte nicht gestartet werden. Siehe local_state/app-error.log. Port 9684 muss frei sein.'}

$edge=Join-Path ${env:ProgramFiles(x86)} 'Microsoft/Edge/Application/msedge.exe'
if(Test-Path -LiteralPath $edge){
    Start-Process -FilePath $edge -ArgumentList @('--disable-gpu',"--app=$url")
}else{
    Start-Process $url
}

Write-Host 'Neral laeuft auf der CPU. Fuer dauerhafte Installation im App-Menue App installieren waehlen.'
