param([switch]$NoBrowser,[switch]$GpuLab)
$ErrorActionPreference='Stop'

function Resolve-NeralRoot {
    $candidates=@(
        $PSScriptRoot,
        (Join-Path (Split-Path -Parent $PSScriptRoot) 'Neral-CPU-App')
    )
    $desktop=[Environment]::GetFolderPath('Desktop')
    if($desktop){$candidates += (Join-Path $desktop 'Neral-CPU-App')}
    if($env:OneDrive){$candidates += (Join-Path $env:OneDrive 'Desktop\Neral-CPU-App')}

    foreach($candidate in $candidates | Select-Object -Unique){
        if(-not $candidate){continue}
        if((Test-Path -LiteralPath (Join-Path $candidate 'app.py') -PathType Leaf) -and
           (Test-Path -LiteralPath (Join-Path $candidate 'deployment') -PathType Container)){
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Der vollstaendige Neral-CPU-App-Ordner wurde nicht gefunden. Der Starter allein enthaelt weder app.py noch das Modell. Lege den vollstaendigen Ordner Neral-CPU-App auf den Desktop und starte erneut.'
}

$appRoot=Resolve-NeralRoot
Set-Location -LiteralPath $appRoot
$state=Join-Path $appRoot 'local_state'
New-Item -ItemType Directory -Force -Path $state | Out-Null

function Test-Python311 {
    param([string]$Exe,[string[]]$Prefix=@())
    if(-not $Exe -or -not(Test-Path -LiteralPath $Exe -PathType Leaf)){return $false}
    try{
        & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }catch{return $false}
}

function Find-SystemPython {
    $known=@(
        $env:NERAL_PYTHON,
        (Join-Path $state 'bootstrap-python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
    )
    # A Python installer can enter maintenance mode and leave the interpreter
    # in its registered original directory, even with a different TargetDir.
    foreach($registryRoot in @('HKCU:\Software\Python\PythonCore','HKLM:\Software\Python\PythonCore','HKLM:\Software\WOW6432Node\Python\PythonCore')){
        foreach($versionKey in Get-ChildItem -LiteralPath $registryRoot -ErrorAction SilentlyContinue){
            $install=Get-ItemProperty -LiteralPath (Join-Path $versionKey.PSPath 'InstallPath') -ErrorAction SilentlyContinue
            if($install.ExecutablePath){$known += $install.ExecutablePath}
            if($install.'(default)'){$known += (Join-Path $install.'(default)' 'python.exe')}
        }
    }
    # Also recognize previous Neral packages when registration is unavailable.
    $desktops=@([Environment]::GetFolderPath('Desktop'))
    if($env:OneDrive){$desktops += (Join-Path $env:OneDrive 'Desktop')}
    foreach($desktop in $desktops | Select-Object -Unique){
        if($desktop){$known += (Join-Path $desktop 'Neral-CPU-App\local_state\bootstrap-python312\python.exe')}
    }
    foreach($exe in $known){if(Test-Python311 $exe){return [pscustomobject]@{Exe=$exe;Prefix=@()}}}

    foreach($candidate in @(
        @{Command='py';Prefix=@('-3')},
        @{Command='python';Prefix=@()},
        @{Command='python3';Prefix=@()}
    )){
        $cmd=Get-Command $candidate.Command -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if($cmd -and (Test-Python311 $cmd.Source $candidate.Prefix)){
            return [pscustomobject]@{Exe=$cmd.Source;Prefix=@($candidate.Prefix)}
        }
    }
    return $null
}

function Install-NeralPython {
    Write-Host 'Python fehlt. Neral richtet Python 3.12 automatisch ein. Das passiert nur beim ersten Start.' -ForegroundColor Yellow
    $bootstrapRoot=Join-Path $state 'bootstrap-python312'
    $pythonExe=Join-Path $bootstrapRoot 'python.exe'
    if(Test-Python311 $pythonExe){return [pscustomobject]@{Exe=$pythonExe;Prefix=@()}}

    New-Item -ItemType Directory -Force -Path $bootstrapRoot | Out-Null
    $installer=Join-Path $state 'python-3.12.10-amd64.exe'
    $url='https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
    try{
        [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    }catch{
        throw 'Python konnte nicht automatisch von python.org geladen werden. Pruefe die Internetverbindung und starte Neral erneut.'
    }

    if((Get-AuthenticodeSignature -LiteralPath $installer).Status -ne 'Valid'){throw 'Die Signatur des Python-Installers ist ungueltig. Installation abgebrochen.'}
    $installArgs=@(
        '/quiet','InstallAllUsers=0',('TargetDir="'+$bootstrapRoot+'"'),'Include_pip=1','Include_launcher=0',
        'Include_test=0','Include_doc=0','Include_tcltk=0','Include_dev=0','Shortcuts=0',
        'AssociateFiles=0','PrependPath=0','SimpleInstall=1','CompileAll=0'
    )
    $process=Start-Process -FilePath $installer -ArgumentList $installArgs -WindowStyle Hidden -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    if($process.ExitCode -notin @(0,3010)){
        throw "Die automatische Python-Installation ist fehlgeschlagen (Code $($process.ExitCode))."
    }
    $installed=Find-SystemPython
    if($installed){return $installed}
    throw 'Python-Installer beendet, aber kein startbares Python gefunden. Die vorhandene Python-Installation muss repariert werden; Details stehen im Python-Installationsprotokoll unter %TEMP%.'
}

# Optional GPU assistant without changing the default CPU-only NERAL app.
# This uses only the existing offline QLoRA Python environment and sealed v5 files.
if($GpuLab){
    $env:NERAL_AGENT_BACKEND='hf_adapter'
    $env:NERAL_AGENT_ADAPTER_CHANNEL='v5'
    if(-not $env:NERAL_PYTHON){
        $env:NERAL_PYTHON=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
    }
    if(-not(Test-Python311 $env:NERAL_PYTHON)){
        throw 'NERAL v5 GPU-Lab braucht die bereits vorhandene QLoRA-Python-Umgebung mit CUDA, Transformers, PEFT und bitsandbytes. Keine automatische Installation.'
    }
    $v5Adapter=Join-Path $env:USERPROFILE 'NERAL-QLoRA-Research\v5-r16-480-42378021782-1\adapter'
    if(-not(Test-Path -LiteralPath (Join-Path $v5Adapter 'adapter_model.safetensors') -PathType Leaf) -or
       -not(Test-Path -LiteralPath (Join-Path $v5Adapter 'neral-adapter.json') -PathType Leaf)){
        throw 'NERAL v5 Adaptergewichte fehlen lokal. Keine Cloud- oder Basismodell-Ersatzantwort.'
    }
}

$pythonCandidates=@(
    $env:NERAL_PYTHON,
    (Join-Path $appRoot '.venv\Scripts\python.exe'),
    (Join-Path $state 'python\Scripts\python.exe')
)
$appPython=$pythonCandidates | Where-Object { $_ -and (Test-Python311 $_) } | Select-Object -First 1

if(-not $appPython){
    $systemPython=Find-SystemPython
    if(-not $systemPython){$systemPython=Install-NeralPython}

    $venvPath=Join-Path $state 'python'
    if(Test-Path -LiteralPath $venvPath){throw 'Vorhandene Python-Umgebung ist defekt. Bitte den App-Ordner erneut entpacken; vorhandene Daten bleiben erhalten.'}
    $venvArgs=@($systemPython.Prefix)+@('-m','venv',$venvPath)
    & $systemPython.Exe @venvArgs
    if($LASTEXITCODE -ne 0){throw 'Die lokale Python-Umgebung fuer Neral konnte nicht erstellt werden.'}

    $appPython=Join-Path $venvPath 'Scripts\python.exe'
    if(-not(Test-Python311 $appPython)){throw 'Die lokale Python-Umgebung wurde erstellt, aber python.exe fehlt.'}

    Write-Host 'Neral richtet beim ersten Start die CPU-Abhaengigkeiten ein. Das kann einige Minuten dauern.' -ForegroundColor Yellow
    & $appPython -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
    if($LASTEXITCODE -ne 0){throw 'CPU-PyTorch konnte nicht installiert werden. Pruefe die Internetverbindung.'}
    & $appPython -m pip install --disable-pip-version-check flask waitress numpy cryptography
    if($LASTEXITCODE -ne 0){throw 'App-Abhaengigkeiten konnten nicht installiert werden.'}
}

$env:PYTHONDONTWRITEBYTECODE='1'
$env:HELIOS_DEVICE='cpu'
if($GpuLab){
    # Keep the legacy Helios checkpoint on CPU but allow the HF agent to use CUDA.
    $env:CUDA_VISIBLE_DEVICES='0'
}else{
    $env:CUDA_VISIBLE_DEVICES=''
}
$env:HELIOS_STATE_ROOT=$state
$env:HELIOS_PREPARATION_MODE='0'
$env:HELIOS_CHECKPOINT='neral50m_i1'
$env:HELIOS_PORT='9686'
$env:HELIOS_ASSETS_ROOT=Join-Path $appRoot 'deployment'

$url='http://127.0.0.1:9686/'
function Get-NeralInfo {
    $null=Invoke-WebRequest $url -UseBasicParsing -SessionVariable appSession -TimeoutSec 2
    Invoke-RestMethod ($url+'api/info') -WebSession $appSession -TimeoutSec 2
}
$ready=$false
try{
    $info=Get-NeralInfo
    $ready=($info.app_version -eq '2.0' -and $info.active_checkpoint -eq 'neral50m_i1' -and $info.checkpoint_loaded -and $info.device -eq 'cpu' -and -not $info.preparation_mode)
}catch{}

if($ready -and $GpuLab){
    throw 'NERAL ist bereits gestartet. Die bestehende App beenden, bevor das v5 GPU-Lab mit anderem Adapter gestartet wird.'
}

if(-not $ready){
    Start-Process -FilePath $appPython -ArgumentList @('-X','utf8','app.py') -WorkingDirectory $appRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $state 'app.log') -RedirectStandardError (Join-Path $state 'app-error.log') | Out-Null
    for($attempt=0;$attempt -lt 45;$attempt++){
        Start-Sleep -Seconds 1
        try{
            $info=Get-NeralInfo
            if($info.checkpoint_loaded -and $info.device -eq 'cpu' -and $info.app_version -eq '2.0' -and $info.active_checkpoint -eq 'neral50m_i1' -and -not $info.preparation_mode){$ready=$true;break}
        }catch{}
    }
}

if(-not $ready){throw 'Neral konnte nicht gestartet werden. Siehe local_state\app-error.log. Port 9686 muss frei sein.'}

$edge=Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
if(-not $NoBrowser){
    if(Test-Path -LiteralPath $edge){Start-Process -FilePath $edge -ArgumentList @('--disable-gpu',"--app=$url")}else{Start-Process $url}
}
if($GpuLab){
    Write-Host 'NERAL v5 GPU-Lab gestartet: Helios bleibt CPU, der lokale 7B-Adapter darf CUDA nutzen. Kein Training, keine automatische Promotion.' -ForegroundColor Green
}else{
    Write-Host 'Neral laeuft. Ab jetzt reicht fuer jeden Start ein Doppelklick auf Start Neral.cmd.' -ForegroundColor Green
}
