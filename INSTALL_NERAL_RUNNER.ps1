[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/Cola968/neral-helios'
$SetupUrl = "$RepoUrl/settings/actions/runners/new"
$InstallDir = 'C:\actions-runner'
$TaskName = 'Neral GitHub Actions Runner'
$RunnerName = "$env:COMPUTERNAME-Neral"
$RunnerLabel = 'neral-gpu'

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Read-SecretText([string]$Prompt) {
    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Resolve-UserEnvironment {
    $assets = [Environment]::GetEnvironmentVariable('NERAL_ASSETS_ROOT', 'User')
    $python = [Environment]::GetEnvironmentVariable('NERAL_PYTHON', 'User')

    if (-not $assets -and (Test-Path 'C:\NeralPrivate' -PathType Container)) {
        $assets = 'C:\NeralPrivate'
    }
    if (-not $python) {
        $candidates = @(
            'C:\NeralPrivate\.venv\Scripts\python.exe',
            (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
        )
        $python = $candidates | Where-Object { Test-Path $_ -PathType Leaf } | Select-Object -First 1
    }

    if (-not $assets) {
        Write-Host ''
        Write-Host 'NERAL_ASSETS_ROOT was not found.' -ForegroundColor Yellow
        $entered = Read-Host 'Path to the private Neral assets folder (Enter to skip for now)'
        if ($entered) {
            if (-not (Test-Path $entered -PathType Container)) { throw "Assets folder does not exist: $entered" }
            $assets = (Resolve-Path $entered).Path
        }
    }
    if (-not $python) {
        Write-Host ''
        Write-Host 'NERAL_PYTHON was not found.' -ForegroundColor Yellow
        $entered = Read-Host 'Path to the CUDA-enabled python.exe used by Neral (Enter to skip for now)'
        if ($entered) {
            if (-not (Test-Path $entered -PathType Leaf)) { throw "python.exe does not exist: $entered" }
            $python = (Resolve-Path $entered).Path
        }
    }

    if ($assets) {
        [Environment]::SetEnvironmentVariable('NERAL_ASSETS_ROOT', $assets, 'User')
        $env:NERAL_ASSETS_ROOT = $assets
        Write-Host "NERAL_ASSETS_ROOT = $assets" -ForegroundColor Green
    }
    if ($python) {
        [Environment]::SetEnvironmentVariable('NERAL_PYTHON', $python, 'User')
        $env:NERAL_PYTHON = $python
        Write-Host "NERAL_PYTHON = $python" -ForegroundColor Green
    }

    return [pscustomobject]@{ Assets = $assets; Python = $python }
}

function Install-RunnerFiles {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    if (Test-Path (Join-Path $InstallDir 'config.cmd') -PathType Leaf) {
        Write-Host "Runner files already exist in $InstallDir" -ForegroundColor Green
        return
    }

    Write-Host 'Downloading the latest stable GitHub Actions runner...' -ForegroundColor Cyan
    $headers = @{ 'User-Agent' = 'NeralRunnerInstaller' }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/actions/runner/releases/latest' -Headers $headers
    $asset = $release.assets |
        Where-Object { $_.name -match '^actions-runner-win-x64-[0-9.]+\.zip$' } |
        Select-Object -First 1
    if (-not $asset) { throw 'Could not locate the latest Windows x64 runner package.' }

    $zip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $headers
    Expand-Archive -Path $zip -DestinationPath $InstallDir -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Write-Host "Installed runner files in $InstallDir" -ForegroundColor Green
}

function Register-LogonAutostart {
    $account = "$env:USERDOMAIN\$env:USERNAME"
    $action = New-ScheduledTaskAction -Execute (Join-Path $InstallDir 'run.cmd') -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
    $principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Write-Host 'Runner autostart is enabled for this Windows user.' -ForegroundColor Green
}

Write-Host '=== Neral self-hosted runner one-click installer ===' -ForegroundColor Cyan
Write-Host 'Repository: Cola968/neral-helios'
Write-Host 'This installer does not modify Neral weights, checkpoints, or training data.'

if (-not (Test-IsAdministrator)) {
    Write-Host 'Requesting Administrator permission for installation...' -ForegroundColor Yellow
    $args = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $args
    exit 0
}

Install-RunnerFiles
$envInfo = Resolve-UserEnvironment

$runnerMarker = Join-Path $InstallDir '.runner'
$serviceMarker = Join-Path $InstallDir '.service'

if (-not (Test-Path $runnerMarker -PathType Leaf)) {
    Write-Host ''
    Write-Host 'One manual secret is still required once: the short-lived GitHub runner registration token.' -ForegroundColor Yellow
    Write-Host 'The GitHub setup page will open now. Keep Windows / x64 selected.'
    Write-Host 'Copy ONLY the value after --token and paste it into the hidden prompt here.'
    Start-Process $SetupUrl
    $token = Read-SecretText 'Runner registration token'
    if ([string]::IsNullOrWhiteSpace($token)) { throw 'No registration token was entered.' }

    Write-Host ''
    Write-Host 'Autostart mode:' -ForegroundColor Cyan
    Write-Host '  1 = current Windows user at logon (recommended for Astra + Neral local files)'
    Write-Host '  2 = Windows service (advanced; runs as NetworkService by default)'
    $mode = Read-Host 'Choose [1]'
    if ([string]::IsNullOrWhiteSpace($mode)) { $mode = '1' }
    if ($mode -notin @('1','2')) { throw 'Choose 1 or 2.' }

    Push-Location $InstallDir
    try {
        $configArgs = @(
            '--unattended',
            '--url', $RepoUrl,
            '--token', $token,
            '--name', $RunnerName,
            '--labels', $RunnerLabel,
            '--work', '_work',
            '--replace'
        )
        if ($mode -eq '2') { $configArgs += '--runasservice' }
        & (Join-Path $InstallDir 'config.cmd') @configArgs
        if ($LASTEXITCODE -ne 0) { throw "config.cmd failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
        $token = $null
    }

    if ($mode -eq '2') {
        if ($envInfo.Assets) { [Environment]::SetEnvironmentVariable('NERAL_ASSETS_ROOT', $envInfo.Assets, 'Machine') }
        if ($envInfo.Python) { [Environment]::SetEnvironmentVariable('NERAL_PYTHON', $envInfo.Python, 'Machine') }
        Write-Host ''
        Write-Host 'Windows service mode is configured.' -ForegroundColor Green
        Write-Host 'Note: NetworkService must have read/write permission to NERAL_ASSETS_ROOT for benchmark/training actions.' -ForegroundColor Yellow
        if (Test-Path $serviceMarker -PathType Leaf) {
            $serviceName = (Get-Content $serviceMarker -Raw).Trim()
            if ($serviceName) { Start-Service -Name $serviceName -ErrorAction SilentlyContinue }
        }
    }
    else {
        Register-LogonAutostart
    }
}
else {
    Write-Host 'Runner is already registered with GitHub.' -ForegroundColor Green
    if (Test-Path $serviceMarker -PathType Leaf) {
        $serviceName = (Get-Content $serviceMarker -Raw).Trim()
        if ($serviceName) {
            Start-Service -Name $serviceName -ErrorAction SilentlyContinue
            Write-Host "Windows service started: $serviceName" -ForegroundColor Green
        }
    }
    else {
        Register-LogonAutostart
    }
}

Write-Host ''
Write-Host 'Checking NVIDIA GPU:' -ForegroundColor Cyan
try {
    & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader
}
catch {
    Write-Host 'nvidia-smi could not be executed. Runner setup itself may still be valid.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'DONE. The runner should now pick up the already queued Neral probe jobs.' -ForegroundColor Green
Write-Host 'Do not close anything manually when logon autostart/service mode is used.'
