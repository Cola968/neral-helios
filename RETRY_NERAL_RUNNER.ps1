[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/Cola968/neral-helios'
$SetupUrl = "$RepoUrl/settings/actions/runners/new"
$InstallDir = 'C:\actions-runner'
$RunnerName = "$env:COMPUTERNAME-Neral"
$RunnerLabel = 'neral-gpu'
$TaskName = 'Neral GitHub Actions Runner'

function Read-SecretText([string]$Prompt) {
    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

function Register-LogonAutostart {
    $account = "$env:USERDOMAIN\$env:USERNAME"
    $action = New-ScheduledTaskAction -Execute (Join-Path $InstallDir 'run.cmd') -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
    $principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}

Write-Host '=== Neral runner registration retry ===' -ForegroundColor Cyan
Write-Host 'This retry does not touch Neral weights, checkpoints, corpora, or training state.'

try {
    $config = Join-Path $InstallDir 'config.cmd'
    if (-not (Test-Path $config -PathType Leaf)) {
        throw 'GitHub Actions runner files are not installed in C:\actions-runner. Run INSTALL_NERAL_RUNNER.cmd first.'
    }

    if (Test-Path (Join-Path $InstallDir '.runner') -PathType Leaf) {
        Write-Host 'Runner is already registered. Starting logon autostart now.' -ForegroundColor Green
        Register-LogonAutostart
    }
    else {
        Write-Host ''
        Write-Host 'Opening a fresh GitHub runner setup page...' -ForegroundColor Yellow
        Write-Host 'Keep Windows / x64 selected. Copy ONLY the value after --token.'
        Start-Process $SetupUrl
        $token = Read-SecretText 'Fresh runner registration token'
        if ([string]::IsNullOrWhiteSpace($token)) { throw 'No token was entered.' }

        Push-Location $InstallDir
        try {
            & $config --unattended --url $RepoUrl --token $token --name $RunnerName --labels $RunnerLabel --work _work --replace
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
            $token = $null
        }

        if ($exitCode -ne 0) {
            throw "config.cmd failed with exit code $exitCode. The most common cause is an invalid/expired registration token. Reload the GitHub setup page and retry with a fresh token."
        }

        Write-Host 'Runner registration succeeded.' -ForegroundColor Green
        Register-LogonAutostart
        Write-Host 'Autostart is enabled for the current Windows user.' -ForegroundColor Green
    }

    Write-Host ''
    Write-Host 'Checking NVIDIA GPU:' -ForegroundColor Cyan
    try {
        & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader
    }
    catch {
        Write-Host 'nvidia-smi could not be executed, but runner registration may still be valid.' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host 'DONE. Leave this window open for a moment; queued GitHub probe jobs should be picked up.' -ForegroundColor Green
}
catch {
    Write-Host ''
    Write-Host 'RUNNER RETRY FAILED:' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ''
    Write-Host 'The window will stay open so the full error remains visible.' -ForegroundColor Yellow
}
