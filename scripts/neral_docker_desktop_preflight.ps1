# Fail-closed Docker availability check for explicitly authorized local TIR12.
# No image pulls, process termination, service changes, or sandbox bypass.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') { throw 'Windows-only Docker preflight' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Local Docker CLI not installed'
}

function Test-DockerReady {
    try {
        $version = & docker version --format '{{.Server.Version}}' 2>$null
        return ($LASTEXITCODE -eq 0 -and
                -not [string]::IsNullOrWhiteSpace(($version | Out-String)))
    } catch {
        return $false
    }
}

if (-not (Test-DockerReady)) {
    # The GitHub runner belongs to the user's Windows PC; start ONLY the
    # already-installed Docker Desktop app. Never download or install.
    $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $desktop -PathType Leaf)) {
        throw 'Docker Desktop is installed neither at the trusted default path nor running'
    }
    Start-Process -FilePath $desktop -ArgumentList '--minimized' -ErrorAction Stop | Out-Null
    # Bounded wait: at most 72 seconds. No daemon process kills.
    for ($attempt = 0; $attempt -lt 24; $attempt++) {
        Start-Sleep -Seconds 3
        if (Test-DockerReady) { break }
    }
}
if (-not (Test-DockerReady)) {
    throw 'Docker Desktop Linux engine is unavailable; stop instead of bypassing sandbox'
}
& docker image inspect python:3.11-alpine *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Pre-installed offline Python Docker sandbox missing'
}
Write-Host 'Docker sandbox verified offline; image pulls and engine bypass disabled'
