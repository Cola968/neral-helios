# NERAL local Windows Docker readiness. No install, downloads, or GPU access.
# Intended to be dot-sourced inside an authorized Windows runner step.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') { throw 'NERAL_DOCKER_WINDOWS_ONLY' }
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) { throw 'NERAL_DOCKER_CLI_MISSING: Docker CLI ist nicht installiert.' }

function Test-DockerReady {
    param([string] $Context = '')
    # Windows PowerShell 5.1 can turn native stderr into terminating errors.
    # cmd.exe swallows only the readiness probe's output; do not hide real errors.
    if ($Context) {
        if ($Context -notmatch '^[A-Za-z0-9_-]{1,64}$') { return $false }
        & cmd.exe /d /c "docker --context $Context info >NUL 2>&1"
    } else {
        & cmd.exe /d /c 'docker info >NUL 2>&1'
    }
    return ($LASTEXITCODE -eq 0)
}

function Get-DockerDesktopExecutable {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
        'D:\Program Files\Docker\Docker\Docker Desktop.exe',
        'D:\Docker\Docker\Docker Desktop.exe',
        (Join-Path $env:LOCALAPPDATA 'Programs\Docker\Docker\Docker Desktop.exe'),
        (Join-Path $env:LOCALAPPDATA 'Docker\Docker Desktop.exe')
    )
    foreach ($reg in @(
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )) {
        $entries = @(Get-ItemProperty -Path $reg -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -eq 'Docker Desktop' })
        foreach ($entry in $entries) {
            if ($entry.InstallLocation) {
                $candidates += (Join-Path ([string]$entry.InstallLocation) 'Docker Desktop.exe')
            }
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    return $null
}

function Test-DockerWSLDistribution {
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        Write-Host 'NERAL_DOCKER_WSL_NOT_INSTALLED'
        return $false
    }
    # Listing does not start a distro. Avoid printing distro/user paths.
    $names = @(& wsl.exe --list --quiet 2>$null)
    $present = (($names -join '') -match 'docker-desktop')
    Write-Host ('NERAL_DOCKER_WSL_DISTRO_PRESENT ' + $present)
    return $present
}

if (-not (Test-DockerReady)) {
    if (Test-DockerReady -Context 'desktop-linux') {
        # In this process only: no persistent change of global Docker context.
        $env:DOCKER_CONTEXT = 'desktop-linux'
        Write-Host 'NERAL_DOCKER_ALT_CONTEXT_READY desktop-linux'
    } else {
        $exe = Get-DockerDesktopExecutable
        $hasWSL = Test-DockerWSLDistribution
        if (-not $exe) {
            throw ('NERAL_DOCKER_DESKTOP_MISSING: Keine Desktop.exe unter C:, D:, LOCALAPPDATA ' +
                   'oder in HKCU/HKLM registriert; docker-desktop WSL vorhanden=' + $hasWSL)
        }
        Write-Host 'NERAL_DOCKER_DESKTOP_INSTALLED_DAEMON_CLOSED'
        try {
            Start-Process -FilePath $exe -ArgumentList '--minimized' -ErrorAction Stop | Out-Null
            Write-Host 'NERAL_DOCKER_DESKTOP_START_REQUESTED'
        } catch {
            throw ('NERAL_DOCKER_DESKTOP_CANNOT_START_IN_RUNNER_SESSION: ' +
                   $_.Exception.GetType().Name)
        }
        # Exactly 30 bounded 3s polls: maximum 90 seconds.
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            if (Test-DockerReady) { break }
            if (Test-DockerReady -Context 'desktop-linux') {
                $env:DOCKER_CONTEXT = 'desktop-linux'
                break
            }
            Start-Sleep -Seconds 3
        }
    }
}
if (-not (Test-DockerReady)) {
    # The WSL command below is diagnostic only and must not create a sandbox
    # or try to bypass the Windows user's Docker Desktop engine.
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    $detail = 'WSL distro unknown'
    if ($wsl) {
        $names = @(& wsl.exe --list --quiet 2>$null)
        if (($names -join '') -match 'docker-desktop') {
            # User can verify via: wsl -d docker-desktop; do not auto-start WSL in headless CI.
            $detail = 'wsl -d docker-desktop registriert, Engine vom Runner aus nicht erreichbar'
        }
    }
    throw ('NERAL_DOCKER_DESKTOP_CLOSED_OR_HEADLESS: Docker Desktop ist geschlossen ' +
           'oder läuft in einer anderen Windows-Sitzung; ' + $detail +
           '. Bitte Docker Desktop interaktiv starten und erneut docker info prüfen.')
}

# Keep the existing offline-only trusted image contract unchanged.
& cmd.exe /d /c 'docker image inspect python:3.11-alpine >NUL 2>&1'
if ($LASTEXITCODE -ne 0) {
    throw 'NERAL_DOCKER_SANDBOX_IMAGE_MISSING: python:3.11-alpine lokal nicht vorhanden; kein Auto-Pull.'
}
Write-Host 'NERAL_DOCKER_SANDBOX_READY_OFFLINE'
