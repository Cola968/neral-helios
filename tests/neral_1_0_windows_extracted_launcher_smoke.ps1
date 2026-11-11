# Real Windows one-click launcher smoke from an extracted release ZIP.
# Fake Ollama supplies only the installed-tag list. No GPU, no inference,
# model pull, private chats, training data or persistent runner changes.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot
$work=Join-Path $env:RUNNER_TEMP ("neral-oneclick-" + $env:GITHUB_RUN_ID + "-" + $env:GITHUB_RUN_ATTEMPT)
if(Test-Path -LiteralPath $work){throw 'Refuse previous or dirty test output'}
New-Item -Path $work -ItemType Directory -ErrorAction Stop | Out-Null
$zip=Join-Path $work 'release.zip'
$installDir=Join-Path $work 'installed'
$oldPath=$env:PATH
$oldConsent=$env:NERAL_PREVIEW_INSTALL_CONSENT
$spawnedOwner=$null
try {
    Push-Location $root
    try {
        & python -B -m scripts.build_neral_1_0_preview --out $zip
        if($LASTEXITCODE -ne 0){throw 'Cannot build exact NERAL ZIP'}
    }finally{Pop-Location}
    Expand-Archive -LiteralPath $zip -DestinationPath $installDir -ErrorAction Stop
    $app=Join-Path $installDir 'NERAL-1.0-Local'
    $starter=Join-Path $app 'Start-Neral-1.0-Local.ps1'
    if(-not(Test-Path -LiteralPath $starter -PathType Leaf)){throw 'Real release launcher missing'}

    # Emulator responds only to ollama list; any pull/run/etc is an error.
    $bin=Join-Path $work 'local-ollama'
    New-Item -Path $bin -ItemType Directory | Out-Null
    @'
@echo off
if /i "%~1"=="list" (
 echo NAME           ID              SIZE MODIFIED
 echo qwen3:8b       500a1f067a9f    5.2 GB now
 exit /b 0
)
exit /b 78
'@ | Set-Content -LiteralPath (Join-Path $bin 'ollama.cmd') -Encoding Ascii
    $env:PATH="$bin;$oldPath"
    $env:NERAL_PREVIEW_INSTALL_CONSENT='1'

    $existing=@(Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
    if($existing.Count -ne 0){throw 'Port 9690 not free; refusing to test another server'}

    & $starter -CheckOnly -Model qwen3:8b
    if(-not $?){throw 'Extracted package failed trusted local model inventory preflight'}

    # A broken or too-old Python must be rejected BEFORE venv / pip setup.
    # Fake Python returns a nonzero version-probe result. The real runner
    # interpreter remains available again immediately after this check.
    $badPythonDir=Join-Path $work 'unsupported-python'
    New-Item -Path $badPythonDir -ItemType Directory | Out-Null
    @'
@echo off
exit /b 27
'@ | Set-Content -LiteralPath (Join-Path $badPythonDir 'python.cmd') -Encoding Ascii
    $unsupportedRejected=$false
    try {
        $env:PATH="$badPythonDir;$bin;$oldPath"
        try {
            & $starter -PrepareOnly -Model qwen3:8b
        }catch{
            if([string]$_ -match 'Python 3\.11\+'){ $unsupportedRejected=$true }
        }
    }finally{
        $env:PATH="$bin;$oldPath"
    }
    if(-not $unsupportedRejected){throw 'Unsupported Python did not fail with actionable version message'}
    if(Test-Path -LiteralPath (Join-Path $app 'local_state\\preview-venv')){
        throw 'Unsupported Python unexpectedly created a preview virtual environment'
    }

    & $starter -PrepareOnly -Model qwen3:8b
    if(-not $?){throw 'Isolated preview Python bootstrap failed'}
    $isolated=Join-Path $app 'local_state\preview-venv\Scripts\python.exe'
    if(-not(Test-Path -LiteralPath $isolated -PathType Leaf)){throw 'Launcher failed to create isolated venv'}

    # This launches the actual Waitress server from the freshly extracted ZIP.
    & $starter -NoBrowser -Model qwen3:8b
    if(-not $?){throw 'Actual local NERAL one-click launch failed'}
    $listener=@(Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort 9690 -State Listen -ErrorAction SilentlyContinue)
    if($listener.Count -ne 1){throw 'Local NERAL Waitress did not bind 127.0.0.1:9690'}
    $spawnedOwner=[int]$listener[0].OwningProcess
    $proc=Get-CimInstance Win32_Process -Filter "ProcessId=$spawnedOwner"
    if(-not $proc -or $proc.CommandLine -notmatch 'neral_preview_server'){
        throw 'Port listener does not have the expected NERAL preview process identity'
    }

    $url='http://127.0.0.1:9690/'
    $page=Invoke-WebRequest -Uri $url -UseBasicParsing -SessionVariable session -TimeoutSec 10
    if($page.StatusCode -ne 200 -or $page.Content -notmatch 'NERAL 1.0 Local'){
        throw 'Real Windows browser entrypoint failed'
    }
    $status=Invoke-RestMethod -Uri ($url+'api/neral/preview/session') -WebSession $session -TimeoutSec 10
    if($status.model -ne 'qwen3:8b' -or $status.release_validated -ne $false -or
       $status.model_origin -ne 'local_foundation_not_neral_adapter'){
        throw 'Local preview reports untrusted model identity'
    }
    $svg=Invoke-WebRequest -Uri ($url+'static/icon.svg') -UseBasicParsing -WebSession $session -TimeoutSec 10
    if($svg.StatusCode -ne 200 -or $svg.Content -notmatch '<svg'){
        throw 'Extracted Windows starter did not serve approved NERAL V2 SVG'
    }

    # Starting a second instance must FAIL instead of silently attaching to
    # an unknown or competing process.
    $duplicateRejected=$false
    try {
        & $starter -NoBrowser -Model qwen3:8b
    }catch {
        if([string]$_ -match 'bereits belegt'){ $duplicateRejected=$true }
    }
    if(-not $duplicateRejected){throw 'Unsafe: second launcher was not rejected on busy port'}
    Write-Host 'NERAL 1.0 REAL WINDOWS EXTRACTED LAUNCHER PASS: local browser, signed-off provenance, SVG, duplicate refusal.'
} finally {
    # Only terminate the process started by this test and positively
    # identified as our preview server. Never kill unrelated Python or Ollama.
    if($null -ne $spawnedOwner){
        $proc=Get-CimInstance Win32_Process -Filter "ProcessId=$spawnedOwner" -ErrorAction SilentlyContinue
        if($proc -and $proc.CommandLine -match 'neral_preview_server'){
            Stop-Process -Id $spawnedOwner -Force -ErrorAction SilentlyContinue
        }
    }
    $env:PATH=$oldPath
    if($null -eq $oldConsent){Remove-Item Env:NERAL_PREVIEW_INSTALL_CONSENT -ErrorAction SilentlyContinue}
    else{$env:NERAL_PREVIEW_INSTALL_CONSENT=$oldConsent}
}
