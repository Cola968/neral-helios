$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$state=Join-Path $PSScriptRoot 'local_state'
$pythonCandidates=@($env:NERAL_PYTHON,(Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),(Join-Path $PSScriptRoot '../neral/.venv/Scripts/python.exe'),(Join-Path $state 'python/Scripts/python.exe'))
$appPython=$pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if(-not $appPython){throw 'Zuerst Start Neral.cmd ausfuehren.'}
$env:PYTHONDONTWRITEBYTECODE='1'
if(-not(Test-Path -LiteralPath 'local_state/phone/config.json')){
    $setupPython=$appPython
    & $setupPython -c "import cryptography" 2>$null
    if($LASTEXITCODE -ne 0){
        $setupRoot=Join-Path $state 'setup-python'
        & $appPython -m venv $setupRoot
        if($LASTEXITCODE -ne 0){throw 'Separate TLS-Umgebung konnte nicht erstellt werden.'}
        $setupPython=Join-Path $setupRoot 'Scripts/python.exe'
        & $setupPython -m pip install cryptography
        if($LASTEXITCODE -ne 0){throw 'TLS-Vorbereitung fehlt.'}
    }
    & $setupPython setup_phone.py
    if($LASTEXITCODE -ne 0){throw 'Handy-Zugang konnte nicht vorbereitet werden.'}
}
Start-Process -FilePath $appPython -ArgumentList @('-X','utf8','phone_gateway.py') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $state 'phone.log') -RedirectStandardError (Join-Path $state 'phone-error.log') | Out-Null
Write-Host 'Handy-Zugang siehe local_state/phone/HANDY-ZUGANG.txt. CPU-App muss laufen. Bei IP-Wechsel Zugang neu einrichten.'
Get-Content -LiteralPath 'local_state/phone/HANDY-ZUGANG.txt'
