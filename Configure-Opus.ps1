param()
$ErrorActionPreference='Stop'

$state=Join-Path $PSScriptRoot 'local_state'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$keyFile=Join-Path $state 'explabs-key.dpapi'

Write-Host 'Claude Opus 5 fuer Helios einrichten' -ForegroundColor Cyan
Write-Host 'Dein Experiential-Key wird verschluesselt fuer dein Windows-Benutzerkonto gespeichert.'
$secure=Read-Host 'Experiential API-Key (xpl_...)' -AsSecureString

$ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try{
    $plain=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    if(-not $plain.StartsWith('xpl_')){throw 'Der Key muss mit xpl_ beginnen.'}
    if($plain.Length -lt 12){throw 'Der Key ist zu kurz.'}
}finally{
    if($ptr -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)}
    if(Get-Variable plain -ErrorAction SilentlyContinue){Remove-Variable plain -Force}
}

$secure | ConvertFrom-SecureString | Set-Content -LiteralPath $keyFile -Encoding ASCII
Write-Host 'Fertig. Starte Neral jetzt neu. Claude Opus 5 erscheint dann in der Modellauswahl.' -ForegroundColor Green
Write-Host 'Der Key wird nicht in Git gespeichert und nicht an den Browser weitergegeben.'
Read-Host 'Enter zum Schliessen' | Out-Null
