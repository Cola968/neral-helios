# CPU-only Kiwix setup/start. Run manually AFTER Blind200.
[CmdletBinding()]
param(
    [string]$ZimDir = 'D:\neral-knowledge\zim',
    [string]$KiwixServe = 'kiwix-serve.exe',
    [string]$ToolsZip = '',
    [string]$ToolsSha256 = '',
    [string]$ToolsDir = 'D:\neral-knowledge\tools',
    [ValidateRange(1,15)][int]$MaxGB = 15,
    [switch]$SetupOnly,
    [switch]$HealthOnly
)
$ErrorActionPreference = 'Stop'
$health = 'http://127.0.0.1:9691/'
function Test-Kiwix {
    try {
        $r = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 2
        $c = Invoke-WebRequest -Uri ($health + 'catalog/v2/root.xml') -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200 -and $c.StatusCode -eq 200 -and $c.Content -match 'opds')
    } catch { return $false }
}
if ($HealthOnly) {
    if (!(Test-Kiwix)) { throw 'Kiwix on 9691 is not healthy.' }
    Write-Output 'KIWIX_9691_HEALTHY'; return
}
New-Item -ItemType Directory -Path $ZimDir -Force | Out-Null
if ($ToolsZip) {
    if ($ToolsSha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Provide a trusted SHA-256 for the official Kiwix tools ZIP.' }
    if ((Get-FileHash -LiteralPath $ToolsZip -Algorithm SHA256).Hash -ne $ToolsSha256) { throw 'Kiwix ZIP checksum mismatch.' }
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    Expand-Archive -LiteralPath $ToolsZip -DestinationPath $ToolsDir
    $exe = @(Get-ChildItem -LiteralPath $ToolsDir -Recurse -File -Filter 'kiwix-serve.exe')
    if ($exe.Count -ne 1) { throw 'Expected exactly one kiwix-serve.exe in tools directory.' }
    $KiwixServe = $exe[0].FullName
}
$zims = @(Get-ChildItem -LiteralPath $ZimDir -File -Filter '*.zim')
$bytes = ($zims | Measure-Object -Property Length -Sum).Sum
if ($bytes -gt ($MaxGB * 1000000000L)) { throw 'ZIM storage budget exceeded; choose smaller packages.' }
Write-Output ('ZIM files: {0}; bytes: {1}' -f $zims.Count, $bytes)
if ($SetupOnly) { return }
if (!$zims.Count) { throw 'No ZIM files. Download mini/top packages from https://download.kiwix.org/zim/ first.' }
if (Test-Kiwix) { Write-Output 'KIWIX_9691_ALREADY_HEALTHY'; return }
if (Get-NetTCPConnection -LocalPort 9691 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 9691 already occupied; no process stopped.' }
$binary = (Get-Command $KiwixServe -CommandType Application -ErrorAction Stop).Source
$arguments = @('--address=127.0.0.1','--port=9691','--threads=2','--searchLimit=4','--blockexternal')
foreach ($zim in $zims) { $arguments += ('"' + $zim.FullName + '"') }
$process = Start-Process -FilePath $binary -ArgumentList $arguments -PassThru -WindowStyle Hidden
for ($i=0; $i -lt 20; $i++) {
    if ($process.HasExited) { throw 'kiwix-serve exited before health check.' }
    if (Test-Kiwix) { Write-Output ('KIWIX_9691_HEALTHY PID=' + $process.Id); return }
    Start-Sleep -Milliseconds 250
}
if (!$process.HasExited) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
throw 'Kiwix health check failed; only the newly started process was stopped.'
