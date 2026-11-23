# CPU-only: never run owner control on public CI runners.
$ErrorActionPreference='Stop'
$script=Join-Path $PSScriptRoot '../scripts/neral_v43_reference50_raw_owner.ps1'
$owner=Join-Path $PSScriptRoot '../scripts/neral_v41_owner.ps1'
foreach($file in @($script,$owner)){
  $tokens=$null;$parseErrors=$null
  $null=[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $file),[ref]$tokens,[ref]$parseErrors)
  if(@($parseErrors).Count -ne 0){throw 'V4.3 owner script PowerShell syntax invalid'}
}
$src=[IO.File]::ReadAllText($script)
if($src -notmatch 'finally\s*\{' -or
   $src -notmatch 'Restore-OriginalOwner' -or
   $src -notmatch 'Get-Owner' -or
   $src -notmatch 'Stop-ConfirmedOwner' -or
   $src -notmatch 'Check-RawCapacityAndForeign' -or
   $src -notmatch 'free -lt 8500' -or
   $src -notmatch 'NERAL_V43_REF50_ORIGINAL_OWNER_RESTORED_HTTP200' -or
   $src -match 'Stop-Process\s+-Name' -or
   $src -match 'docker\s+run\s+--privileged'){
  throw 'Raw 9B owner safety contract missing'
}
$original=[IO.File]::ReadAllText($owner)
if($original -notmatch 'approvedV43Reference50' -or
   $original -notmatch 'NERAL_V42_SMOKE5_REVERT' -or
   $original -notmatch 'Write-NeralGpuBaseline'){
  throw 'V4.3 reference50 stage or rollback not authorized safely'
}
$restore=Join-Path $PSScriptRoot '../scripts/neral_v43_reference50_restore.ps1'
if(-not (Test-Path -LiteralPath $restore -PathType Leaf)){
  throw 'Independent post-raw owner restore script absent'
}
$parseErrors=$null;$tokens=$null
$null=[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $restore),[ref]$tokens,[ref]$parseErrors)
if(@($parseErrors).Count -ne 0){throw 'Independent restore script syntax invalid'}
$restoreSource=[IO.File]::ReadAllText($restore)
foreach($required in @('Get-ScheduledTask','Get-NetTCPConnection','Start-ScheduledTask',
                      'NERAL_V43_REF50_ORIGINAL_OWNER_RESTORED_HTTP200',
                      'NERAL_V43_REF50_ORIGINAL_OWNER_VERIFIED_HTTP200')){
  if(-not $restoreSource.Contains($required)){throw ('Restore safety missing '+$required)}
}
if($restoreSource -match 'Stop-Process|Stop-ScheduledTask|Remove-Item|Invoke-Expression'){
  throw 'Independent restore may not kill processes or alter files'
}

Write-Host 'PASS V43 owner raw9B exclusivity and original-HTTP200 rollback CPU static checks'
