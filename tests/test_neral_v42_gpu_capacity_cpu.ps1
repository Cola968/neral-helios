# CPU-only exact acceptance of pre-model and already-loaded GPU states.
$ErrorActionPreference='Stop'
$capacity=Join-Path $PSScriptRoot '../scripts/neral_v42_gpu_capacity.ps1'
if(-not(Test-Path -LiteralPath $capacity)){throw 'Pre/Post GPU capacity helper absent'}
. $capacity

function Assert-Denied([scriptblock]$Check,[string]$Reason){
  $failed=$false
  try { & $Check } catch { $failed=[string]$_.Exception.Message -like ('*'+$Reason+'*') }
  if(-not $failed){throw ('GPU capacity did not deny '+$Reason)}
}
Assert-NeralGpuCapacity -Phase PreLoad -FreeMiB 9799 -UsedMiB 2213
Assert-NeralGpuCapacity -Phase PreLoad -FreeMiB 8500 -UsedMiB 3782
Assert-Denied { Assert-NeralGpuCapacity -Phase PreLoad -FreeMiB 8499 -UsedMiB 3783 } 'driver_preload_free_below_8500_mib'
Assert-NeralGpuCapacity -Phase PostLoad -FreeMiB 1979 -UsedMiB 10303
Assert-NeralGpuCapacity -Phase PostLoad -FreeMiB 1000 -UsedMiB 10752
Assert-Denied { Assert-NeralGpuCapacity -Phase PostLoad -FreeMiB 999 -UsedMiB 10303 } 'driver_postload_free_below_1000_mib'
Assert-Denied { Assert-NeralGpuCapacity -Phase PostLoad -FreeMiB 1029 -UsedMiB 10753 } 'driver_postload_used_above_10752_mib'
Assert-Denied { Assert-NeralGpuCapacity -Phase PreLoad -FreeMiB -1 -UsedMiB 500 } 'invalid_gpu_memory_measurement'

# Enforce the call sites, not only a free-floating helper that nobody uses.
$owner=[IO.File]::ReadAllText((Join-Path $PSScriptRoot '../scripts/neral_v41_owner.ps1'))
$preflight=[IO.File]::ReadAllText((Join-Path $PSScriptRoot '../scripts/neral_v42_gpu_owner_preflight.ps1'))
if($owner -notmatch 'Assert-NeralGpuCapacity\s+-Phase\s+PreLoad' -or
   $preflight -notmatch 'Assert-NeralGpuCapacity\s+-Phase\s+PostLoad'){
  throw 'PreLoad/PostLoad capacity phases not wired into their actual owner paths'
}
if($owner.IndexOf('StopOwned\n  $changed=$true'.Replace('\n',"`n")) -ge 0){
  # The live deployment has to stop the previous owner before taking a baseline.
  if($owner.IndexOf('Write-NeralGpuBaseline') -lt 0){throw 'No pre-load baseline function'}
}
Write-Host 'PASS PRELOAD_9799_ALLOWED_PRELOAD_8499_BLOCKED'
Write-Host 'PASS POSTLOAD_1979_ALLOWED_POSTLOAD_999_AND_USED_10753_BLOCKED'
