# CPU-only contract: the preflight's built-in synthetic mode must test
# bracketed WDDM [N/A] and normal dwm against a foreign Python compute owner.
$ErrorActionPreference='Stop'
$out=@(& (Join-Path $PSScriptRoot '../scripts/neral_v42_gpu_owner_preflight.ps1') -CpuTest)
if($LASTEXITCODE -ne 0 -or
  -not ($out -match 'NERAL_WDDM_CPU_GUI_NA_ALLOWED') -or
  -not ($out -match 'NERAL_WDDM_CPU_FOREIGN_PYTHON_BLOCKED') -or
  -not ($out -match 'NERAL_WDDM_CPU_LOW_FREE_BLOCKED') -or
  -not ($out -match 'NERAL_WDDM_CPU_SYSTEM_CAP_BLOCKED') -or
  -not ($out -match 'NERAL_WDDM_CPU_POSTLOAD_1979_ALLOWED')){
  throw 'WDDM CPU preflight regression missing'
}
Write-Host 'PASS final WDDM CPU preflight contracts'
