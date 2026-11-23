# CPU-only capacity policy shared by the owner deployment and WDDM preflight.
# The pre-load check runs AFTER the previous owner stops, BEFORE StartOwned.
# The post-load check runs only with authenticated NERAL on HTTP 9690.
function Assert-NeralGpuCapacity {
  param(
    [Parameter(Mandatory=$true)][ValidateSet('PreLoad','PostLoad')][string]$Phase,
    [Parameter(Mandatory=$true)][int]$FreeMiB,
    [Parameter(Mandatory=$true)][int]$UsedMiB
  )
  if($FreeMiB -lt 0 -or $UsedMiB -lt 0){
    throw 'NERAL_GPU_PREFLIGHT_REASON:invalid_gpu_memory_measurement'
  }
  if($Phase -ceq 'PreLoad'){
    if($FreeMiB -lt 8500){
      throw 'NERAL_GPU_PREFLIGHT_REASON:driver_preload_free_below_8500_mib'
    }
    return
  }
  # NERAL's own already-loaded GPU tensors reduce free memory. Never apply
  # the 8,500 MiB startup requirement to the running NERAL owner.
  if($UsedMiB -gt 10752){
    throw 'NERAL_GPU_PREFLIGHT_REASON:driver_postload_used_above_10752_mib'
  }
  if($FreeMiB -lt 1000){
    throw 'NERAL_GPU_PREFLIGHT_REASON:driver_postload_free_below_1000_mib'
  }
}
