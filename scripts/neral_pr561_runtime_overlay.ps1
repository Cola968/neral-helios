# Complete, guarded replacement of the reviewed source-only preview package.
# Dot-source from the owner-verified V1 installer; never touch local_state,
# model weights, user accounts, secrets or the owned Task Scheduler script.

function Initialize-NeralManagedSnapshot {
  param([string]$Repo,[string]$Root,[string]$Backup,[string]$Python)
  $ErrorActionPreference='Stop'
  $raw=@(& $Python -B -c 'import json;from scripts.build_neral_1_0_preview import REQUIRED_FILES;print(json.dumps(REQUIRED_FILES))' 2>&1)
  if($LASTEXITCODE -ne 0 -or @($raw).Count -ne 1){
    throw 'Managed preview manifest unavailable'
  }
  $files=@(ConvertFrom-Json -InputObject ([string]$raw[0]))
  if($files.Count -lt 80 -or $files.Count -gt 150){
    throw 'Unexpected number of managed preview sources'
  }
  $snapshot=Join-Path $Backup 'source-package-snapshot'
  New-Item -ItemType Directory -Path $snapshot -Force|Out-Null
  $before=@{}
  foreach($rel in $files){
    if($rel -notmatch '^[A-Za-z0-9_./ -]{1,160}$' -or
       $rel -match '(^|/)\.\.?(/|$)' -or $rel.StartsWith('/')){
      throw 'Unsafe managed package path'
    }
    $source=Join-Path $Repo $rel
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){
      throw "Missing approved runtime source: $rel"
    }
    if(((Get-Item -LiteralPath $source -Force).Attributes -band
         [IO.FileAttributes]::ReparsePoint) -ne 0){
      throw "Symlinked runtime source not allowed: $rel"
    }
    $target=Join-Path $Root $rel
    $exists=Test-Path -LiteralPath $target -PathType Leaf
    if($exists -and (((Get-Item -LiteralPath $target -Force).Attributes -band
         [IO.FileAttributes]::ReparsePoint) -ne 0)){
      throw "Symlinked installed runtime source not allowed: $rel"
    }
    $oldHash=''
    if($exists){
      $oldHash=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
      $saved=Join-Path $snapshot $rel
      if(-not(Test-Path -LiteralPath $saved -PathType Leaf)){
        New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force|Out-Null
        Copy-Item -LiteralPath $target -Destination $saved -ErrorAction Stop
      }
      if((Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash -ne $oldHash){
        throw "Managed rollback snapshot differs from installed file: $rel"
      }
    }
    $before[$rel]=@{existed=$exists;sha=$oldHash}
  }
  Write-Host ('NERAL_PR561_MANAGED_SOURCE_SNAPSHOT_READY files='+$files.Count)
  return @{files=$files;before=$before;snapshot=$snapshot}
}

function Install-NeralManagedRuntime {
  param([hashtable]$State,[string]$Repo,[string]$Root)
  $ErrorActionPreference='Stop'
  foreach($rel in $State.files){
    $source=Join-Path $Repo $rel
    $target=Join-Path $Root $rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force -ErrorAction Stop
    if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
       (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash){
      throw "Runtime file did not install exactly: $rel"
    }
  }
  Write-Host ('NERAL_PR561_FULL_MANAGED_RUNTIME_HASHES_PASS files='+$State.files.Count)
}

function Restore-NeralManagedRuntime {
  param([hashtable]$State,[string]$Root)
  $ErrorActionPreference='Stop'
  foreach($rel in $State.files){
    $target=Join-Path $Root $rel
    $previous=$State.before[$rel]
    if($previous.existed){
      Copy-Item -LiteralPath (Join-Path $State.snapshot $rel) -Destination $target -Force -ErrorAction Stop
      if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $previous.sha){
        throw "Managed rollback SHA256 did not match: $rel"
      }
    }elseif(Test-Path -LiteralPath $target -PathType Leaf){
      Remove-Item -LiteralPath $target -Force -ErrorAction Stop
    }
  }
  Write-Host ('NERAL_PR561_FULL_MANAGED_RUNTIME_ROLLBACK_PASS files='+$State.files.Count)
}
