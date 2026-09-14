param(
  [Parameter(Mandatory=$true)][string]$ScratchWin,
  [Parameter(Mandatory=$true)][string]$StopFile,
  [long]$IntervalTokens = 50000000,
  [int]$Keep = 4,
  [int]$PollSeconds = 8
)

$ErrorActionPreference = 'Stop'
if($IntervalTokens -lt 1000000){ throw 'IntervalTokens too small' }
if($Keep -lt 1 -or $Keep -gt 8){ throw 'Keep must be 1..8' }

$seen = @{}

function Get-LastTrainRow([string]$metricsPath){
  $lines = Get-Content -LiteralPath $metricsPath -Tail 80 -ErrorAction SilentlyContinue
  for($i=$lines.Count-1; $i -ge 0; $i--){
    try {
      $row = $lines[$i] | ConvertFrom-Json
      if($row.event -eq 'train'){ return $row }
    } catch {}
  }
  return $null
}

function Copy-Atomic([string]$src,[string]$dst){
  $tmp = $dst + '.tmp-' + [guid]::NewGuid().ToString('N')
  Copy-Item -LiteralPath $src -Destination $tmp -Force
  Move-Item -LiteralPath $tmp -Destination $dst -Force
}

while(-not (Test-Path -LiteralPath $StopFile)){
  if(Test-Path -LiteralPath $ScratchWin){
    $metrics = Get-ChildItem -LiteralPath $ScratchWin -Filter metrics.jsonl -File -Recurse -ErrorAction SilentlyContinue
    foreach($m in $metrics){
      $row = Get-LastTrainRow $m.FullName
      if($null -eq $row){ continue }
      $newTokens = [long]$row.new_tokens
      $processed = [long]$row.processed_tokens
      if($newTokens -le 0){ continue }
      $bucket = [math]::Floor($newTokens / $IntervalTokens)
      if($bucket -lt 1){ continue }
      $trial = $m.Directory.FullName
      $key = $trial
      $last = if($seen.ContainsKey($key)){ [long]$seen[$key] } else { 0 }
      if($bucket -le $last){ continue }
      $resume = Join-Path $trial 'resume.pt'
      if(-not (Test-Path -LiteralPath $resume)){ continue }
      $age = (Get-Date) - (Get-Item -LiteralPath $resume).LastWriteTime
      if($age.TotalSeconds -lt 2){ continue }
      $stamp = $bucket * $IntervalTokens
      $dense = Join-Path $trial ("recovery_{0:D12}.pt" -f $stamp)
      $mid = Join-Path $trial 'midpoint.pt'
      Copy-Atomic $resume $dense
      Copy-Atomic $dense $mid
      $seen[$key] = $bucket
      Write-Host ("DENSE_CHECKPOINT trial={0} new_tokens={1} processed={2} path={3}" -f $trial,$newTokens,$processed,$dense)
      $old = Get-ChildItem -LiteralPath $trial -Filter 'recovery_*.pt' -File | Sort-Object Name
      while($old.Count -gt $Keep){
        Remove-Item -LiteralPath $old[0].FullName -Force
        $old = @($old | Select-Object -Skip 1)
      }
    }
  }
  Start-Sleep -Seconds $PollSeconds
}
