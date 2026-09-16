param(
  [Parameter(Mandatory=$true)][string]$ReportPath,
  [Parameter(Mandatory=$true)][string]$ScratchPath,
  [Parameter(Mandatory=$true)][string]$Repository,
  [Parameter(Mandatory=$true)][string]$Branch,
  [Parameter(Mandatory=$true)][string]$Token,
  [Parameter(Mandatory=$true)][string]$RunId,
  [long]$TargetTokens = 2000000000,
  [int]$IntervalSeconds = 60,
  [string]$LivePath = 'frontier350_live/v7_2.json'
)

$ErrorActionPreference = 'Continue'
$headers = @{
  Authorization = "Bearer $Token"
  Accept = 'application/vnd.github+json'
  'X-GitHub-Api-Version' = '2022-11-28'
}
$api = "https://api.github.com/repos/$Repository/contents/$LivePath"

function Read-JsonFile([string]$Path) {
  try {
    if (Test-Path -LiteralPath $Path) {
      return (Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop)
    }
  } catch {}
  return $null
}

function Get-LatestTrainMetric {
  try {
    $files = @(Get-ChildItem -LiteralPath $ScratchPath -Recurse -Filter metrics.jsonl -File -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTimeUtc -Descending)
    foreach ($f in $files) {
      $lines = @(Get-Content -LiteralPath $f.FullName -Tail 80 -ErrorAction SilentlyContinue)
      for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        try {
          $o = $lines[$i] | ConvertFrom-Json -ErrorAction Stop
          if ($o.event -eq 'train' -and $null -ne $o.processed_tokens) {
            return [pscustomobject]@{ metric=$o; file=$f.FullName; mtime=$f.LastWriteTimeUtc }
          }
        } catch {}
      }
    }
  } catch {}
  return $null
}

function Get-LatestProgress {
  $p = Read-JsonFile (Join-Path $ReportPath 'progress.json')
  if ($p -is [System.Array] -and $p.Count -gt 0) { return $p[-1] }
  return $null
}

function Get-Gpu {
  try {
    $line = (& nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ($line) {
      $v = $line.Split(',') | ForEach-Object { $_.Trim() }
      return [ordered]@{ temp_c=[int]$v[0]; vram_used_mib=[int]$v[1]; vram_free_mib=[int]$v[2]; util_pct=[int]$v[3] }
    }
  } catch {}
  return $null
}

function Publish-Live([string]$State='running') {
  try {
    $train = Get-LatestTrainMetric
    $progress = Get-LatestProgress
    $selection = Read-JsonFile (Join-Path $ReportPath 'selection.json')

    $processed = $null
    $tps = $null
    $loss = $null
    $stageStep = $null
    if ($train) {
      $processed = [long]$train.metric.processed_tokens
      if ($null -ne $train.metric.tokens_per_second) { $tps = [double]$train.metric.tokens_per_second }
      if ($null -ne $train.metric.loss) { $loss = [double]$train.metric.loss }
      if ($null -ne $train.metric.step) { $stageStep = [int]$train.metric.step }
    }

    $accepted = $null
    $health = $null
    $audit = $null
    if ($progress) {
      if ($null -ne $progress.tokens_after) { $accepted = [long]$progress.tokens_after }
      elseif ($null -ne $progress.continuity_tokens) { $accepted = [long]$progress.continuity_tokens }
      if ($null -ne $progress.health) { $health = $progress.health }
      if ($null -ne $progress.audit) { $audit = $progress.audit }
    }
    if ($selection -and $null -ne $selection.final_train_tokens) { $accepted = [long]$selection.final_train_tokens }

    # Only quality-accepted continuity counts toward completion. In-progress tokens
    # can be rejected and are reported separately above; never inflate progress.
    $basis = $accepted
    $pct = if ($basis -gt 0) { [math]::Round(100.0 * $basis / $TargetTokens, 4) } else { $null }
    $remaining = if ($basis -gt 0) { [math]::Max(0, $TargetTokens - $basis) } else { $null }
    # A single instantaneous training sample cannot predict quality-approved wall time.
    # Leave legacy ETA field null instead of publishing misleading 90-hour idle estimates.
    $etaSec = $null

    $payload = [ordered]@{
      schema = 1
      run_id = "$RunId"
      state = $State
      updated_at_utc = [DateTime]::UtcNow.ToString('o')
      target_tokens = $TargetTokens
      processed_tokens_live = $processed
      accepted_continuity_tokens = $accepted
      progress_pct_live = $pct
      remaining_tokens_live = $remaining
      tokens_per_second_last = $tps
      loss_last = $loss
      stage_step_last = $stageStep
      pure_compute_eta_seconds = $etaSec
      health_last = $health
      audit_last = $audit
      gpu = Get-Gpu
      source_metrics = if ($train) { $train.file } else { $null }
      note = 'Progress/remaining count accepted tokens only; instantaneous trainer tps is not end-to-end throughput. ETA unavailable until validated. Telemetry never changes training, gates or recovery.'
    }

    $json = $payload | ConvertTo-Json -Depth 12
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    $content = [Convert]::ToBase64String($bytes)
    $sha = $null
    try {
      $existing = Invoke-RestMethod -Uri ($api + '?ref=' + [uri]::EscapeDataString($Branch)) -Headers $headers -Method Get -ErrorAction Stop
      $sha = $existing.sha
    } catch {}
    $body = [ordered]@{ message="telemetry: v7.2 live $RunId"; content=$content; branch=$Branch }
    if ($sha) { $body.sha = $sha }
    Invoke-RestMethod -Uri $api -Headers $headers -Method Put -Body ($body | ConvertTo-Json) -ContentType 'application/json' -ErrorAction Stop | Out-Null
  } catch {
    Write-Host "LIVE_TELEMETRY_WARN: $($_.Exception.Message)"
  }
}

Write-Host "LIVE_TELEMETRY_START run=$RunId interval=${IntervalSeconds}s path=$LivePath"
while ($true) {
  Publish-Live 'running'
  Start-Sleep -Seconds $IntervalSeconds
}
