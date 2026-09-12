param(
  [Parameter(Mandatory=$true)][string]$ScratchWin,
  [Parameter(Mandatory=$true)][string]$ReportWin,
  [Parameter(Mandatory=$true)][string]$StopFile,
  [Parameter(Mandatory=$true)][string]$Repo,
  [Parameter(Mandatory=$true)][int]$IssueNumber,
  [Parameter(Mandatory=$true)][string]$RunId,
  [Parameter(Mandatory=$true)][double]$TargetTps,
  [Parameter(Mandatory=$true)][int]$UpdatesPerTrial,
  [int]$IntervalSeconds=20,
  [int]$TotalTrials=7
)

$ErrorActionPreference='SilentlyContinue'
$token=$env:NERAL_TELEMETRY_TOKEN
if([string]::IsNullOrWhiteSpace($token)){ exit 0 }

$headers=@{
  Authorization="Bearer $token"
  Accept='application/vnd.github+json'
  'X-GitHub-Api-Version'='2022-11-28'
  'User-Agent'='neral-live-telemetry'
}
$issueApi="https://api.github.com/repos/$Repo/issues/$IssueNumber"
$started=Get-Date

function Get-Median([double[]]$Values){
  if(-not $Values -or $Values.Count -eq 0){ return $null }
  $s=@($Values | Sort-Object)
  $n=$s.Count
  if($n % 2 -eq 1){ return [double]$s[[int][math]::Floor($n/2)] }
  return ([double]$s[$n/2-1]+[double]$s[$n/2])/2.0
}

function Get-GpuState {
  try {
    $line=(nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.free,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits | Select-Object -First 1)
    $p=@($line -split ',' | ForEach-Object {$_.Trim()})
    return [pscustomobject]@{temp=[int]$p[0]; used=[int]$p[1]; free=[int]$p[2]; total=[int]$p[3]; util=[int]$p[4]; power=[double]$p[5]}
  } catch { return $null }
}

function Get-LiveMetric {
  try {
    $file=Get-ChildItem -LiteralPath $ScratchWin -Filter metrics.jsonl -File -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if(-not $file){ return $null }
    $rows=@()
    foreach($line in (Get-Content -LiteralPath $file.FullName -Tail 80)){
      try {
        $r=$line | ConvertFrom-Json
        if($r.event -eq 'train'){ $rows += $r }
      } catch {}
    }
    if($rows.Count -eq 0){ return $null }
    $recent=@($rows | Select-Object -Last 10)
    $rates=@($recent | ForEach-Object {[double]$_.tokens_per_second})
    $last=$recent[-1]
    $trial=Split-Path -Leaf (Split-Path -Parent $file.FullName)
    return [pscustomobject]@{
      trial=$trial
      step=[int]$last.step
      loss=[double]$last.loss
      current=[double]$last.tokens_per_second
      median=(Get-Median $rates)
      peak=[double](($rates | Measure-Object -Maximum).Maximum)
      processed=[long]$last.processed_tokens
      newtokens=[long]$last.new_tokens
      tpu=[long]$last.tokens_per_update
      alloc=[long]$last.peak_allocated_bytes
      reserved=[long]$last.peak_reserved_bytes
    }
  } catch { return $null }
}

function Publish([string]$Phase){
  try {
    $gpu=Get-GpuState
    $m=Get-LiveMetric
    $completed=0
    if(Test-Path -LiteralPath $ReportWin){
      $completed=@(Get-ChildItem -LiteralPath $ReportWin -Filter '*_summary.json' -File).Count
    }
    $now=Get-Date
    $elapsed=($now-$started).TotalSeconds
    $lines=@(
      '# Neral Live Telemetry',
      '',
      "**Run:** [$RunId](https://github.com/$Repo/actions/runs/$RunId)  ",
      "**Updated:** $($now.ToString('yyyy-MM-dd HH:mm:ss K'))  ",
      "**Phase:** $Phase  ",
      "**Trials completed:** $completed / $TotalTrials  "
    )
    if($m){
      $pct=100.0*$m.median/$TargetTps
      $remaining=[math]::Max(0,$UpdatesPerTrial-$m.step)
      $etaTrial=$null
      if($m.median -gt 0){ $etaTrial=$remaining*$m.tpu/$m.median }
      $lines += @(
        '',
        '## Throughput',
        "- Trial: **$($m.trial)**",
        "- Current: **$([math]::Round($m.current,0)) tok/s**",
        "- Rolling median (last <=10): **$([math]::Round($m.median,0)) tok/s**",
        "- Rolling peak: **$([math]::Round($m.peak,0)) tok/s**",
        "- Target progress: **$([math]::Round($pct,1))% of $([math]::Round($TargetTps,0)) tok/s**",
        "- Step: **$($m.step) / $UpdatesPerTrial**",
        "- Loss: **$([math]::Round($m.loss,5))**",
        "- Processed tokens reported by trainer: **$($m.processed)**",
        "- Current-trial ETA: **$(if($etaTrial -ne $null){[timespan]::FromSeconds($etaTrial).ToString('mm\:ss')}else{'n/a'})**",
        "- Trainer peak allocated VRAM: **$([math]::Round($m.alloc/1GB,2)) GiB**",
        "- Trainer peak reserved VRAM: **$([math]::Round($m.reserved/1GB,2)) GiB**"
      )
    } else {
      $lines += @('', '## Throughput', '- Waiting for the first trainer metric row; compile/startup may still be running.')
    }
    if($gpu){
      $lines += @(
        '',
        '## GPU',
        "- Temperature: **$($gpu.temp) C**",
        "- Utilization: **$($gpu.util)%**",
        "- VRAM: **$($gpu.used) / $($gpu.total) MiB** (free $($gpu.free) MiB)",
        "- Power: **$([math]::Round($gpu.power,1)) W**"
      )
    }
    $lines += @(
      '',
      '---',
      "Telemetry sidecar elapsed: $([timespan]::FromSeconds($elapsed).ToString('hh\:mm\:ss')).",
      'This channel is independent of the GitHub Actions downloadable log blob, so `BlobNotFound` does not block live status.'
    )
    $payload=@{body=($lines -join "`n")} | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Patch -Uri $issueApi -Headers $headers -Body $payload -ContentType 'application/json' | Out-Null
  } catch {}
}

Publish 'running'
while(-not (Test-Path -LiteralPath $StopFile)){
  Start-Sleep -Seconds $IntervalSeconds
  if(Test-Path -LiteralPath $StopFile){ break }
  Publish 'running'
}
Publish 'sweep step finished; final selection follows in Actions'
exit 0
