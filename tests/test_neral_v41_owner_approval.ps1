# Execute only the authorization prefix, never the owner deployment.
$ErrorActionPreference='Stop'
$path=Join-Path $PSScriptRoot '../scripts/neral_v41_owner.ps1'
$source=[IO.File]::ReadAllText($path)
$boundary=$source.IndexOf('$root=Join-Path')
if($boundary -lt 0){throw 'Authorization prefix boundary missing'}
$guard=[scriptblock]::Create($source.Substring(0,$boundary))
$token='NERAL-V41-BLIND200-APPROVED'
$mergeTitle='NERAL V4.1 owner-approved Blind200 final launch 20261121'
$cases=@(
  @{name='manual valid';ref='refs/heads/main';event='workflow_dispatch';token=$token;title='';attempt='1';preflight='success';ok=$true},
  @{name='merge valid';ref='refs/heads/main';event='push';token=$token;title=$mergeTitle;attempt='1';preflight='success';ok=$true},
  @{name='merge replay';ref='refs/heads/main';event='push';token=$token;title=$mergeTitle;attempt='2';preflight='success';ok=$false},
  @{name='wrong merge';ref='refs/heads/main';event='push';token=$token;title='Other commit';attempt='1';preflight='success';ok=$false},
  @{name='missing approval';ref='refs/heads/main';event='push';token='';title=$mergeTitle;attempt='1';preflight='success';ok=$false},
  @{name='wrong branch';ref='refs/heads/other';event='push';token=$token;title=$mergeTitle;attempt='1';preflight='success';ok=$false},
  @{name='failed CPU preflight';ref='refs/heads/main';event='push';token=$token;title=$mergeTitle;attempt='1';preflight='failure';ok=$false},
  @{name='manual invalid';ref='refs/heads/main';event='workflow_dispatch';token='bad';title='';attempt='1';preflight='success';ok=$false},
  @{name='existing pinned feature';ref='refs/heads/codex/neral-v41-context-memory-tiers-20261121';event='push';token='';title='';attempt='1';preflight='';ok=$true}
)
foreach($c in $cases){
  $env:GITHUB_REF=$c.ref
  $env:GITHUB_EVENT_NAME=$c.event
  $env:NERAL_V41_BLIND200_APPROVED=$c.token
  $env:NERAL_V41_BLIND200_MERGE_TITLE=$c.title
  $env:GITHUB_RUN_ATTEMPT=$c.attempt
  $env:NERAL_V41_BLIND200_PREFLIGHT=$c.preflight
  $accepted=$true
  try { & $guard } catch { $accepted=$false }
  if($accepted -ne $c.ok){throw ('Authorization regression: '+$c.name)}
  Write-Host ('PASS '+$c.name)
}

# New one-shot A/B50 authorization requires an exact merge title, token and CPU gate.
$env:GITHUB_REF='refs/heads/main'
$env:GITHUB_EVENT_NAME='push'
$env:GITHUB_RUN_ATTEMPT='1'
$env:NERAL_V41_BLIND200_APPROVED=''
$env:NERAL_V42_AB50_APPROVED='NERAL-V42-AB50-APPROVED'
$env:NERAL_V42_AB50_PREFLIGHT='success'
$env:NERAL_V42_AB50_MERGE_TITLE='NERAL V4.2 owner-approved A/B50 Quartz Obsidian 20261122'
& $guard
Write-Host 'PASS V42 approved launch'
$env:NERAL_V42_AB50_PREFLIGHT='failure'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 accepted failed CPU preflight'}
$env:NERAL_V42_AB50_PREFLIGHT='success'
$env:NERAL_V42_AB50_MERGE_TITLE='Unrelated push'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 accepted unrelated push'}
$env:NERAL_V42_AB50_MERGE_TITLE='NERAL V4.2 owner-approved A/B50 Quartz Obsidian 20261122'
$env:GITHUB_RUN_ATTEMPT='2'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 accepted replay'}
Write-Host 'PASS V42 negative launch guards'

# V4.2 five-case smoke: main-only, single attempt and passed CPU tests.
$env:GITHUB_REF='refs/heads/main'
$env:GITHUB_EVENT_NAME='push'
$env:GITHUB_RUN_ATTEMPT='1'
$env:NERAL_V41_BLIND200_APPROVED=''
$env:NERAL_V42_AB50_APPROVED=''
$env:NERAL_V42_SMOKE5_APPROVED='NERAL-V42-SMOKE5-APPROVED'
$env:NERAL_V42_SMOKE5_PREFLIGHT='success'
$env:NERAL_V42_SMOKE5_MERGE_TITLE='NERAL V4.2 owner-approved deadline smoke5 20261122'
& $guard
Write-Host 'PASS V42 smoke5 approved'
$env:NERAL_V42_SMOKE5_PREFLIGHT='failure'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 smoke5 accepted failed CPU preflight'}
$env:NERAL_V42_SMOKE5_PREFLIGHT='success'
$env:NERAL_V42_SMOKE5_MERGE_TITLE='Any other commit'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 smoke5 accepted unrelated commit'}
$env:NERAL_V42_SMOKE5_MERGE_TITLE='NERAL V4.2 owner-approved deadline smoke5 20261122'
$env:GITHUB_RUN_ATTEMPT='2'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 smoke5 accepted replay'}
Write-Host 'PASS V42 smoke5 approval/negative guards'

# Exact single-use 3x2 telemetry: must never authorize unrelated benchmark runs.
$env:GITHUB_REF='refs/heads/main'
$env:GITHUB_EVENT_NAME='push'
$env:GITHUB_RUN_ATTEMPT='1'
$env:NERAL_V41_BLIND200_APPROVED=''
$env:NERAL_V42_AB50_APPROVED=''
$env:NERAL_V42_SMOKE5_APPROVED=''
$env:NERAL_V42_DIAG3_APPROVED='NERAL-V42-DIAG3-APPROVED'
$env:NERAL_V42_DIAG3_PREFLIGHT='success'
$env:NERAL_V42_DIAG3_MERGE_TITLE='NERAL V4.2 owner-approved telemetry 3x2 diagnosis 20261122'
& $guard
Write-Host 'PASS V42 diag3 exact owner authorization'
$env:NERAL_V42_DIAG3_PREFLIGHT='failure'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 diag3 accepts failed CPU preflight'}
$env:NERAL_V42_DIAG3_PREFLIGHT='success'
$env:NERAL_V42_DIAG3_MERGE_TITLE='Other merge title'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 diag3 accepts unrelated commit'}
$env:NERAL_V42_DIAG3_MERGE_TITLE='NERAL V4.2 owner-approved telemetry 3x2 diagnosis 20261122'
$env:GITHUB_RUN_ATTEMPT='2'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'V42 diag3 accepts replay'}
Write-Host 'PASS V42 diag3 denial and replay'

# V4.2 explicitly approved second diagnostic: no 50-case authorization.
$env:GITHUB_REF='refs/heads/main'
$env:GITHUB_EVENT_NAME='push'
$env:GITHUB_RUN_ATTEMPT='1'
$env:NERAL_V42_DIAG3_APPROVED='NERAL-V42-DIAG3-APPROVED'
$env:NERAL_V42_DIAG3_PREFLIGHT='success'
$env:NERAL_V42_DIAG3_MERGE_TITLE='NERAL V4.2 owner-approved telemetry 3x2 bottleneck recheck 20261122'
& $guard
Write-Host 'PASS second 3x3 diagnostic one-shot authorization'
$env:GITHUB_RUN_ATTEMPT='2'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'Second diag3 accepted replay'}
$env:GITHUB_RUN_ATTEMPT='1'
$env:NERAL_V42_DIAG3_PREFLIGHT='failure'
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'Second diag3 accepted failed preflight'}
$env:NERAL_V42_DIAG3_PREFLIGHT='success'
$env:NERAL_V42_DIAG3_APPROVED=''
$denied=$false
try { & $guard } catch { $denied=$true }
if(-not $denied){throw 'Second diag3 accepted missing owner token'}
Write-Host 'PASS second 3x3 diagnostic negative/replay authorization'
