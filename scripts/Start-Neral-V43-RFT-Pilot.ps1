# Local owner-operated preparation. No workflow or automatic GPU execution.
[CmdletBinding()]
param(
 [Parameter(Mandatory)][string]$Python,
 [Parameter(Mandatory)][string]$Bundle,
 [Parameter(Mandatory)][string]$ManifestSha,
 [Parameter(Mandatory)][string]$Protected,
 [Parameter(Mandatory)][string]$ProtectedSha,
 [Parameter(Mandatory)][string]$CheckpointManifest,
 [Parameter(Mandatory)][string]$CheckpointSha,
 [Parameter(Mandatory)][string]$Output,
 [switch]$RunGpu,
 [string]$Confirm=''
)
$ErrorActionPreference='Stop'
$pin='5877b73851c986f39f4dbeb470cd8fb08bd2707d'
$approval='V43-RFT-QLORA-SMOKE-ONCE'
if ($RunGpu -and $Confirm -cne $approval) { throw 'NEW_EXPLICIT_GPU_APPROVAL_REQUIRED' }
if (-not $RunGpu -and $Confirm) { throw 'GPU_TOKEN_NOT_ALLOWED_FOR_CPU_CHECK' }
foreach ($value in @($ManifestSha,$ProtectedSha,$CheckpointSha)) {
 if ($value -cnotmatch '^[0-9a-f]{64}$') { throw 'PIN_REQUIRED' }
}
$private=Join-Path $env:LOCALAPPDATA 'NERAL-private/v43-rft'
$root=[IO.Path]::GetFullPath($private)
$out=[IO.Path]::GetFullPath($Output)
if (-not $out.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'NEW_RFT_OUTPUT_REQUIRED' }
if (Test-Path -LiteralPath $out) { throw 'ONE_SHOT_OUTPUT_EXISTS' }
$baseArgs=@('-B','-m','scripts.neral_v43_rft_train','--bundle',$Bundle,'--manifest-sha',$ManifestSha,'--protected',$Protected,'--protected-sha',$ProtectedSha,'--checkpoint-manifest',$CheckpointManifest,'--checkpoint-sha',$CheckpointSha)
function Invoke-LocalPhase([string]$Phase,[string[]]$Arguments) {
 # Native warnings are not process failures. Never upload raw logs.
 $quoted=@($Arguments | ForEach-Object {
   if ($_ -match '["\r\n]') { throw 'UNSAFE_ARGUMENT' }
   '"'+($_ -replace '(\\+)$','$1$1')+'"'
 }) -join ' '
 $stdout=Join-Path $scratch ($Phase+'-stdout.log')
 $stderr=Join-Path $scratch ($Phase+'-stderr.log')
 if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'PHASE_LOG_ALREADY_EXISTS' }
 $process=Start-Process -FilePath $Python -ArgumentList $quoted -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
 if ($null -eq $process.ExitCode) { throw ('RFT_'+$Phase+'_EXIT_UNAVAILABLE') }
 if ($process.ExitCode -ne 0) { throw ('RFT_'+$Phase+'_EXIT_'+$process.ExitCode) }
}
# CPU validation before owner access; scratch is private, fresh, separate.
$scratch=Join-Path $root ('preflight-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch -ErrorAction Stop | Out-Null
$oldCuda=$env:CUDA_VISIBLE_DEVICES
try {
 $env:CUDA_VISIBLE_DEVICES='-1'
 Invoke-LocalPhase 'cpu' ($baseArgs+@('--check-tokenizer'))
} finally { $env:CUDA_VISIBLE_DEVICES=$oldCuda }
Write-Host 'RFT_CPU_PREFLIGHT_OK'
if (-not $RunGpu) { return }
# Future one-step pilot only. The current task authorizes no execution of this block.
if ($env:COMPUTERNAME -cne 'W39-GAMINGPC') { throw 'WRONG_COMPUTER' }
& git merge-base --is-ancestor $pin HEAD
if ($LASTEXITCODE -ne 0 -or @(git status --porcelain --untracked-files=no).Count -ne 0) { throw 'UNVERIFIED_CHECKOUT' }
# Reuse only named owner helpers from the exact already tested baseline.
$source=@(git show "${pin}:scripts/neral_v43_reference50_raw_owner.ps1") -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'PINNED_OWNER_SOURCE_MISSING' }
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors)
if ($errors.Count -ne 0) { throw 'OWNER_SOURCE_PARSE_FAILED' }
$names=@('Get-Owner','Stop-ConfirmedOwner','Check-RawCapacityAndForeign','Restore-OriginalOwner')
$helpers=@($ast.EndBlock.Statements | Where-Object { $_ -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -in $names })
if ($helpers.Count -ne 4 -or @($helpers.Name | Sort-Object -Unique).Count -ne 4) { throw 'OWNER_HELPERS_MISSING' }
$helperFile=Join-Path $scratch 'owner-helpers.ps1'
$helpers | ForEach-Object { $_.Extent.Text } | Set-Content -LiteralPath $helperFile -Encoding UTF8
. $helperFile
$taskName='NERAL-PR558-ThreeTier-909d1419c9ff'
function HttpReady {
 $code=& curl.exe --noproxy '*' --silent --connect-timeout 3 --max-time 8 --output NUL --write-out '%{http_code}' 'http://127.0.0.1:9690/health'
 return ($LASTEXITCODE -eq 0 -and $code -ceq '200')
}
function RequireOwnerTask {
 $service=Join-Path $env:LOCALAPPDATA 'NERAL/three-tier-pr558-909d1419c9ff/NERAL-1.0-Local/Start-Neral-Three-Tier-Service.ps1'
 $scheduled=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
 if (@($scheduled.Actions).Count -ne 1 -or [string]$scheduled.Actions[0].Arguments -notlike ('*'+$service+'*')) { throw 'ORIGINAL_TASK_IDENTITY_MISMATCH' }
 return $scheduled
}
if (-not (HttpReady) -or (RequireOwnerTask).State -ne 'Running') { throw 'ORIGINAL_OWNER_NOT_READY' }
$owner=Get-Owner
Invoke-LocalPhase 'owner-idle' @('-B','-m','scripts.neral_v42_owner_availability')
$stopRequested=$false
try {
 $stopRequested=$true
 Stop-ConfirmedOwner $owner
 Check-RawCapacityAndForeign
 $bundleManifest=Join-Path $Bundle 'manifest.json'
 if ((Get-FileHash -LiteralPath $bundleManifest -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ManifestSha) { throw 'TRAIN_MANIFEST_CHANGED' }
 $train=Get-Content -LiteralPath $bundleManifest -Raw | ConvertFrom-Json
 $receipt=Join-Path $scratch 'owner-receipt.json'
 $record=@{approval=$approval;original_http_status=200;stop_owned=$true;checkpoint_manifest_sha256=$CheckpointSha;train_sha256=$train.train_sha256} | ConvertTo-Json -Compress
 # UTF8 without BOM, Python JSON reader expects UTF8.
 [IO.File]::WriteAllText($receipt,$record,(New-Object System.Text.UTF8Encoding($false)))
 $receiptSha=(Get-FileHash -LiteralPath $receipt -Algorithm SHA256).Hash.ToLowerInvariant()
 Invoke-LocalPhase 'smoke' ($baseArgs+@('--run-gpu','--confirm',$Confirm,'--owner-receipt',$receipt,'--owner-receipt-sha',$receiptSha,'--out',$out))
} finally {
 if ($stopRequested -and -not (HttpReady)) {
   if ((RequireOwnerTask).State -ne 'Ready') { throw 'ROLLBACK_OWNER_STATE_UNEXPECTED' }
   Check-RawCapacityAndForeign
   Restore-OriginalOwner
 }
 if (-not (HttpReady) -or (RequireOwnerTask).State -ne 'Running') { throw 'ROLLBACK_HTTP200_UNVERIFIED' }
 $null=Get-Owner
 Write-Host 'ORIGINAL_NERAL_ROLLBACK_HTTP200'
 # Separate CPU seal validation even after training failure.
 try {
  $env:CUDA_VISIBLE_DEVICES='-1'
  Invoke-LocalPhase 'post-integrity' $baseArgs
 } finally { $env:CUDA_VISIBLE_DEVICES=$oldCuda }
}
Write-Host 'RFT_SMOKE_COMPLETED_NOT_PROMOTED'
