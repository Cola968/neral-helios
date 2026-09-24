$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot '..'
$targets=@('Qualify-Arctus.ps1','Setup-Arctus350.ps1')
foreach($name in $targets){
  $path=Join-Path $root $name
  $tokens=$null;$errors=$null
  [System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors) | Out-Null
  if($errors.Count){throw "$name syntax failed: $($errors[0].Message)"}
}
$setup=Get-Content -LiteralPath (Join-Path $root 'Setup-Arctus350.ps1') -Raw
if($setup -notmatch 'f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7'){throw 'Continuity SHA missing from guided setup'}
if($setup -notmatch '4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'){throw 'Tokenizer SHA missing from guided setup'}
$qualify=Get-Content -LiteralPath (Join-Path $root 'Qualify-Arctus.ps1') -Raw
if($qualify -notmatch 'torch.cuda.is_available'){throw 'CUDA preflight missing'}
if($qualify -notmatch 'qualify_frontier350.py'){throw 'Qualification runner missing'}
Write-Host 'PASS: Arctus guided setup and qualification launchers parse and retain locked SHA/CUDA contracts'
