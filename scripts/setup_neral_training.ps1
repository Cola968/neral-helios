# Setup only: no model download, training, inference or global CUDA changes.
param([Parameter(Mandatory=$true)][string]$VenvDir)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$target = [IO.Path]::GetFullPath($VenvDir).TrimEnd('\')
if ((Test-Path -LiteralPath $target) -or $target.StartsWith($repo.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase) -or $target -eq $repo) {
    throw 'Choose a NEW directory outside the repository and the working inference environment.'
}
function Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
Checked 'py' @('-3.12', '-m', 'venv', $target)
$python = Join-Path $target 'Scripts\python.exe'
Checked $python @('-m', 'pip', 'install', 'torch==2.8.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $repo 'requirements-neral-training.txt'))
Checked $python @('-m', 'pip', 'check')
Checked $python @((Join-Path $repo 'scripts\neral_training_preflight.py'))
# No activation: every command explicitly uses this new interpreter.
Write-Output "Prepared $python. CUDA check remains manual; training requires separate approval."
