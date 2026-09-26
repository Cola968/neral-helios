param(
    [string]$Bundle = "",
    [int]$ContextTokens = 512
)
$ErrorActionPreference='Stop'

function Resolve-NeralRoot {
    $candidates=@($PSScriptRoot)
    $desktop=[Environment]::GetFolderPath('Desktop')
    if($desktop){$candidates += (Join-Path $desktop 'Neral-CPU-App')}
    if($env:OneDrive){$candidates += (Join-Path $env:OneDrive 'Desktop\Neral-CPU-App')}
    foreach($candidate in $candidates | Select-Object -Unique){
        if($candidate -and (Test-Path -LiteralPath (Join-Path $candidate 'qualify_frontier350.py') -PathType Leaf)){
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Neral-Ordner mit qualify_frontier350.py wurde nicht gefunden.'
}

function Test-FrontierPython {
    param([string]$Exe)
    if(-not $Exe -or -not(Test-Path -LiteralPath $Exe -PathType Leaf)){return $false}
    try{
        & $Exe -c "import torch, tokenizers; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }catch{return $false}
}

$root=Resolve-NeralRoot
Set-Location -LiteralPath $root

if(-not $Bundle){
    $Bundle=Join-Path $root 'deployment\installed_models\neral-arctus-350'
}
if(-not(Test-Path -LiteralPath $Bundle -PathType Container)){
    throw "Arctus-Deployment-Bundle fehlt: $Bundle. Importiere zuerst den SHA-geprüften 2.073B-Checkpoint mit import_frontier350.py."
}

$candidates=@(
    $env:HELIOS_FRONTIER_PYTHON,
    $env:NERAL_FRONTIER_PYTHON,
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root 'local_state\python\Scripts\python.exe')
) | Where-Object { $_ } | Select-Object -Unique

$cmd=Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if($cmd){$candidates += $cmd.Source}

$python=$null
foreach($candidate in $candidates){
    if(Test-FrontierPython $candidate){$python=(Resolve-Path -LiteralPath $candidate).Path;break}
}
if(-not $python){
    throw 'Keine Python-Umgebung mit CUDA-PyTorch, BF16 und tokenizers gefunden. Setze HELIOS_FRONTIER_PYTHON auf deine funktionierende GPU-Python-Umgebung.'
}

Write-Host "GPU-Python: $python" -ForegroundColor Cyan
try{
    & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu --format=csv,noheader
}catch{
    Write-Warning 'nvidia-smi konnte nicht gelesen werden; PyTorch-CUDA-Prüfung war trotzdem erfolgreich.'
}

& $python -X utf8 (Join-Path $root 'qualify_frontier350.py') --bundle $Bundle --context-tokens $ContextTokens
$exit=$LASTEXITCODE
if($exit -ne 0){
    throw "Arctus-Runtime-Qualifikation ist fehlgeschlagen (Exit $exit). Prüfe local_state\qualifications."
}
Write-Host 'Arctus-Runtime-Qualifikation bestanden. Bericht liegt unter local_state\qualifications.' -ForegroundColor Green
