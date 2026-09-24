param()
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms

$root=$PSScriptRoot
if(-not(Test-Path -LiteralPath (Join-Path $root 'import_frontier350.py') -PathType Leaf)){
    throw 'import_frontier350.py fehlt im Neral-Ordner.'
}

function Test-FrontierPython {
    param([string]$Exe)
    if(-not $Exe -or -not(Test-Path -LiteralPath $Exe -PathType Leaf)){return $false}
    try{
        & $Exe -c "import torch, tokenizers; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }catch{return $false}
}

function Pick-File {
    param([string]$Title,[string]$Filter)
    $dialog=New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title=$Title
    $dialog.Filter=$Filter
    $dialog.CheckFileExists=$true
    $dialog.Multiselect=$false
    if($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK){throw 'Auswahl abgebrochen.'}
    return $dialog.FileName
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
    throw 'Keine CUDA/BF16-Python-Umgebung mit tokenizers gefunden. Setze HELIOS_FRONTIER_PYTHON auf deine laufende GPU-Python-Umgebung.'
}

$bundle=Join-Path $root 'deployment\installed_models\neral-arctus-350'
if(-not(Test-Path -LiteralPath $bundle -PathType Container)){
    [System.Windows.Forms.MessageBox]::Show('Wähle zuerst den bekannten 2.073B-Arctus-Checkpoint aus. Danach wählst du die zugehörige tokenizer.json.','Arctus 350M einrichten',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
    $checkpoint=Pick-File '2.073B Arctus-Checkpoint auswählen' 'PyTorch checkpoint (*.pt)|*.pt|Alle Dateien (*.*)|*.*'
    $tokenizer=Pick-File 'Arctus tokenizer.json auswählen' 'Tokenizer (tokenizer.json)|tokenizer.json|JSON (*.json)|*.json'
    $sourceSha='f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7'
    $tokenizerSha='4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'
    Write-Host 'Prüfe und bereinige den historischen Arctus-Checkpoint...' -ForegroundColor Cyan
    $importArgs=@('-X','utf8',(Join-Path $root 'import_frontier350.py'),'--source',$checkpoint,'--sha256',$sourceSha,'--tokenizer',$tokenizer,'--tokenizer-sha256',$tokenizerSha,'--id','neral-arctus-350','--label','Arctus 350M · 2.073B','--assets',(Join-Path $root 'deployment'))
    & $python @importArgs
    if($LASTEXITCODE -ne 0){throw 'Arctus-Import wurde abgelehnt. Der Quellcheckpoint wurde nicht verändert.'}
}

$env:HELIOS_FRONTIER_PYTHON=$python
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'Qualify-Arctus.ps1') -Bundle $bundle
if($LASTEXITCODE -ne 0){throw 'Arctus-Qualifikation fehlgeschlagen.'}

[System.Windows.Forms.MessageBox]::Show('Arctus 350M ist importiert und die technische GPU-Qualifikation wurde ausgeführt. Öffne local_state\qualifications für den Bericht.','Arctus 350M',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
