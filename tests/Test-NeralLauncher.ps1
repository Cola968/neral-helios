# Offline regression checks: load function definitions without starting the app.
param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
$source=Join-Path $PSScriptRoot '..\Start-Neral.ps1'
$tokens=$null;$parseErrors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($source,[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw 'Launcher syntax failed'}
foreach($definition in $ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true)){
    Invoke-Expression $definition.Extent.Text
}
$script:existingPython=(Resolve-Path -LiteralPath $Python).Path
if(-not(Test-Python311 $existingPython)){throw 'Test interpreter must actually run Python 3.11+'}
$state=Join-Path $PSScriptRoot '..\.test-temp-launcher with spaces'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$savedPython=$env:NERAL_PYTHON;$env:NERAL_PYTHON=''
try{
    # Simulate an interpreter registered outside the requested package folder.
    function Get-ChildItem {param($LiteralPath,$ErrorAction) if($LiteralPath -eq 'HKCU:\Software\Python\PythonCore'){[pscustomobject]@{PSPath='HKCU:\Software\Python\PythonCore\3.12'}}}
    function Get-ItemProperty {param($LiteralPath,$ErrorAction) [pscustomobject]@{ExecutablePath=$script:existingPython}}
    if((Find-SystemPython).Exe -ne $existingPython){throw 'Registered custom interpreter was missed'}
    Remove-Item Function:\Get-ChildItem,Function:\Get-ItemProperty

    # Reproduce maintenance-mode exit 0 with no executable at TargetDir.
    function Invoke-WebRequest {param($Uri,$OutFile,[switch]$UseBasicParsing) Set-Content -LiteralPath $OutFile -Value 'fixture'}
    function Get-AuthenticodeSignature {param($LiteralPath) [pscustomobject]@{Status='Valid'}}
    function Start-Process {
        param($FilePath,$ArgumentList,$WindowStyle,[switch]$Wait,[switch]$PassThru)
        if(-not($ArgumentList | Where-Object {$_ -match '^TargetDir=".+ with spaces.+"$'})){throw 'Target directory was not quoted'}
        [pscustomobject]@{ExitCode=0}
    }
    function Find-SystemPython {[pscustomobject]@{Exe=$script:existingPython;Prefix=@()}}
    if((Install-NeralPython).Exe -ne $existingPython){throw 'Successful maintenance install was reported as failure'}
    Write-Host 'PASS: real Python version probe; custom registered path; installer exit 0 recovery; TargetDir with spaces'
}finally{$env:NERAL_PYTHON=$savedPython}
