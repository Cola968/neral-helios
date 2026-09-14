param([switch]$NoBrowser)
$ErrorActionPreference='Stop'

$keyFile=Join-Path $PSScriptRoot 'local_state\explabs-key.dpapi'
if(Test-Path -LiteralPath $keyFile -PathType Leaf){
    try{
        $encrypted=(Get-Content -LiteralPath $keyFile -Raw).Trim()
        $secure=$encrypted | ConvertTo-SecureString
        $ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try{
            $env:EXPLABS_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        }finally{
            if($ptr -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)}
        }
    }catch{
        throw 'Der gespeicherte Experiential-Key konnte fuer dieses Windows-Benutzerkonto nicht entschluesselt werden. Fuehre Configure Opus.cmd erneut aus.'
    }
}

& (Join-Path $PSScriptRoot 'Start-Neral.ps1') -NoBrowser:$NoBrowser
