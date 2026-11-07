# Read-only Windows system security status for NERAL Cyber Guard.
# No SSID, IP/MAC address, DNS history, process list, password, or log is collected.
$ErrorActionPreference = 'Stop'

$firewall = @(
    Get-NetFirewallProfile -PolicyStore ActiveStore | ForEach-Object {
        [PSCustomObject]@{
            name = [string]$_.Name
            enabled = (([string]$_.Enabled) -eq 'True')
            inbound = [string]$_.DefaultInboundAction
        }
    }
)

$networks = @(
    Get-NetConnectionProfile | ForEach-Object {
        [string]$_.NetworkCategory
    }
)

$defender = $null
try {
    $state = Get-MpComputerStatus -ErrorAction Stop
    $defender = [PSCustomObject]@{
        antivirus_enabled = (([string]$state.AntivirusEnabled) -eq 'True')
        realtime_enabled = (([string]$state.RealTimeProtectionEnabled) -eq 'True')
    }
} catch {
    # Microsoft Defender may not be installed/available if another AV is in use.
    # This is unknown, never equivalent to "antivirus disabled".
    $defender = $null
}

[PSCustomObject]@{
    schema = 1
    source = 'windows-local-readonly'
    firewall_profiles = @($firewall)
    network_categories = @($networks)
    defender = $defender
} | ConvertTo-Json -Depth 5 -Compress
