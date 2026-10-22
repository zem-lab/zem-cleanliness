param(
  [Parameter(Mandatory = $true)][string]$ProgramPath,
  [string]$RemoteAddress = '192.168.0.15',
  [string]$ResultPath = ''
)
$ErrorActionPreference = 'Stop'
$resolvedProgram = (Resolve-Path -LiteralPath $ProgramPath).Path
if ((Split-Path -Leaf $resolvedProgram) -ne 'zem-cleanliness-backend.exe') {
  throw 'Expected the installed zem-cleanliness-backend.exe'
}
[void][System.Net.IPAddress]::Parse($RemoteAddress)
$ruleName = 'ZEM-Cleanliness-SEM-UDP-Replies'
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
  Set-NetFirewallRule -Name $ruleName -Enabled True -Direction Inbound -Action Allow -Profile Any -Program $resolvedProgram -Protocol UDP -RemoteAddress $RemoteAddress | Out-Null
} else {
  New-NetFirewallRule -Name $ruleName -DisplayName 'ZEM Cleanliness - SEM UDP replies' -Enabled True -Direction Inbound -Action Allow -Profile Any -Program $resolvedProgram -Protocol UDP -RemoteAddress $RemoteAddress | Out-Null
}
$result = [pscustomobject]@{ rule=$ruleName; program=$resolvedProgram; remoteAddress=$RemoteAddress; protocol='UDP'; direction='Inbound'; action='Allow' }
if ($ResultPath) { $result | ConvertTo-Json | Set-Content -LiteralPath $ResultPath -Encoding UTF8 }
$result | ConvertTo-Json
