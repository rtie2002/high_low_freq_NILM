# Opens a visible PowerShell window, runs a remote command via plink, keeps window open.
# Password/hostkey: MATUDA_NILM/training_device.secrets.md or env MATUDA_SSH_PASSWORD
param(
    [Parameter(Mandatory = $false)]
    [string]$RemoteCommand = "nvidia-smi -L"
)

$ErrorActionPreference = "Stop"
$plink = "C:\Program Files\PuTTY\plink.exe"
$hostkey = "SHA256:lX6KCWyehRD5pUp9u/aliLZ7S/BVc2A2KFohf3q4hK4"
$user = "raymond"
$hostAddr = "100.110.55.5"

$secrets = Join-Path $PSScriptRoot "..\training_device.secrets.md"
$pw = $env:MATUDA_SSH_PASSWORD
if (-not $pw -and (Test-Path $secrets)) {
    $m = Select-String -Path $secrets -Pattern '(?i)password\s*[:=]\s*(\S+)' | Select-Object -First 1
    if ($m) { $pw = $m.Matches[0].Groups[1].Value }
}
if (-not $pw) {
    Write-Error "Set MATUDA_SSH_PASSWORD or create training_device.secrets.md (gitignored)."
}

if (-not (Test-Path $plink)) {
    Write-Error "plink not found: $plink"
}

Write-Host "==== Training device visible run ====" -ForegroundColor Cyan
Write-Host "Host: $user@$hostAddr"
Write-Host "Command:"
Write-Host $RemoteCommand
Write-Host "=====================================" -ForegroundColor Cyan

& $plink -ssh "$user@$hostAddr" -pw $pw -batch -hostkey $hostkey $RemoteCommand
$code = $LASTEXITCODE

Write-Host ""
Write-Host "Exit code: $code" -ForegroundColor Yellow
Write-Host "Press Enter to close..."
[void][System.Console]::ReadLine()
exit $code
