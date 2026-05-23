param(
    [string]$RemoteCommand = "git pull; powershell -NoProfile -ExecutionPolicy Bypass -File .\hello_training_device.ps1"
)

$HostKey = "SHA256:lX6KCWyehRD5pUp9u/aliLZ7S/BVc2A2KFohf3q4hK4"
$UserHost = "raymond@100.110.55.5"
$Workspace = "D:\Raymond\high_low_freq_NILM"

$Command = @"
Write-Host '=== Training device visible runner ==='
Write-Host 'Remote: $UserHost'
Write-Host 'Workspace: $Workspace'
Write-Host 'Command: $RemoteCommand'
Write-Host ''
Write-Host 'Enter the training-device password when plink asks for it.'
Write-Host ''
plink -hostkey "$HostKey" $UserHost "Set-Location $Workspace; $RemoteCommand"
Write-Host ''
Write-Host '=== Finished. Press Enter to close this window. ==='
Read-Host
"@

Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-NoExit",
    "-Command",
    $Command
)
