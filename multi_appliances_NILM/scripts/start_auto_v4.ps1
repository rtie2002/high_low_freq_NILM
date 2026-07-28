$ErrorActionPreference = "Continue"
Set-Location D:\Raymond\high_low_freq_NILM\multi_appliances_NILM
$logDir = "runs\_auto_v4"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$out = Join-Path $logDir "train_loop.out.log"
$err = Join-Path $logDir "train_loop.err.log"
$py = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
Remove-Item -Recurse -Force adapters\__pycache__, model\__pycache__ -ErrorAction SilentlyContinue
Start-Process -FilePath $py -ArgumentList "scripts\auto_v4_until_f1.py" -WorkingDirectory (Get-Location) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
Write-Host "STARTED auto_v4_until_f1"
Write-Host "stdout: $out"
Start-Sleep -Seconds 3
Get-Content $out -Tail 20 -ErrorAction SilentlyContinue
Get-Content $err -Tail 20 -ErrorAction SilentlyContinue
