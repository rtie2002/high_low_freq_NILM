$ErrorActionPreference = "Continue"
Set-Location D:\Raymond\high_low_freq_NILM\multi_appliances_NILM
New-Item -ItemType Directory -Force -Path runs\_hard_gate | Out-Null
$out = "runs\_hard_gate\train.out.log"
$err = "runs\_hard_gate\train.err.log"
$py = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
# Stop previous auto_v4 / matuda trains so this hard-gate run owns the GPU
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | ForEach-Object {
  if ($_.CommandLine -match "auto_v4_until_f1|main.py.*matuda") {
    Write-Host ("Stopping PID " + $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
}
Start-Sleep -Seconds 2
Remove-Item -Recurse -Force adapters\__pycache__, model\__pycache__ -ErrorAction SilentlyContinue
$args = "main.py --mode train_evaluate --model matuda --experiment config/experiment_ukdale_matuda_v3.yaml --model-config config/models/matuda_v3.yaml"
Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory (Get-Location) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
Write-Host "STARTED hard-gate matuda v3"
Start-Sleep -Seconds 5
Get-Content $out -Tail 25 -ErrorAction SilentlyContinue
Get-Content $err -Tail 15 -ErrorAction SilentlyContinue
