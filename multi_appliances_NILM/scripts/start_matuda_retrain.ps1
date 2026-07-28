$ErrorActionPreference = "Continue"
Set-Location D:\Raymond\high_low_freq_NILM\multi_appliances_NILM
New-Item -ItemType Directory -Force -Path runs\_matuda | Out-Null
$out = "runs\_matuda\train.out.log"
$err = "runs\_matuda\train.err.log"
$py = "C:\Users\PC\anaconda3\envs\nilm\python.exe"

Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | ForEach-Object {
  if ($_.CommandLine -match "auto_v4|main.py.*matuda|hard_gate") {
    Write-Host ("Stopping PID " + $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
}
Start-Sleep -Seconds 2

# Remove obsolete versioned configs on the training PC
@(
  "config\models\matuda_v2.yaml",
  "config\models\matuda_v3.yaml",
  "config\models\matuda_v4.yaml",
  "config\models\matuda_v4_source_only.yaml",
  "config\models\matuda_source_only.yaml",
  "config\models\matuda_global_uda.yaml",
  "config\experiment_ukdale_matuda_v2.yaml",
  "config\experiment_ukdale_matuda_v3.yaml",
  "config\experiment_ukdale_matuda_v4.yaml",
  "config\experiment_ukdale_matuda_v4_so.yaml"
) | ForEach-Object {
  if (Test-Path $_) { Remove-Item -Force $_; Write-Host "deleted $_" }
}

Remove-Item -Recurse -Force adapters\__pycache__, model\__pycache__ -ErrorAction SilentlyContinue
$arg = "main.py --mode train_evaluate --model matuda --experiment config/experiment_ukdale_matuda.yaml --model-config config/models/matuda.yaml"
Start-Process -FilePath $py -ArgumentList $arg -WorkingDirectory (Get-Location) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
Write-Host "STARTED single matuda (hard gate) retrain"
Start-Sleep -Seconds 6
Get-Content $out -Tail 30 -ErrorAction SilentlyContinue
Get-Content $err -Tail 10 -ErrorAction SilentlyContinue
