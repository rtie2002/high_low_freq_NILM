$ErrorActionPreference = "Continue"
# Kill current matuda/auto jobs, sync is done from local; launch v3 + reeval v1
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='cmd.exe'" |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match 'auto_experiment|main.py --mode train_evaluate --model matuda') } |
  ForEach-Object { "KILL $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep 3

$py = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
$wd = "D:\Raymond\high_low_freq_NILM\multi_appliances_NILM"
$logDir = Join-Path $wd "runs\_auto_loop"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 1) Re-eval v1 with F1 fix
$reeval = Join-Path $logDir "reeval_v1_fixed.log"
$ckpt = Join-Path $wd "runs\ukdale_matuda_egc_h1h5_to_h2\matuda\best.pt"
$bat1 = Join-Path $logDir "reeval_v1.bat"
@"
@echo off
cd /d "$wd"
"$py" scripts\reeval_matuda_ckpt.py --checkpoint "$ckpt" --experiment config\experiment_ukdale_matuda.yaml --model-config config\models\matuda.yaml > "$reeval" 2>&1
echo EXIT %ERRORLEVEL% >> "$reeval"
"@ | Set-Content $bat1 -Encoding ASCII

# 2) Train v3 after reeval
$out = Join-Path $logDir "matuda_v3_train.log"
$bat2 = Join-Path $logDir "run_v3.bat"
@"
@echo off
cd /d "$wd"
call "$bat1"
"$py" main.py --mode train_evaluate --model matuda --experiment config/experiment_ukdale_matuda_v3.yaml --model-config config/models/matuda_v3.yaml --seed 2026 > "$out" 2>&1
echo EXIT %ERRORLEVEL% >> "$out"
"@ | Set-Content $bat2 -Encoding ASCII

$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "cmd.exe /c `"$bat2`""
  CurrentDirectory = $wd
}
"LAUNCHED return=$($res.ReturnValue) pid=$($res.ProcessId)"
"reeval_log=$reeval"
"v3_log=$out"
