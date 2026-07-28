# Detach the autonomous experiment loop (survives SSH disconnect)
$py = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
$wd = "D:\Raymond\high_low_freq_NILM\multi_appliances_NILM"
$logDir = Join-Path $wd "runs\_auto_loop"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$bat = Join-Path $logDir "run_auto_loop.bat"
$out = Join-Path $logDir "auto_loop_stdout.log"
@"
@echo off
cd /d "$wd"
"$py" -u scripts\auto_experiment_loop.py > "$out" 2>&1
echo EXIT %ERRORLEVEL% >> "$out"
"@ | Set-Content -Path $bat -Encoding ASCII

$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "cmd.exe /c `"$bat`""
  CurrentDirectory = $wd
}
"AUTO_LOOP return=$($res.ReturnValue) pid=$($res.ProcessId)"
"stdout=$out"
