# Monitor MATUDA pipeline + auto-loop on the training device.
$pipe = "D:\Raymond\high_low_freq_NILM\multi_appliances_NILM"
"=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
"=== python jobs ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'main.py|auto_experiment|matuda|train_matuda' } |
  ForEach-Object { "PID $($_.ProcessId) $($_.CommandLine.Substring(0,[Math]::Min(140,$_.CommandLine.Length)))" }
"=== auto loop ==="
Get-Content "$pipe\runs\_auto_loop\loop.log" -Tail 12 -ErrorAction SilentlyContinue
"=== scoreboard ==="
Get-Content "$pipe\runs\_auto_loop\SCOREBOARD.md" -ErrorAction SilentlyContinue
"=== latest matuda history ==="
Get-ChildItem "$pipe\runs" -Recurse -Filter history.csv -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object {
    $_.FullName
    Import-Csv $_.FullName | Select-Object -Last 3 | Format-Table epoch,val_f1,val_mae,train_loss -AutoSize
  }
"=== latest test_metrics.csv ==="
Get-ChildItem "$pipe\runs" -Recurse -Filter test_metrics.csv -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 2 |
  ForEach-Object {
    ""
    $_.FullName
    Get-Content $_.FullName
  }
