# Kill MATUDA training python processes on the training PC.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'train_matuda|run_s1_ukdale' } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "killed $($_.ProcessId)"
  }
Write-Host "done"
