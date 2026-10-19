$ErrorActionPreference = "Stop"

$projectRoot = "D:\Raymond\high_low_freq_NILM\multi_appliances_NILM"
$pythonExe = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
$logDir = Join-Path $projectRoot "runs\_logs"
$logPath = Join-Path $logDir "unetnilm_paper_3seed.log"
$exitPath = Join-Path $logDir "unetnilm_paper_3seed.exitcode"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Set-Location -LiteralPath $projectRoot

try {
    & $pythonExe "scripts\run_mart_nilm_paper_tables.py" `
        --table v `
        --models unetnilm `
        --seeds 2026 2027 2028 *>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "UNet-NILM experiment command exited with code $LASTEXITCODE"
    }

    & $pythonExe "scripts\benchmark_paper_efficiency.py" `
        --models mart_nilm unetnilm `
        --seeds 2026 2027 2028 *>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Efficiency benchmark exited with code $LASTEXITCODE"
    }

    & $pythonExe "scripts\summarize_mart_nilm_paper_tables.py" `
        --tables v vi `
        --models mart_nilm unetnilm `
        --seeds 2026 2027 2028 *>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Summary command exited with code $LASTEXITCODE"
    }
    Set-Content -LiteralPath $exitPath -Value "0"
}
catch {
    $_ | Out-String | Tee-Object -FilePath $logPath -Append
    Set-Content -LiteralPath $exitPath -Value "1"
    throw
}
