# Run the paper-faithful TransferNILM baseline as a durable Windows task.
$ErrorActionPreference = "Continue"

$projectRoot = "D:\Raymond\high_low_freq_NILM\multi_appliances_NILM"
$pythonExe = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
$logDir = Join-Path $projectRoot "runs\_logs"
$stdoutPath = Join-Path $logDir "transfernilm_paper_3seed.stdout.log"
$stderrPath = Join-Path $logDir "transfernilm_paper_3seed.stderr.log"
$exitPath = Join-Path $logDir "transfernilm_paper_3seed.exit.txt"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Set-Location -LiteralPath $projectRoot

try {
    & $pythonExe "scripts\run_mart_nilm_paper_tables.py" `
        --table v `
        --models transfer_multi_appliance `
        --seeds 2026 2027 2028 1>> $stdoutPath 2>> $stderrPath
    if ($LASTEXITCODE -ne 0) {
        throw "TransferNILM experiment command exited with code $LASTEXITCODE"
    }

    & $pythonExe "scripts\benchmark_paper_efficiency.py" `
        --models mart_nilm unetnilm mat_nilm transfer_multi_appliance `
        --seeds 2026 2027 2028 1>> $stdoutPath 2>> $stderrPath
    if ($LASTEXITCODE -ne 0) {
        throw "Efficiency benchmark exited with code $LASTEXITCODE"
    }

    & $pythonExe "scripts\summarize_mart_nilm_paper_tables.py" `
        --tables v vi `
        --models mart_nilm unetnilm mat_nilm transfer_multi_appliance `
        --seeds 2026 2027 2028 1>> $stdoutPath 2>> $stderrPath
    if ($LASTEXITCODE -ne 0) {
        throw "Summary command exited with code $LASTEXITCODE"
    }
    Set-Content -LiteralPath $exitPath -Value "0"
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrPath
    Set-Content -LiteralPath $exitPath -Value "1"
    throw
}
