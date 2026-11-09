$ErrorActionPreference = 'Stop'
Set-Location 'D:\Raymond\high_low_freq_NILM\multi_appliances_NILM'
$logDir = 'D:\Raymond\high_low_freq_NILM\multi_appliances_NILM\runs\_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pythonExe = 'C:\Users\PC\anaconda3\envs\nilm\python.exe'
& $pythonExe -u scripts\run_mart_reviewer_controls.py 1> "$logDir\reviewer_controls_20261109.stdout.log" 2> "$logDir\reviewer_controls_20261109.stderr.log"
$trainingExit = $LASTEXITCODE
if ($trainingExit -eq 0) {
    $reference = 'runs\paper_5w\table_ix\no_relative_energy'
    $control = 'runs\paper_5w\reviewer_controls\no_relation'
    $predictions = @(Get-ChildItem -LiteralPath $reference,$control -Filter predictions.npz -Recurse | Select-Object -ExpandProperty FullName)
    if ($predictions.Count -eq 12) {
        & $pythonExe scripts\audit_mart_concurrent_states.py --predictions @predictions --output runs\paper_5w\reviewer_controls\concurrent_states.csv 1>> "$logDir\reviewer_controls_20261109.stdout.log" 2>> "$logDir\reviewer_controls_20261109.stderr.log"
        $trainingExit = $LASTEXITCODE
    } else {
        Write-Output "Expected 12 frozen test prediction files, found $($predictions.Count); concurrent audit not executed." | Out-File "$logDir\reviewer_controls_20261109.stderr.log" -Append
        $trainingExit = 2
    }
}
$trainingExit | Out-File "$logDir\reviewer_controls_20261109.exit.txt"
exit $trainingExit
