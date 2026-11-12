$ErrorActionPreference = 'Stop'
$project = 'D:\Raymond\high_low_freq_NILM\Neural-adjoint inverse optimization'
$python = 'C:\Users\PC\anaconda3\envs\nilm\python.exe'
Set-Location -LiteralPath $project

function Run-Variant([string]$name, [string]$config) {
    $root = Join-Path $project ("runs\" + $name)
    if (Test-Path -LiteralPath $root) {
        throw "Refusing to overwrite an existing run: $root"
    }
    New-Item -ItemType Directory -Path $root -Force | Out-Null

    & $python forward_training\step1_train_forward_model.py --config $config --run-dir (Join-Path $root 'forward')
    & $python forward_model\evaluate_reconstruction.py --config $config --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'reconstruction\validation') --split validation --device cuda
    & $python diagnose_zero_background.py --config $config --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'zero_background_diagnostic') --windows 48 --train-windows 128 --steps 150 --learning-rate 0.05 --batch-size 8
    & $python diagnose_source_attribution.py $root (Join-Path $root 'source_attribution')
    & $python diagnose_identifiability.py $root (Join-Path $root 'identifiability')
    & $python step2_inverse_optimization.py --config $config --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'inverse_validation') --split validation --steps 150 --learning-rate 0.05 --batch-size 8 --train-mean-init --device cuda
}

Run-Variant 'paper1_e2_specific' 'forward_training\train_forward_e2_specific.yaml'
Run-Variant 'paper1_e3_event_balanced' 'forward_training\train_forward_e3_event_balanced.yaml'
Run-Variant 'paper1_e4_specific_event_balanced' 'forward_training\train_forward_e4_specific_event_balanced.yaml'
