$ErrorActionPreference = 'Stop'
$project = 'D:\Raymond\high_low_freq_NILM\Neural-adjoint inverse optimization'
$python = 'C:\Users\PC\anaconda3\envs\nilm\python.exe'
$root = Join-Path $project 'runs\paper1_e1_shared'
$sourceCheckpoint = Join-Path $project 'runs\rerun_20261111_8889037\forward\best.pt'
Set-Location -LiteralPath $project
if (Test-Path -LiteralPath $root) { throw "Refusing to overwrite: $root" }
New-Item -ItemType Directory -Path (Join-Path $root 'forward') -Force | Out-Null
Copy-Item -LiteralPath $sourceCheckpoint -Destination (Join-Path $root 'forward\best.pt')

& $python forward_model\evaluate_reconstruction.py --config forward_training\train_forward.yaml --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'reconstruction\validation') --split validation --device cuda
& $python diagnose_zero_background.py --config forward_training\train_forward.yaml --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'zero_background_diagnostic') --windows 48 --train-windows 128 --steps 150 --learning-rate 0.05 --batch-size 8
& $python diagnose_source_attribution.py $root (Join-Path $root 'source_attribution')
& $python diagnose_identifiability.py $root (Join-Path $root 'identifiability')
& $python step2_inverse_optimization.py --config forward_training\train_forward.yaml --checkpoint (Join-Path $root 'forward\best.pt') --run-dir (Join-Path $root 'inverse_validation') --split validation --steps 150 --learning-rate 0.05 --batch-size 8 --train-mean-init --device cuda
