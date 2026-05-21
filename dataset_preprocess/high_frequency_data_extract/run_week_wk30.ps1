# Config-driven batch: reads weeks + appliances from hf_config.yaml (default wk30).
Set-Location $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
Write-Host "Running batch from hf_config.yaml (168 FLACs per week — may take hours)."
& $Python "high_frequency_data_extract.py" --config "hf_config.yaml"
