from pathlib import Path
import subprocess

ROOT = Path(r"D:\Raymond\high_low_freq_NILM\multi_appliances_NILM")
log = ROOT / "runs" / "_hard_gate" / "train.out.log"
err = ROOT / "runs" / "_hard_gate" / "train.err.log"
log.parent.mkdir(parents=True, exist_ok=True)
py = r"C:\Users\PC\anaconda3\envs\nilm\python.exe"
cmd = [
    py,
    "main.py",
    "--mode",
    "train_evaluate",
    "--model",
    "matuda",
    "--experiment",
    "config/experiment_ukdale_matuda_v3.yaml",
    "--model-config",
    "config/models/matuda_v3.yaml",
]
# Detach via PowerShell Start-Process
ps = f'''
Set-Location "{ROOT}"
Start-Process -FilePath "{py}" -ArgumentList "main.py --mode train_evaluate --model matuda --experiment config/experiment_ukdale_matuda_v3.yaml --model-config config/models/matuda_v3.yaml" -WorkingDirectory "{ROOT}" -RedirectStandardOutput "{log}" -RedirectStandardError "{err}" -WindowStyle Hidden
Write-Host STARTED
'''
Path(ROOT / "_start_hard.ps1").write_text(ps, encoding="utf-8")
print("script written", ROOT / "_start_hard.ps1")
