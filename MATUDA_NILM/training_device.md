# Training Device Workflow

Use this workflow when coding locally in this project and running the result on
the training device.

## 1. Code Locally

Make code changes in the local workspace first.

Local project path:

```powershell
C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM
```

Before using the training device, commit the local changes so the training
device can pull the same code:

```powershell
git status
git add <changed-files>
git commit -m "your commit message"
```

If the remote repository needs the commit, push it before entering the training
device:

```powershell
git push
```

## 2. Enter Training Device

Use SSH:

```powershell
ssh raymond@100.110.55.5
```

Password:

```text
(see gitignored `training_device.secrets.md` or env `MATUDA_SSH_PASSWORD`)
```

Go to the training workspace:

```powershell
Set-Location D:\Raymond\high_low_freq_NILM
```

Pull the latest committed code:

```powershell
git pull
```

If Git reports `detected dubious ownership`, run this once on the training
device, then retry `git pull`:

```powershell
git config --global --add safe.directory D:/Raymond/high_low_freq_NILM
```

## 3. ALWAYS use this conda / Python (RTX 4090)

**Do not** use bare `python` on PATH, and **do not** use `D:\Raymond\miniconda3`
for training. SSH user is `raymond`, but the working GPU env is under user `PC`:

```text
Env name:   nilm
Python:     C:\Users\PC\anaconda3\envs\nilm\python.exe
Activate:   C:\Users\PC\anaconda3\Scripts\activate.bat  (then: conda activate nilm)
Verified:   torch 2.6.0+cu124, cuda=True, device=NVIDIA GeForce RTX 4090
```

Also CUDA-OK (optional): `C:\Users\PC\anaconda3\envs\matnilm\python.exe`

### Activate in an interactive SSH session

```powershell
& "C:\Users\PC\anaconda3\Scripts\activate.bat" nilm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Always run training / scripts with the full path

```powershell
& "C:\Users\PC\anaconda3\envs\nilm\python.exe" your_script.py
```

For plink / AI automation, **always** prefix remote Python with that full path
(never rely on `python` alone).

Extra secrets / hostkey: see `training_device.secrets.md` (gitignored).

## 4. Run Code On Training Device

Check GPU:

```powershell
nvidia-smi
```

Check GPU in the **nilm** env (required):

```powershell
& "C:\Users\PC\anaconda3\envs\nilm\python.exe" -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Run the requested script or training command from:

```powershell
D:\Raymond\high_low_freq_NILM
```

using the **nilm** python above.
## Visible Automated Run

To let the AI automate the training-device command while you watch the output,
run this from the local project:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_training_device_visible.ps1
```

This opens a visible PowerShell window, logs into the training device with
`plink`, enters `D:\Raymond\high_low_freq_NILM`, runs the remote command, and
keeps the window open after finishing.

To run a different command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_training_device_visible.ps1 -RemoteCommand "git pull; nvidia-smi"
```

## Notes For Future AI

**Always use** `C:\Users\PC\anaconda3\envs\nilm\python.exe` for any remote
training / eval / torch job on this machine (RTX 4090). Do not invent another
env path unless the user updates this file.

Always follow this order unless the user says otherwise:

1. Edit code locally.
2. Test locally if possible.
3. Commit the local changes.
4. Push if the training device pulls from the remote repository.
5. SSH / plink into the training device.
6. Enter the password (see this file / `training_device.secrets.md`).
7. `Set-Location D:\Raymond\high_low_freq_NILM`
8. `git pull`
9. Run the requested command with **`C:\Users\PC\anaconda3\envs\nilm\python.exe`**.

Do not edit code directly on the training device unless the user explicitly asks.
