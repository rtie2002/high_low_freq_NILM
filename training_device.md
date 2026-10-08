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

## 2. Connect to the Training Device

The two machines communicate through Tailscale. Confirm that Tailscale is
connected locally before diagnosing SSH failures. Use key-based SSH:

```powershell
$key = "$env:USERPROFILE\.ssh\codex_nilm_ed25519"
ssh -i $key -o BatchMode=yes PC@100.110.55.5
```

Connection details:

- Host: `100.110.55.5`
- User: `PC`
- Remote Windows host name: `DESKTOP-5BRNFTF`
- Authentication: the public key is installed in
  `C:\ProgramData\ssh\administrators_authorized_keys`

Do not add the private key or a password to this repository. If the expected
private key is missing, ask the user to restore or authorize a key rather than
falling back to a stored password.

Go to the training workspace:

```powershell
Set-Location D:\Raymond\high_low_freq_NILM
```

Pull the latest committed code:

```powershell
git pull --ff-only
```

If Git reports `detected dubious ownership`, run this once on the training
device, then retry `git pull`:

```powershell
git config --global --add safe.directory D:/Raymond/high_low_freq_NILM
```

## 3. ALWAYS use this conda / Python (RTX 4090)

**Do not** use bare `python` on PATH, and **do not** use `D:\Raymond\miniconda3`
for training. The SSH user and working GPU environment are under user `PC`:

```text
Env name:   nilm
Python:     C:\Users\PC\anaconda3\envs\nilm\python.exe
Activate:   & "C:\Users\PC\anaconda3\Scripts\activate.bat" nilm
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

For AI automation, **always** use that full Python path (never bare `python`).

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

## 5. Run a Job That Survives SSH Disconnection

Do not rely on a plain remote `Start-Process` for a long training run. On this
machine its child process can be terminated when the SSH session closes. Create
a PowerShell launcher under `runs\_logs`, then register and start a Windows
Scheduled Task. The launcher should:

- change to `D:\Raymond\high_low_freq_NILM\multi_appliances_NILM`;
- invoke `C:\Users\PC\anaconda3\envs\nilm\python.exe`;
- redirect standard output and error to distinct files under `runs\_logs`;
- write the process exit code to an `.exit.txt` file.

Use a unique, descriptive task name for each experiment. Monitor it through the
task state, exit-code file, log tail, GPU process, and run directory. After a
terminal state has been recorded, unregister only that specific task. Never
delete its logs or experiment outputs automatically.

Example monitoring commands inside the remote PowerShell session:

```powershell
Get-ScheduledTask -TaskName <task-name>
Get-Content runs\_logs\<experiment>.stdout.log -Tail 30
Get-Content runs\_logs\<experiment>.stderr.log -Tail 30
Get-Content runs\_logs\<experiment>.exit.txt
nvidia-smi
```

## 6. Inspect and Record Results

Do not judge an experiment only from the live loss plot. At minimum inspect:

- `validation_metrics.csv` for model selection;
- each test house's `metrics.csv` for report-only generalization;
- fridge and microwave FPR, AP, F1, ON MAE, and OFF MAE;
- representative waveform plots, especially noisy-background failures;
- the selected checkpoint epoch and process exit code.

Use validation results to select configurations. Do not select a configuration
because it performs better on UK-DALE house 2 or REFIT house 20. Record the
controlled change, result paths, metrics, conclusion, and remaining failure in
the experiment log.

## Notes For Future Codex Sessions

**Always use** `C:\Users\PC\anaconda3\envs\nilm\python.exe` for any remote
training / eval / torch job on this machine (RTX 4090). Do not invent another
env path unless the user updates this file.

Always follow this order unless the user says otherwise:

1. Edit code locally.
2. Test locally if possible.
3. Commit the local changes.
4. Push if the training device pulls from the remote repository.
5. Connect by key-based SSH as user `PC`.
6. Do not expose or store authentication secrets.
7. `Set-Location D:\Raymond\high_low_freq_NILM`
8. `git pull --ff-only`
9. Run the requested command with **`C:\Users\PC\anaconda3\envs\nilm\python.exe`**.
10. Read metrics and waveform outputs and update the experiment log.

Do not edit code directly on the training device unless the user explicitly asks.
