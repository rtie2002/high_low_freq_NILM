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
RtiE2002
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

## 3. Run Code On Training Device

Check GPU:

```powershell
nvidia-smi
```

Check GPU in Python/PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Run the requested script or training command from:

```powershell
D:\Raymond\high_low_freq_NILM
```

## Notes For Future AI

Always follow this order unless the user says otherwise:

1. Edit code locally.
2. Test locally if possible.
3. Commit the local changes.
4. Push if the training device pulls from the remote repository.
5. SSH into the training device.
6. Enter the password.
7. `Set-Location D:\Raymond\high_low_freq_NILM`
8. `git pull`
9. Run the requested command on the training device.

Do not edit code directly on the training device unless the user explicitly asks.
