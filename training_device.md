# Training Device Login

Use SSH:

```powershell
ssh raymond@100.110.55.5
```

Password:

```text
RtiE2002
```

Workspace:

```powershell
Set-Location D:\Raymond
```

Check GPU:

```powershell
nvidia-smi
```

Check GPU in Python/PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```
