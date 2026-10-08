# Codex Instructions for This Repository

## Remote training machine

Read [`training_device.md`](training_device.md) before running remote tests,
evaluation, or training.

Current connection facts:

- Tailscale host: `100.110.55.5`
- SSH user: `PC`
- Local key path: `%USERPROFILE%\.ssh\codex_nilm_ed25519`
- Remote repository: `D:\Raymond\high_low_freq_NILM`
- Multi-appliance project: `D:\Raymond\high_low_freq_NILM\multi_appliances_NILM`
- Required Python: `C:\Users\PC\anaconda3\envs\nilm\python.exe`

Never store a password, private-key contents, access token, or other secret in
the repository. Use key-based SSH and never use bare `python` for remote NILM
jobs.

For each approved remote experiment: edit locally, run a narrow check, commit
only task-related files, push, connect by SSH, use `git pull --ff-only`, run the
remote check/job with the required Python, inspect the generated metrics and
waveforms, and record the result in the relevant experiment log. Preserve all
unrelated local and remote changes and untracked files.

For training that must survive SSH disconnection, use a Windows Scheduled Task
as described in `training_device.md`; an ordinary remote `Start-Process` job may
terminate when the SSH session closes.
