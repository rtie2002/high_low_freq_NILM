<#
.SYNOPSIS
    One-click launcher for the Neural-Adjoint NILM pipeline.

.DESCRIPTION
    1. Commits any uncommitted local changes.
    2. Pushes to the remote Git repository.
    3. SSH into the training machine (RTX 4090 @ 100.110.55.5).
    4. Pulls the latest code.
    5. Registers a Windows Scheduled Task that runs pipeline.py.
    6. Streams stdout / stderr logs back to this terminal in real time.
    7. Prints the final metrics summary when the task finishes.

.PARAMETER Smoke
    Run a quick smoke test (1 epoch, 2 windows, 2 inverse steps).

.PARAMETER RunDir
    Custom output directory on the REMOTE machine.
    Default: D:\Raymond\high_low_freq_NILM\Neural-adjoint inverse optimization\runs\pipeline_<timestamp>

.PARAMETER Overwrite
    Pass --overwrite to pipeline.py (allow reusing an existing run-dir).

.EXAMPLE
    # Full pipeline (double-click or press the IDE Run button):
    .\run_pipeline.ps1

    # Quick smoke test:
    .\run_pipeline.ps1 -Smoke

    # Custom run directory:
    .\run_pipeline.ps1 -RunDir "D:\Raymond\...\runs\my_experiment"
#>

param(
    [switch]$Smoke,
    [string]$RunDir   = "",
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ────────────────────────────────────────────────────────────
function Write-Banner([string]$Title, [string]$Colour = "Cyan") {
    $line = "─" * 72
    Write-Host ""
    Write-Host $line           -ForegroundColor $Colour
    Write-Host "  $Title"     -ForegroundColor $Colour
    Write-Host $line           -ForegroundColor $Colour
}
function Write-Ok([string]$Msg)   { Write-Host "  ✓  $Msg" -ForegroundColor Green  }
function Write-Info([string]$Msg) { Write-Host "  $Msg"                             }
function Write-Warn([string]$Msg) { Write-Host "  ⚠  $Msg" -ForegroundColor Yellow }
function Write-Fail([string]$Msg) { Write-Host "  ✗  $Msg" -ForegroundColor Red; exit 1 }

# ── Constants ─────────────────────────────────────────────────────────────────
$LocalRoot   = "C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM"
$RemoteHost  = "100.110.55.5"
$RemoteUser  = "PC"
$SshKey      = "$env:USERPROFILE\.ssh\codex_nilm_ed25519"
$RemoteRoot  = "D:\Raymond\high_low_freq_NILM"
$RemoteProj  = "$RemoteRoot\Neural-adjoint inverse optimization"
$RemotePy    = "C:\Users\PC\anaconda3\envs\nilm\python.exe"
$LogDir      = "$RemoteRoot\runs\_logs"
$Timestamp   = (Get-Date -Format "yyyyMMdd_HHmmss")
$TaskName    = "NeuralAdjointPipeline_$Timestamp"

# ── Resolve pipeline arguments ────────────────────────────────────────────────
$PipelineArgs = ""
if ($Smoke)     { $PipelineArgs += " --smoke" }
if ($Overwrite) { $PipelineArgs += " --overwrite" }
if ($RunDir -ne "") {
    $PipelineArgs += " --run-dir `"$RunDir`""
} else {
    $suffix   = if ($Smoke) { "_smoke" } else { "" }
    $RunDir   = "$RemoteProj\runs\pipeline_$Timestamp$suffix"
    $PipelineArgs += " --run-dir `"$RunDir`""
}

$StdoutLog = "$LogDir\$TaskName.stdout.log"
$StderrLog = "$LogDir\$TaskName.stderr.log"
$ExitFile  = "$LogDir\$TaskName.exit.txt"

# ─────────────────────────────────────────────────────────────────────────────
Write-Banner "NEURAL-ADJOINT NILM — ONE-CLICK PIPELINE LAUNCHER" "Green"
Write-Info "Started  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Info "Local    : $LocalRoot"
Write-Info "Remote   : $RemoteHost  ($RemoteProj)"
Write-Info "Smoke    : $($Smoke.IsPresent)"
Write-Info "Run dir  : $RunDir"
Write-Info ""

# ─── STEP 1: Git commit + push ────────────────────────────────────────────────
Write-Banner "STEP 1 — Git commit and push" "Cyan"
Push-Location $LocalRoot

$status = git status --porcelain 2>&1
if ($status) {
    Write-Info "Uncommitted changes detected — staging and committing..."
    git add --all
    git commit -m "pipeline: auto-commit before run_pipeline.ps1 launch ($Timestamp)"
    Write-Ok "Committed."
} else {
    Write-Info "Working tree is clean — nothing to commit."
}

Write-Info "Pushing to remote repository..."
git push
Write-Ok "Push complete."
Pop-Location

# ─── STEP 2: Verify SSH reachability ─────────────────────────────────────────
Write-Banner "STEP 2 — Verify SSH connectivity" "Cyan"
Write-Info "Testing SSH to $RemoteUser@$RemoteHost ..."

$sshTest = ssh -i $SshKey -o BatchMode=yes -o ConnectTimeout=10 `
    "$RemoteUser@$RemoteHost" "echo OK" 2>&1
if ($sshTest -ne "OK") {
    Write-Fail "SSH test failed. Check Tailscale and the key at $SshKey.`nOutput: $sshTest"
}
Write-Ok "SSH connection OK."

# ─── STEP 3: Pull latest code on remote ──────────────────────────────────────
Write-Banner "STEP 3 — Pull latest code on remote machine" "Cyan"

$pullCmd = @"
cd /d "$RemoteRoot" && git pull --ff-only 2>&1
"@
$pullResult = ssh -i $SshKey "$RemoteUser@$RemoteHost" $pullCmd
Write-Info $pullResult
Write-Ok "Remote repository is up to date."

# ─── STEP 4: Create the log directory on remote ───────────────────────────────
Write-Banner "STEP 4 — Create remote log directory" "Cyan"

$mkdirCmd = "if not exist `"$LogDir`" mkdir `"$LogDir`""
ssh -i $SshKey "$RemoteUser@$RemoteHost" "cmd /c $mkdirCmd" | Out-Null
Write-Ok "Log directory ready: $LogDir"

# ─── STEP 5: Build + register the Scheduled Task on remote ───────────────────
Write-Banner "STEP 5 — Register Scheduled Task on remote machine" "Cyan"

# The launcher script runs pipeline.py and writes stdout/stderr/exit-code.
$LauncherContent = @"
@echo off
cd /d "$RemoteProj"
"$RemotePy" -u pipeline.py$PipelineArgs > "$StdoutLog" 2> "$StderrLog"
echo %ERRORLEVEL% > "$ExitFile"
"@

$RemoteLauncher = "$LogDir\$TaskName.bat"

# Write the .bat launcher to the remote machine via SSH heredoc trick.
$escapedContent = $LauncherContent -replace '"', '\"'

# Write line-by-line using echo commands over SSH
$writeCmd = "echo @echo off > `"$RemoteLauncher`""
ssh -i $SshKey "$RemoteUser@$RemoteHost" "cmd /c $writeCmd" | Out-Null
foreach ($line in $LauncherContent -split "`n") {
    $line = $line.TrimEnd("`r")
    if ($line -eq "") { continue }
    # Escape special cmd characters
    $escaped = $line -replace '([&<>|])', '^$1' -replace '"', '\"'
    $appendCmd = "echo $escaped >> `"$RemoteLauncher`""
    ssh -i $SshKey "$RemoteUser@$RemoteHost" "cmd /c $appendCmd" | Out-Null
}
Write-Ok "Launcher written: $RemoteLauncher"

# Register and immediately start the Scheduled Task.
$registerCmd = @"
schtasks /create /tn "$TaskName" /tr "`"$RemoteLauncher`"" /sc once /st 00:00 /ru "$RemoteUser" /f /rl highest
schtasks /run /tn "$TaskName"
"@
$taskResult = ssh -i $SshKey "$RemoteUser@$RemoteHost" "cmd /c $registerCmd" 2>&1
Write-Info $taskResult
Write-Ok "Task '$TaskName' registered and started on the remote machine."

# ─── STEP 6: Stream logs in real time ────────────────────────────────────────
Write-Banner "STEP 6 — Streaming live log output" "Cyan"
Write-Info "(Press Ctrl+C to stop streaming; the remote task will continue running.)"
Write-Info ""

$PollInterval = 3   # seconds between log polls
$TimeoutMin   = 300 # max wait time in minutes before giving up

$startTime    = Get-Date
$lastLine     = 0

:poll while ($true) {

    # Check if the exit file has appeared → task is done.
    $exitCheck = ssh -i $SshKey "$RemoteUser@$RemoteHost" `
        "if exist `"$ExitFile`" (type `"$ExitFile`") else (echo RUNNING)" 2>&1

    # Fetch new stdout lines since last poll.
    $fetchCmd = @"
powershell -command "if (Test-Path '$StdoutLog') { Get-Content '$StdoutLog' | Select-Object -Skip $lastLine }"
"@
    $newLines = ssh -i $SshKey "$RemoteUser@$RemoteHost" $fetchCmd 2>&1
    if ($newLines) {
        foreach ($line in ($newLines -split "`n")) {
            $line = $line.TrimEnd("`r")
            if ($line -ne "") { Write-Host "  $line" }
            $lastLine++
        }
    }

    if ($exitCheck.Trim() -ne "RUNNING") {
        $exitCode = $exitCheck.Trim()
        break poll
    }

    # Timeout guard.
    if (((Get-Date) - $startTime).TotalMinutes -ge $TimeoutMin) {
        Write-Warn "Polling timeout ($TimeoutMin min). The remote task may still be running."
        Write-Info "Check manually:"
        Write-Info "  ssh -i $SshKey $RemoteUser@$RemoteHost"
        Write-Info "  Get-Content `"$StdoutLog`" -Tail 40"
        exit 0
    }

    Start-Sleep -Seconds $PollInterval
}

# ─── STEP 7: Final status and metrics ────────────────────────────────────────
Write-Banner "STEP 7 — Pipeline complete" "Green"

if ($exitCode -eq "0") {
    Write-Ok "Pipeline finished successfully (exit 0)."
} else {
    Write-Warn "Pipeline exited with code $exitCode. Check:"
    Write-Info "  $StderrLog"
}

# Print the last 30 lines of stdout (contains the summary table).
Write-Info ""
Write-Info "── Final output ──────────────────────────────────────────────────────"
$tailCmd = "powershell -command `"Get-Content '$StdoutLog' -Tail 40`""
$tail    = ssh -i $SshKey "$RemoteUser@$RemoteHost" $tailCmd 2>&1
foreach ($line in ($tail -split "`n")) {
    Write-Host "  $($line.TrimEnd("`r"))"
}

# Try to cat the pipeline report.
$reportPath = "$RunDir\pipeline_report.md"
$reportCmd  = "if exist `"$reportPath`" (type `"$reportPath`")"
$reportText = ssh -i $SshKey "$RemoteUser@$RemoteHost" "cmd /c $reportCmd" 2>&1
if ($reportText) {
    Write-Info ""
    Write-Info "── pipeline_report.md ────────────────────────────────────────────────"
    foreach ($line in ($reportText -split "`n")) {
        Write-Host "  $($line.TrimEnd("`r"))"
    }
}

# Unregister the Scheduled Task now that it has completed.
ssh -i $SshKey "$RemoteUser@$RemoteHost" `
    "cmd /c schtasks /delete /tn `"$TaskName`" /f" | Out-Null
Write-Ok "Scheduled Task '$TaskName' unregistered."
Write-Info ""
Write-Ok "All outputs are on the remote machine under:"
Write-Info "  $RunDir"
Write-Info ""
Write-Info "Logs:"
Write-Info "  stdout : $StdoutLog"
Write-Info "  stderr : $StderrLog"
Write-Info ""
Write-Info "Done at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')."

