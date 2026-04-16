# download_ukdale_16khz.ps1 - High Performance Dataset Downloader

# ==========================================
# 1. SETTINGS (Edit these to change dataset)
# ==========================================
$houseNum = "1"
$weekNum = "30"

# Auto-generate URL and Folder name
$baseUrl = "https://dap.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0/house_$houseNum/2013/wk$weekNum/"
$folderName = "house" + $houseNum + "_wk" + $weekNum
$targetDir = Join-Path $PSScriptRoot $folderName

# ==========================================
# 2. PREPARATION
# ==========================================
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir
    Write-Host "Created directory: $targetDir" -ForegroundColor Green
}

Write-Host "Fetching file list for House $houseNum Week $weekNum from CEDA..." -ForegroundColor Cyan
try {
    # Fetch webpage and extract links
    $response = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing -ErrorAction Stop
}
catch {
    Write-Host "Error: Could not reach CEDA server. Check your URL or Internet connection." -ForegroundColor Red
    exit
}

$links = $response.Links | Where-Object { $_.href -like "*.flac" }
Write-Host "Found $($links.Count) files. Preparing parallel download..." -ForegroundColor Green

# ==========================================
# 3. DOWNLOAD (High-Speed BITS Mode)
# ==========================================
$tasks = @()

foreach ($link in $links) {
    $fileName = $link.href
    $outPath = Join-Path $targetDir $fileName
    
    if ((Test-Path $outPath) -and ((Get-Item $outPath).Length -gt 0)) {
        continue
    }

    $sourceUrl = $baseUrl + $link.href
    Write-Host "Queuing: $fileName" -ForegroundColor Yellow

    # Use BITS (Background Intelligent Transfer Service) - Native Windows Engine
    # -Priority Foreground makes it as fast as possible
    # -Asynchronous allows starting multiple tasks at once
    $job = Start-BitsTransfer -Source $sourceUrl -Destination $outPath -Priority Foreground -Asynchronous
    $tasks += $job
}

if ($tasks.Count -eq 0) {
    Write-Host "All files are already current!" -ForegroundColor Green
} else {
    Write-Host "Waiting for $($tasks.Count) high-speed transfers to complete..." -ForegroundColor Magenta
    
    $completed = 0
    while ($completed -lt $tasks.Count) {
        $completed = ($tasks | Where-Object { $_.JobState -eq "Transferred" -or $_.JobState -eq "Error" }).Count
        $percent = [math]::Round(($completed / $tasks.Count) * 100, 2)
        Write-Progress -Activity "Downloading UK-DALE High-Freq" -Status "$percent% Done ($completed/$($tasks.Count))" -PercentComplete $percent
        
        # Complete finished jobs
        $tasks | Where-Object { $_.JobState -eq "Transferred" } | ForEach-Object { $_ | Complete-BitsTransfer }
        
        Start-Sleep -Seconds 2
    }
    Write-Host "`nAll transfers finished!" -ForegroundColor Green
}
