# download_ukdale_16khz.ps1 - High Performance Dataset Downloader (Auto-Aria2 Version)

# ==========================================
# 1. SETTINGS
# ==========================================
$houseNum = "1"
$weekNum = "30"

$baseUrl = "https://dap.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0/house_$houseNum/2013/wk$weekNum/"
$folderName = "house" + $houseNum + "_wk" + $weekNum
$targetDir = Join-Path $PSScriptRoot $folderName
$toolsDir = Join-Path $PSScriptRoot "tools"

# ==========================================
# 2. AUTO-ARIA2 CHECK & INSTALL
# ==========================================
$ariaExe = "aria2c.exe"
$ariaPath = Get-Command $ariaExe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source

if (!$ariaPath) {
    # Check if we already downloaded it to the tools folder
    $localAria = Join-Path $toolsDir "aria2-1.37.0-win-64bit-build1\aria2c.exe"
    if (Test-Path $localAria) {
        $ariaPath = $localAria
    } else {
        Write-Host "Aria2 not found. Automatically downloading for high-speed support..." -ForegroundColor Yellow
        if (!(Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir }
        
        $ariaZipUrl = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
        $zipPath = Join-Path $toolsDir "aria2.zip"
        
        Write-Host "Downloading Aria2 from GitHub..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $ariaZipUrl -OutFile $zipPath
        
        Write-Host "Extracting Aria2..." -ForegroundColor Cyan
        Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
        Remove-Item $zipPath
        
        $ariaPath = $localAria
        Write-Host "Aria2 installed successfully in tools folder!" -ForegroundColor Green
    }
}

# ==========================================
# 3. PREPARATION
# ==========================================
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir
}

Write-Host "Fetching file list for House $houseNum Week $weekNum..." -ForegroundColor Cyan
$response = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing
$links = $response.Links | Where-Object { $_.href -like "*.flac" }
Write-Host "Found $($links.Count) files." -ForegroundColor Green

# ==========================================
# 4. DOWNLOAD (TURBO MODE)
# ==========================================
Write-Host "Starting TURBO DOWNLOAD using Aria2..." -ForegroundColor Magenta
foreach ($link in $links) {
    $outPath = Join-Path $targetDir $link.href
    if (!(Test-Path $outPath) -or (Get-Item $outPath).Length -eq 0) {
        $fullUrl = $baseUrl + $link.href
        # Run aria2c with 16 connections per file
        Start-Process -FilePath $ariaPath -ArgumentList "-x16", "-s16", "-d", "`"$targetDir`"", "`"$fullUrl`"" -Wait -NoNewWindow
    } else {
        Write-Host "Skipping $($link.href) (already exists)" -ForegroundColor Gray
    }
}

Write-Host "`nAll tasks completed!" -ForegroundColor Green
