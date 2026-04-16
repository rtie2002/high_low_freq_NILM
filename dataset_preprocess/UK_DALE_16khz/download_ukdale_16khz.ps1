# download_ukdale_16khz.ps1 - High Performance Dataset Downloader (Multi-File Turbo)

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
$localAria = Join-Path $toolsDir "aria2-1.37.0-win-64bit-build1\aria2c.exe"

if (!$ariaPath -and !(Test-Path $localAria)) {
    Write-Host "Aria2 not found. Automatically downloading for high-speed support..." -ForegroundColor Yellow
    if (!(Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir }
    $ariaZipUrl = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
    $zipPath = Join-Path $toolsDir "aria2.zip"
    Invoke-WebRequest -Uri $ariaZipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
    Remove-Item $zipPath
}
$ariaPath = if ($ariaPath) { $ariaPath } else { $localAria }

# ==========================================
# 3. PREPARATION
# ==========================================
if (!(Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir }

Write-Host "Fetching file list for House $houseNum Week $weekNum..." -ForegroundColor Cyan
$response = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing
$links = $response.Links | Where-Object { $_.href -like "*.flac" }

# Create a links file for Aria2
$linksFilePath = Join-Path $targetDir "download_links.txt"
$linksContent = ""
foreach ($link in $links) {
    $linksContent += ($baseUrl + $link.href) + "`n"
}
[System.IO.File]::WriteAllText($linksFilePath, $linksContent)

# ==========================================
# 4. MULTI-FILE TURBO DOWNLOAD
# ==========================================
Write-Host "Starting MULTI-FILE TURBO DOWNLOAD using Aria2..." -ForegroundColor Magenta
Write-Host "Concurrent files: 5 | Connections per file: 10" -ForegroundColor Cyan

# -i: Input from file
# -j5: Download 5 files at once
# -x10: 10 connections per file
# --min-split-size=1M: Encourage segmenting
& $ariaPath -i "$linksFilePath" -j5 -x10 -s10 --dir="$targetDir" --min-split-size=1M --continue=true

Write-Host "`nAll tasks completed!" -ForegroundColor Green
