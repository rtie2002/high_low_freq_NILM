# download_ukdale_16khz.ps1 - UK-DALE 16 kHz bulk downloader using CEDA JSON listing + aria2

param(
    [string]$HouseNum = "2",
    [string]$Year = "2013",
    [string[]]$WeekNums = @("31", "32"),
    [int]$ConcurrentFiles = 5,
    [int]$ConnectionsPerFile = 10
)

$dataBaseUrl = "https://data.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0"
$toolsDir = Join-Path $PSScriptRoot "tools"

function Expand-WeekNums {
    param([string[]]$RawWeeks)
    $weeks = @()
    foreach ($week in $RawWeeks) {
        foreach ($part in ($week -split ",")) {
            $clean = $part.Trim().ToLower().Replace("wk", "")
            if ($clean) {
                if ($clean -match '^\d+$') { $clean = $clean.PadLeft(2, "0") }
                $weeks += $clean
            }
        }
    }
    return $weeks
}

function Get-CedaJson {
    param([string]$Url)
    Write-Host "Reading CEDA JSON: $Url" -ForegroundColor DarkCyan
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing
    return $response.Content | ConvertFrom-Json
}

function Get-DirNames {
    param($Listing)
    return @($Listing.items | Where-Object { $_.type -eq "dir" -or $_.type -eq "directory" } | Select-Object -ExpandProperty name)
}

function Assert-Contains {
    param(
        [string]$Wanted,
        [string[]]$Available,
        [string]$Label
    )
    if ($Available -notcontains $Wanted) {
        Write-Host "`nInvalid $Label`: $Wanted" -ForegroundColor Red
        Write-Host "Available $Label values:" -ForegroundColor Yellow
        foreach ($item in $Available) { Write-Host "  - $item" -ForegroundColor Yellow }
        return $false
    }
    return $true
}

# ==========================================
# 1. AUTO-ARIA2 CHECK & INSTALL
# ==========================================
$ariaExe = "aria2c.exe"
$ariaPath = Get-Command $ariaExe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
$localAria = Join-Path $toolsDir "aria2-1.37.0-win-64bit-build1\aria2c.exe"

if (!$ariaPath -and !(Test-Path $localAria)) {
    Write-Host "Aria2 not found. Automatically downloading for high-speed support..." -ForegroundColor Yellow
    if (!(Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir | Out-Null }
    $ariaZipUrl = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
    $zipPath = Join-Path $toolsDir "aria2.zip"
    Invoke-WebRequest -Uri $ariaZipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
    Remove-Item $zipPath
}
$ariaPath = if ($ariaPath) { $ariaPath } else { $localAria }

# ==========================================
# 2. VALIDATE USER INPUT AGAINST CEDA WEBSITE
# ==========================================
$houseName = "house_$HouseNum"
$weeks = Expand-WeekNums -RawWeeks $WeekNums

$rootListing = Get-CedaJson "$dataBaseUrl`?json"
$availableHouses = Get-DirNames $rootListing
if (!(Assert-Contains -Wanted $houseName -Available $availableHouses -Label "house")) { exit 1 }

$houseUrl = "$dataBaseUrl/$houseName"
$houseListing = Get-CedaJson "$houseUrl`?json"
$availableYears = Get-DirNames $houseListing
if (!(Assert-Contains -Wanted $Year -Available $availableYears -Label "year")) { exit 1 }

$yearUrl = "$houseUrl/$Year"
$yearListing = Get-CedaJson "$yearUrl`?json"
$availableWeeks = Get-DirNames $yearListing

foreach ($weekNum in $weeks) {
    $weekName = "wk$weekNum"
    if (!(Assert-Contains -Wanted $weekName -Available $availableWeeks -Label "week")) {
        continue
    }

    $weekUrl = "$yearUrl/$weekName"
    $weekListing = Get-CedaJson "$weekUrl`?json"
    $links = @($weekListing.items | Where-Object { $_.type -eq "file" -and $_.name -like "*.flac" })

    if (!$links -or $links.Count -eq 0) {
        Write-Host "No .flac files found for $houseName/$Year/$weekName" -ForegroundColor Yellow
        continue
    }

    $targetDir = Join-Path $PSScriptRoot ("$houseName\$Year\$weekName")
    if (!(Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir | Out-Null }

    $linksFilePath = Join-Path $targetDir "download_links.txt"
    $linksContent = ""
    foreach ($link in $links) {
        $dapUrl = $link.download
        $dataUrl = $dapUrl -replace 'dap.ceda.ac.uk', 'data.ceda.ac.uk'
        # List data.ceda.ac.uk first since dap.ceda.ac.uk is currently returning 503
        $linksContent += "$dataUrl`t$dapUrl`r`n"
    }
    [System.IO.File]::WriteAllText($linksFilePath, $linksContent)

    Write-Host "`nFound $($links.Count) .flac files for $houseName/$Year/$weekName." -ForegroundColor Green
    Write-Host "Target folder: $targetDir" -ForegroundColor Cyan
    Write-Host "Starting MULTI-FILE TURBO DOWNLOAD using Aria2..." -ForegroundColor Magenta
    Write-Host "Concurrent files: $ConcurrentFiles | Connections per file: $ConnectionsPerFile" -ForegroundColor Cyan

    $downloadSuccess = $false
    while (!$downloadSuccess) {
        # 2>$null redirects stderr (where Aria2 outputs [ERROR] logs) to null, keeping only the beautiful stdout progress bar.
        & $ariaPath -i "$linksFilePath" -j $ConcurrentFiles -x $ConnectionsPerFile -s $ConnectionsPerFile --dir="$targetDir" --min-split-size=1M --continue=true 2>$null
        if ($LASTEXITCODE -eq 0) {
            $downloadSuccess = $true
        } else {
            Write-Host "`n[Aria2] Some files failed to download (likely due to CEDA server 503 errors)." -ForegroundColor Yellow
            Write-Host "[Retry] Waiting 10 seconds before retrying remaining files..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
    }
}

Write-Host "`nAll requested download attempts completed!" -ForegroundColor Green
