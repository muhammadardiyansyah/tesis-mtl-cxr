param(
    [string]$ProjectRoot = (Get-Location).Path,
    [Parameter(Mandatory = $true)]
    [string]$RawRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

$requiredSourceFolders = @("chestxray14", "chexpert", "tbx11k")
$requiredProjectFiles = @(
    "data\metadata\harmonized_model_ready_v1.csv",
    "data\splits\patient_level_v1\train.csv",
    "data\splits\patient_level_v1\val.csv",
    "data\splits\patient_level_v1\test.csv",
    "data\splits\patient_level_v1\split_report.json"
)

if (Test-Path -LiteralPath $OutputRoot) {
    throw "Output sudah ada: $OutputRoot. Gunakan folder output baru agar tidak menimpa data."
}

foreach ($folderName in $requiredSourceFolders) {
    $sourcePath = Join-Path $RawRoot $folderName
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
        throw "Folder raw tidak ditemukan: $sourcePath"
    }
}

foreach ($relativePath in $requiredProjectFiles) {
    $sourcePath = Join-Path $ProjectRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "File proyek tidak ditemukan: $sourcePath"
    }
}

$archivesDir = Join-Path $OutputRoot "archives"
$metadataDir = Join-Path $OutputRoot "metadata"
$splitsDir = Join-Path $OutputRoot "splits\patient_level_v1"
New-Item -ItemType Directory -Path $archivesDir, $metadataDir, $splitsDir | Out-Null

Copy-Item -LiteralPath (Join-Path $ProjectRoot "data\metadata\harmonized_model_ready_v1.csv") -Destination $metadataDir

foreach ($fileName in @("train.csv", "val.csv", "test.csv", "split_report.json")) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "data\splits\patient_level_v1\$fileName") -Destination $splitsDir
}

foreach ($folderName in $requiredSourceFolders) {
    $archivePath = Join-Path $archivesDir "$folderName.tar"
    Write-Host "Membuat $archivePath ..."
    & tar.exe -cf $archivePath -C $RawRoot $folderName
    if ($LASTEXITCODE -ne 0) {
        throw "Gagal membuat arsip $folderName (exit code $LASTEXITCODE)."
    }
}

Write-Host ""
Write-Host "Paket Kaggle selesai: $OutputRoot"
Get-ChildItem -LiteralPath $archivesDir -File |
    Select-Object Name, @{Name = "UkuranGB"; Expression = { [math]::Round($_.Length / 1GB, 2) }} |
    Format-Table -AutoSize
Write-Host "Pastikan Kaggle Dataset diatur PRIVATE."
