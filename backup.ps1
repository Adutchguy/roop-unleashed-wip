# ============================================================
# Configuration
# ============================================================
$verbose = $false   # Set to $true for detailed logging

# Folders to exclude entirely
$excludeFolders = @(
    'models', 
    'venv',
    'env', 
    'output', 
    'temp', 
    '__pycache__', 
    '.git', 
    'saved_configs',
    'insightface',      # face analysis models
    'codeformer',       # enhancer models
    'gfpgan',           # enhancer models
    'checkpoints',      # model checkpoints
    'weights'           # model weights
)

# File extensions to exclude
$excludeExtensions = @(
    '.onnx', '.pth', '.pt', '.bin', '.pkl',     # model files
    '.mp4', '.avi', '.mkv', '.webm', '.gif',    # video files
    '.jpg', '.jpeg', '.png', '.webp', '.bmp',   # image files
    '.zip', '.tar', '.gz', '.7z',               # archives
    '.npy', '.npz',                             # numpy arrays
    '.safetensors', '.ckpt',                    # more model formats
    '.db', '.sqlite',                           # databases
    '.exe', '.dll', '.so', '.pyd'               # binaries
)

# ============================================================
# Setup
# ============================================================
$source = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = "$PSScriptRoot\..\roop-unleashed_backup_$timestamp.zip"
$logFile = "$PSScriptRoot\..\roop-unleashed_backup_$timestamp.log"

function Write-Log {
    param([string]$message, [string]$level = "INFO")
    $entry = "[$(Get-Date -Format 'HH:mm:ss')] [$level] $message"
    if ($verbose) {
        switch ($level) {
            "INFO"    { Write-Host $entry -ForegroundColor Cyan }
            "INCLUDE" { Write-Host $entry -ForegroundColor Green }
            "EXCLUDE" { Write-Host $entry -ForegroundColor Yellow }
            "WARN"    { Write-Host $entry -ForegroundColor Magenta }
            "ERROR"   { Write-Host $entry -ForegroundColor Red }
            default   { Write-Host $entry }
        }
    } else {
        if ($level -eq "INFO" -or $level -eq "ERROR" -or $level -eq "WARN") {
            Write-Host $entry
        }
    }
    Add-Content -Path $logFile -Value $entry
}

# ============================================================
# Main
# ============================================================
Write-Log "Backup started"
Write-Log "Source : $source"
Write-Log "Output : $dest"
Write-Log "Verbose: $verbose"
Write-Log "Scanning files..."

$includedFiles = @()
$excludedCount = 0
$allFiles = Get-ChildItem -Path $source -Recurse -File

foreach ($file in $allFiles) {
    $reason = $null
    $relativePath = $file.FullName.Substring($source.Length + 1).Replace('\', '/')

    # Check excluded folders
    foreach ($folder in $excludeFolders) {
        if ($file.FullName -like "*\$folder\*" -or $file.FullName -like "*\$folder") {
            $reason = "excluded folder '$folder'"
            break
        }
    }

    # Check excluded extensions
    if (-not $reason -and ($excludeExtensions -contains $file.Extension.ToLower())) {
        $reason = "excluded extension '$($file.Extension)'"
    }

    # Check file size - skip anything over 50MB individually
    if (-not $reason -and $file.Length -gt 50MB) {
        $reason = "file too large ($([math]::Round($file.Length / 1MB, 1)) MB)"
    }

    if ($reason) {
        Write-Log "SKIP : $relativePath — $reason" "EXCLUDE"
        $excludedCount++
    } else {
        Write-Log "INCLUDE : $relativePath" "INCLUDE"
        $includedFiles += $file.FullName
    }
}

Write-Log "Scan complete — $($includedFiles.Count) included, $excludedCount excluded" "INFO"

if ($includedFiles.Count -eq 0) {
    Write-Log "No files to archive — aborting!" "ERROR"
    Start-Sleep -Seconds 5
    exit
}

# ============================================================
# Compress using ZipFile .NET class to preserve folder structure
# ============================================================
Write-Log "Compressing files..." "INFO"

try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    Add-Type -AssemblyName System.IO.Compression

    $zip = [System.IO.Compression.ZipFile]::Open($dest, 'Create')

    foreach ($filePath in $includedFiles) {
        # Calculate relative path to preserve directory structure
        $relativePath = $filePath.Substring($source.Length + 1).Replace('\', '/')
        Write-Log "Zipping: $relativePath" "INCLUDE"
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, 
            $filePath, 
            $relativePath, 
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }

    $zip.Dispose()

    $sizeMB = [math]::Round((Get-Item $dest).Length / 1MB, 2)
    Write-Log "Backup complete!" "INFO"
    Write-Log "Output : $dest" "INFO"
    Write-Log "Size   : $sizeMB MB" "INFO"

    if ($sizeMB -gt 30) {
        Write-Log "WARNING: File is larger than Claude's 30MB upload limit of 30MB!" "WARN"
        Write-Log "Consider setting verbose=true and checking the log to find large files." "WARN"
    } else {
        Write-Log "File is within Claude's 30MB upload limit." "INFO"
    }

} catch {
    if ($zip) { $zip.Dispose() }
    Write-Log "Compression failed: $_" "ERROR"
}

Write-Log "Log saved to: $logFile" "INFO"
Write-Host ""
Start-Sleep -Seconds 5