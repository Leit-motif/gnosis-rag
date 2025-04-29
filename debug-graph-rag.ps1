param (
    [string]$Mode = "all"  # Default to running all debug tools
)

# Gnosis Graph RAG Debugging Tool
Write-Host "Gnosis Graph RAG Debugging Tool" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Define function to check if a path is valid
function Test-ValidPath {
    param (
        [string]$Path
    )
    
    if (-not $Path) { return $false }
    if (-not (Test-Path -Path $Path)) { return $false }
    return $true
}

# Activate virtual environment
if (Test-Path "venv\Scripts\activate.ps1") {
    . .\venv\Scripts\activate.ps1
} else {
    Write-Host "Virtual environment not found. Please run: python -m venv venv" -ForegroundColor Red
    exit 1
}

# Get vault path
$vaultPath = $env:OBSIDIAN_VAULT_PATH
if (-not (Test-ValidPath -Path $vaultPath)) {
    Write-Host "Error: OBSIDIAN_VAULT_PATH environment variable not set or path is invalid" -ForegroundColor Red
    Write-Host "Please set it to your Obsidian vault path, e.g.:" -ForegroundColor Yellow
    Write-Host '$env:OBSIDIAN_VAULT_PATH = "C:\path\to\your\vault"' -ForegroundColor Yellow
    exit 1
}

Write-Host "Using Obsidian vault path: $vaultPath" -ForegroundColor Cyan

# Set PYTHONIOENCODING to ensure proper UTF-8 handling
$env:PYTHONIOENCODING = "utf-8"

# Also set the OBSIDIAN_VAULT_PATH in the environment for child processes
$env:OBSIDIAN_VAULT_PATH = $vaultPath

# Run the appropriate debug tool(s) based on mode
try {
    if ($Mode -eq "all" -or $Mode -eq "graph") {
        Write-Host "Running Graph RAG Debugger..." -ForegroundColor Green
        # Pass the vault path without quotes to avoid PowerShell variable interpretation issues
        python backend/debug_graph_rag.py $vaultPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Debug Graph RAG failed with exit code $LASTEXITCODE" -ForegroundColor Red
        } else {
            Write-Host "Debug results written to debug_results.json" -ForegroundColor Green
        }
    }
    
    if ($Mode -eq "all" -or $Mode -eq "cycles") {
        Write-Host "Running Cycle Detection..." -ForegroundColor Green
        python backend/debug_cycles.py $vaultPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Cycle detection failed with exit code $LASTEXITCODE" -ForegroundColor Red
        } else {
            Write-Host "Cycle detection results written to cycle_detection_results.json" -ForegroundColor Green
        }
    }
    
    if ($Mode -eq "all" -or $Mode -eq "expansion") {
        Write-Host "Running Expansion Debugger..." -ForegroundColor Green
        python backend/debug_expansion.py $vaultPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Expansion debugger failed with exit code $LASTEXITCODE" -ForegroundColor Red
        } else {
            Write-Host "Expansion debug results written to expansion_debug_results.json" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
} finally {
    # Deactivate virtual environment
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    }
}

Write-Host "All debug tools completed" -ForegroundColor Green
# Count markdown files in the vault for verification
$markdownCount = (Get-ChildItem -Path $vaultPath -Filter "*.md" -Recurse -File).Count
Write-Host "Found $markdownCount markdown files in the vault" -ForegroundColor Cyan

Write-Host "`nPress any key to continue..."
[void][System.Console]::ReadKey($true) 