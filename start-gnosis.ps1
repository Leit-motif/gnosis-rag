# PowerShell script to start Gnosis RAG system
Write-Host "Starting Gnosis RAG system..." -ForegroundColor Cyan

# Get the script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Check if required commands exist
if (-not (Test-CommandExists "uvicorn")) {
    Write-Host "Error: uvicorn command not found. Make sure you activate your virtual environment first." -ForegroundColor Red
    Write-Host "Try running: venv\Scripts\activate" -ForegroundColor Yellow
    pause
    exit 1
}

if (-not (Test-CommandExists "cloudflared")) {
    Write-Host "Error: cloudflared command not found. Make sure it's installed and in your PATH." -ForegroundColor Red
    pause
    exit 1
}

# Start FastAPI server in a new window
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$scriptPath'; & venv\Scripts\activate; uvicorn backend.main:app --host 0.0.0.0 --port 8000"

# Wait for server to start
Write-Host "Waiting for server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Cloudflare Tunnel in a new window
Write-Host "Starting Cloudflare Tunnel..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cloudflared tunnel --url http://localhost:8000"

Write-Host "`nGnosis RAG system is now starting!" -ForegroundColor Cyan
Write-Host "FastAPI server: http://localhost:8000" -ForegroundColor White
Write-Host "Cloudflare Tunnel: https://closer-ma-besides-ted.trycloudflare.com" -ForegroundColor White
Write-Host "`nThe system is running in separate PowerShell windows. To stop everything, close those windows." -ForegroundColor Yellow

pause 