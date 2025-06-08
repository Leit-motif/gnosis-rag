# Fast minimal startup for Gnosis RAG
Write-Host "Starting Gnosis RAG (minimal mode)..." -ForegroundColor Cyan

# Get script directory and move there
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check for uvicorn
if (-not (Get-Command "uvicorn" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: uvicorn not found. Activate your venv first." -ForegroundColor Red
    Write-Host "Try: venv\Scripts\activate" -ForegroundColor Yellow
    exit 1
}

# Check for cloudflared
if (-not (Get-Command "cloudflared" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: cloudflared not found in PATH." -ForegroundColor Red
    exit 1
}

# Start FastAPI server (background)
Write-Host "Starting FastAPI server..." -ForegroundColor Green
$fastapiProc = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList "-Command", "Set-Location '$scriptPath'; venv\Scripts\activate; uvicorn backend.main:app --host 0.0.0.0 --port 8000"

# Wait briefly for server to start
Start-Sleep -Seconds 3

# Start cloudflared and capture URL
Write-Host "Starting Cloudflare Tunnel..." -ForegroundColor Green
$tunnelURL = $null
$cloudflaredProc = Start-Process -FilePath "cloudflared" -ArgumentList "tunnel --url http://localhost:8000" -NoNewWindow -RedirectStandardOutput "cloudflared.log" -PassThru

# Wait for tunnel URL in log (timeout 20s)
$timeout = 20
$elapsed = 0
while ($elapsed -lt $timeout) {
    if (Test-Path "cloudflared.log") {
        $lines = Get-Content "cloudflared.log" -Raw
        if ($lines -match "https://[a-zA-Z0-9\-]+\.trycloudflare\.com") {
            $tunnelURL = $matches[0]
            break
        }
    }
    Start-Sleep -Seconds 1
    $elapsed++
}
if (-not $tunnelURL) {
    Write-Host "Could not detect tunnel URL. Check cloudflared.log." -ForegroundColor Yellow
    $tunnelURL = "https://check-tunnel-window.trycloudflare.com"
} else {
    Write-Host "Tunnel URL: $tunnelURL" -ForegroundColor Green
}

# Update config.json
if (Test-Path "config.json") {
    try {
        $config = Get-Content "config.json" | ConvertFrom-Json
        $config | Add-Member -Type NoteProperty -Name "tunnel_url" -Value $tunnelURL -Force
        $config | ConvertTo-Json -Depth 10 | Set-Content "config.json"
        Write-Host "✓ config.json updated" -ForegroundColor Green
    } catch {
        Write-Host "Warning: Could not update config.json: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Copy URL to clipboard
try {
    $tunnelURL | Set-Clipboard
    Write-Host "✓ URL copied to clipboard" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not copy to clipboard" -ForegroundColor Yellow
}

Write-Host "`nGnosis RAG is running!" -ForegroundColor Cyan
Write-Host "FastAPI: http://localhost:8000" -ForegroundColor White
Write-Host "Tunnel: $tunnelURL" -ForegroundColor Green
Write-Host "To stop, close the background processes or use Task Manager." -ForegroundColor Yellow 