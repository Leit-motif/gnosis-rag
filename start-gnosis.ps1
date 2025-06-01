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

# Function to capture tunnel URL from cloudflared output
function Start-TunnelWithCapture {
    Write-Host "Starting Cloudflare Tunnel and monitoring for URL..." -ForegroundColor Green
    
    # Start cloudflared process and capture output
    $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processStartInfo.FileName = "cloudflared"
    $processStartInfo.Arguments = "tunnel --url http://localhost:8000"
    $processStartInfo.UseShellExecute = $false
    $processStartInfo.RedirectStandardOutput = $true
    $processStartInfo.RedirectStandardError = $true
    $processStartInfo.CreateNoWindow = $false
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processStartInfo
    
    # Variables to capture URL
    $tunnelURL = ""
    $urlFound = $false
    
    # Event handlers for output
    $outputHandler = {
        param($sender, $e)
        if ($e.Data -and $e.Data -match "https://[a-zA-Z0-9\-]+\.trycloudflare\.com") {
            $script:tunnelURL = $matches[0]
            $script:urlFound = $true
            Write-Host "✓ Tunnel URL detected: $($script:tunnelURL)" -ForegroundColor Green
        }
        if ($e.Data) {
            Write-Host "Tunnel: $($e.Data)" -ForegroundColor DarkGray
        }
    }
    
    $errorHandler = {
        param($sender, $e)
        if ($e.Data -and $e.Data -match "https://[a-zA-Z0-9\-]+\.trycloudflare\.com") {
            $script:tunnelURL = $matches[0]
            $script:urlFound = $true
            Write-Host "✓ Tunnel URL detected: $($script:tunnelURL)" -ForegroundColor Green
        }
        if ($e.Data) {
            Write-Host "Tunnel Error: $($e.Data)" -ForegroundColor Yellow
        }
    }
    
    # Register event handlers
    Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outputHandler | Out-Null
    Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errorHandler | Out-Null
    
    # Start the process
    $process.Start() | Out-Null
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    
    # Wait for URL to be detected (up to 30 seconds)
    $timeout = 30
    $elapsed = 0
    Write-Host "Waiting for tunnel URL (timeout: ${timeout}s)..." -ForegroundColor Yellow
    
    while (-not $urlFound -and $elapsed -lt $timeout) {
        Start-Sleep -Seconds 1
        $elapsed++
        if ($elapsed % 5 -eq 0) {
            Write-Host "Still waiting... (${elapsed}s elapsed)" -ForegroundColor DarkYellow
        }
    }
    
    if ($urlFound) {
        Write-Host "✓ Successfully captured tunnel URL!" -ForegroundColor Green
        return $tunnelURL
    } else {
        Write-Host "⚠ Timeout waiting for tunnel URL. Process may still be starting..." -ForegroundColor Yellow
        return $null
    }
}

# Also start a visible tunnel window for user monitoring
Write-Host "Starting visible tunnel window..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Cloudflare Tunnel Monitor' -ForegroundColor Cyan; cloudflared tunnel --url http://localhost:8000"

# Capture the tunnel URL
$tunnelURL = Start-TunnelWithCapture

if (-not $tunnelURL) {
    Write-Host "Could not automatically detect tunnel URL." -ForegroundColor Yellow
    Write-Host "Please check the tunnel window and manually copy the URL." -ForegroundColor Yellow
    $tunnelURL = "https://check-tunnel-window.trycloudflare.com"
} else {
    Write-Host "Successfully detected tunnel URL: $tunnelURL" -ForegroundColor Green
}

# Update config.json with the new tunnel URL
Write-Host "Updating config.json with tunnel URL..." -ForegroundColor Green
try {
    $config = Get-Content "config.json" | ConvertFrom-Json
    $config | Add-Member -Type NoteProperty -Name "tunnel_url" -Value $tunnelURL -Force
    $config | ConvertTo-Json -Depth 10 | Set-Content "config.json"
    Write-Host "✓ Config.json updated successfully" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not update config.json: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Copy URL to clipboard
Write-Host "Copying URL to clipboard..." -ForegroundColor Green
try {
    $tunnelURL | Set-Clipboard
    Write-Host "✓ URL copied to clipboard" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not copy to clipboard: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n===================================" -ForegroundColor Cyan
Write-Host "Gnosis RAG system is now running!" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "FastAPI server: http://localhost:8000" -ForegroundColor White
Write-Host "Cloudflare Tunnel: $tunnelURL" -ForegroundColor Green
Write-Host "`nThe tunnel URL has been:" -ForegroundColor Yellow
Write-Host "  - Updated in config.json" -ForegroundColor White
Write-Host "  - Copied to your clipboard" -ForegroundColor White
Write-Host "`nThe system is running in separate PowerShell windows." -ForegroundColor Yellow
Write-Host "To stop everything, close those windows or press Ctrl+C." -ForegroundColor Yellow

pause 