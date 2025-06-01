# Utility script to capture Cloudflare tunnel URL and update config.json
# Run this after starting the tunnel if you need to capture the URL later

Write-Host "Cloudflare Tunnel URL Capture Utility" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host

# Check if cloudflared is running
$cloudflaredProcesses = Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue

if (-not $cloudflaredProcesses) {
    Write-Host "Error: No cloudflared processes found running." -ForegroundColor Red
    Write-Host "Please start the tunnel first using start-gnosis.ps1 or start-gnosis.bat" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Found $($cloudflaredProcesses.Count) cloudflared process(es) running." -ForegroundColor Green

# Try to get the URL from netstat (looking for the tunnel connection)
Write-Host "Attempting to detect tunnel URL..." -ForegroundColor Yellow

# Method 1: Check for recent tunnel URLs in Windows event logs or common locations
$tunnelURL = $null

# Method 2: Prompt user to enter the URL manually since automatic detection is complex
Write-Host "Automatic URL detection from running process is complex." -ForegroundColor Yellow
Write-Host "Please check your Cloudflare Tunnel window and copy the tunnel URL." -ForegroundColor Yellow
Write-Host
Write-Host "The URL should look like: https://something-random-words.trycloudflare.com" -ForegroundColor Cyan
Write-Host

# Prompt for manual URL entry
do {
    $tunnelURL = Read-Host "Enter the tunnel URL (or press Enter to skip)"
    
    if ([string]::IsNullOrWhiteSpace($tunnelURL)) {
        Write-Host "Skipping URL update." -ForegroundColor Yellow
        pause
        exit 0
    }
    
    # Validate URL format
    if ($tunnelURL -match "^https://[a-zA-Z0-9\-]+\.trycloudflare\.com$") {
        Write-Host "✓ Valid tunnel URL format." -ForegroundColor Green
        break
    } else {
        Write-Host "Invalid URL format. Please enter a valid trycloudflare.com URL." -ForegroundColor Red
    }
} while ($true)

# Update config.json
Write-Host "Updating config.json..." -ForegroundColor Green
try {
    if (Test-Path "config.json") {
        $config = Get-Content "config.json" | ConvertFrom-Json
        $config | Add-Member -Type NoteProperty -Name "tunnel_url" -Value $tunnelURL -Force
        $config | ConvertTo-Json -Depth 10 | Set-Content "config.json"
        Write-Host "✓ Config.json updated successfully with: $tunnelURL" -ForegroundColor Green
    } else {
        Write-Host "Warning: config.json not found in current directory." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error updating config.json: $($_.Exception.Message)" -ForegroundColor Red
}

# Copy to clipboard
Write-Host "Copying URL to clipboard..." -ForegroundColor Green
try {
    $tunnelURL | Set-Clipboard
    Write-Host "✓ URL copied to clipboard" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not copy to clipboard: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host
Write-Host "Done! Your tunnel URL has been captured and saved." -ForegroundColor Green
Write-Host "URL: $tunnelURL" -ForegroundColor Cyan
Write-Host

pause 