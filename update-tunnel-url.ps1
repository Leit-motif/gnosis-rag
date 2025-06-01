# Script to update tunnel URL in plugin configuration files
# Run this after getting your new tunnel URL to update all references

param(
    [Parameter(Mandatory=$true)]
    [string]$TunnelURL
)

Write-Host "Updating Tunnel URL Configuration" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host

# Validate URL format
if (-not ($TunnelURL -match "^https://[a-zA-Z0-9\-]+\.trycloudflare\.com$")) {
    Write-Host "Error: Invalid tunnel URL format." -ForegroundColor Red
    Write-Host "Expected format: https://something-random-words.trycloudflare.com" -ForegroundColor Yellow
    exit 1
}

Write-Host "Tunnel URL: $TunnelURL" -ForegroundColor Green
Write-Host

# Files to update
$filesToUpdate = @(
    "customgpt_schema.yaml",
    "plugin/openapi.yaml", 
    "plugin/ai-plugin.json"
)

foreach ($file in $filesToUpdate) {
    if (Test-Path $file) {
        Write-Host "Updating $file..." -ForegroundColor Yellow
        
        try {
            $content = Get-Content $file -Raw
            
            # Replace various placeholder patterns
            $content = $content -replace "YOUR_SERVER_URL_HERE", $TunnelURL
            $content = $content -replace "YOUR_CLOUDFLARE_TUNNEL_URL", $TunnelURL
            $content = $content -replace "https://[a-zA-Z0-9\-]+\.trycloudflare\.com", $TunnelURL
            
            Set-Content $file -Value $content
            Write-Host "✓ Updated $file" -ForegroundColor Green
        }
        catch {
            Write-Host "✗ Failed to update $file`: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "⚠ File not found: $file" -ForegroundColor Yellow
    }
}

# Also update config.json if it exists
if (Test-Path "config.json") {
    Write-Host "Updating config.json..." -ForegroundColor Yellow
    try {
        $config = Get-Content "config.json" | ConvertFrom-Json
        $config | Add-Member -Type NoteProperty -Name "tunnel_url" -Value $TunnelURL -Force
        $config | ConvertTo-Json -Depth 10 | Set-Content "config.json"
        Write-Host "✓ Updated config.json" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to update config.json`: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host
Write-Host "Configuration Update Complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy the updated customgpt_schema.yaml content to your CustomGPT configuration" -ForegroundColor White
Write-Host "2. Restart your FastAPI server to serve the updated plugin files" -ForegroundColor White
Write-Host "3. Test the save functionality in your CustomGPT" -ForegroundColor White
Write-Host

# Copy URL to clipboard
try {
    $TunnelURL | Set-Clipboard
    Write-Host "✓ Tunnel URL copied to clipboard" -ForegroundColor Green
}
catch {
    Write-Host "Note: Could not copy URL to clipboard" -ForegroundColor DarkYellow
}

Write-Host
pause 