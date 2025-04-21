# Set environment variables for testing
$env:OBSIDIAN_VAULT_PATH = "C:\Users\Rando\Sync\My Obsidian Vault"  # Correct vault path

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# Run test script
Write-Host "Running test script..."
python test_save_conversation.py

# Deactivate virtual environment
deactivate 