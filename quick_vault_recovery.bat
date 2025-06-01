@echo off
echo =====================================
echo   Gnosis RAG Vault Recovery Tool
echo =====================================
echo.

echo Checking for Python environment...
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found, using system Python
)

echo.
echo Stopping any running backend processes...
taskkill /F /IM python.exe /T 2>nul
timeout /t 3 /nobreak >nul

echo.
echo Running vault reset script...
python reset_vault_with_rate_limiting.py

echo.
echo Recovery process completed.
echo.
echo Next steps:
echo 1. Check the vault_reset.log file for details
echo 2. Restart your application
echo 3. Monitor for any new issues
echo.

pause 