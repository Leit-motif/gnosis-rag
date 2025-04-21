@echo off
echo Starting Gnosis RAG system...
echo.

REM Path to the project
set PROJECT_PATH=%~dp0
cd %PROJECT_PATH%

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment.
    echo Make sure the virtual environment exists at %PROJECT_PATH%venv
    pause
    exit /b 1
)

REM Start FastAPI server in a new window
echo Starting FastAPI server...
start "Gnosis FastAPI Server" cmd /k "cd %PROJECT_PATH% && call venv\Scripts\activate.bat && uvicorn backend.main:app --host 0.0.0.0 --port 8000"
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to start FastAPI server.
    pause
    exit /b 1
)

REM Wait a moment for the server to start
echo Waiting for server to initialize...
timeout /t 5 /nobreak > nul

REM Start Cloudflare Tunnel in a new window
echo Starting Cloudflare Tunnel...
start "Cloudflare Tunnel" cmd /k "cloudflared tunnel --url http://localhost:8000"
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to start Cloudflare Tunnel.
    echo Make sure cloudflared is installed and in your PATH.
    pause
    exit /b 1
)

echo.
echo Gnosis RAG system is now starting!
echo FastAPI server: http://localhost:8000
echo Cloudflare Tunnel: https://closer-ma-besides-ted.trycloudflare.com
echo.
echo The system is running in separate windows. To stop everything, close those windows.
echo.
pause 