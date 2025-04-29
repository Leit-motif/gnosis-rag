@echo off
setlocal enabledelayedexpansion

REM Define parameter with default value
set MODE=%1
if "!MODE!"=="" set MODE=all

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run: python -m venv venv
    exit /b 1
)

REM Get vault path
set VAULT_PATH=%OBSIDIAN_VAULT_PATH%
if "!VAULT_PATH!"=="" (
    echo Error: OBSIDIAN_VAULT_PATH environment variable not set
    echo Please set it to your Obsidian vault path, e.g.:
    echo set OBSIDIAN_VAULT_PATH=C:\path\to\your\vault
    exit /b 1
)

REM Check if path exists
if not exist "!VAULT_PATH!" (
    echo Error: Vault path "!VAULT_PATH!" does not exist
    exit /b 1
)

echo Using Obsidian vault path: !VAULT_PATH!

REM Set PYTHONIOENCODING for proper UTF-8 handling
set PYTHONIOENCODING=utf-8

REM Run the appropriate debug tool(s) based on mode
if "!MODE!"=="all" (
    echo Running Graph RAG Debugger...
    python backend\debug_graph_rag.py "!VAULT_PATH!"
    if errorlevel 1 (
        echo Debug Graph RAG failed with exit code !errorlevel!
    ) else (
        echo Debug results written to debug_results.json
    )

    echo Running Cycle Detection...
    python backend\debug_cycles.py "!VAULT_PATH!"
    if errorlevel 1 (
        echo Cycle detection failed with exit code !errorlevel!
    ) else (
        echo Cycle detection results written to cycle_detection_results.json
    )

    echo Running Expansion Debugger...
    python backend\debug_expansion.py "!VAULT_PATH!"
    if errorlevel 1 (
        echo Expansion debugger failed with exit code !errorlevel!
    ) else (
        echo Expansion debug results written to expansion_debug_results.json
    )
) else if "!MODE!"=="graph" (
    echo Running Graph RAG Debugger...
    python backend\debug_graph_rag.py "!VAULT_PATH!"
) else if "!MODE!"=="cycles" (
    echo Running Cycle Detection...
    python backend\debug_cycles.py "!VAULT_PATH!"
) else if "!MODE!"=="expansion" (
    echo Running Expansion Debugger...
    python backend\debug_expansion.py "!VAULT_PATH!"
) else (
    echo Unknown debug mode: !MODE!
    echo Available modes: all, graph, cycles, expansion
    exit /b 1
)

echo All debug tools completed

REM Count markdown files in the vault for verification
set MD_COUNT=0
for /R "!VAULT_PATH!" %%F in (*.md) do set /a MD_COUNT+=1
echo Found !MD_COUNT! markdown files in the vault

REM Deactivate virtual environment
call deactivate

pause 