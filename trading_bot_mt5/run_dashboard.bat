@echo off
cd /d "%~dp0"
title V22 Bot Dashboard
echo ========================================
echo   V22 BOT DASHBOARD
echo ========================================
echo.

:: Detect Python
set PYTHON_CMD=python

py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto :run
)

python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; exit(0) if sys.version_info >= (3,10) else exit(1)" >nul 2>&1
    if not errorlevel 1 (
        goto :run
    )
)

echo [!] Python 3.10+ not found!
echo     Run setup_bot.bat first to install Python automatically.
pause
exit /b 1

:run
echo Starting V22 Bot Dashboard...
echo Open: http://localhost:5000
echo Close this window to stop the dashboard.
echo.
%PYTHON_CMD% dashboard_simple.py
echo.
echo Dashboard stopped.
pause
