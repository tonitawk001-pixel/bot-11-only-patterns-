@echo off
title V22 Bot Setup
cd /d "%~dp0"

echo ========================================
echo   V22 GOLD SCALPING BOT - SETUP
echo ========================================
echo.

:: ---- STEP 1: Detect Python 3.12 ----
echo [1/4] Checking Python 3.12...

set PYTHON_CMD=python

:: Try py -3.12 first (Python launcher)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto :python_found
)

:: Try python --version
python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; exit(0) if sys.version_info >= (3,10) else exit(1)" >nul 2>&1
    if not errorlevel 1 (
        goto :python_found
    )
)

:: Python not found or too old
echo   [!] Python 3.10+ not found!
echo.
echo   Choose an option:
echo   1. Install Python 3.12 automatically (recommended)
echo   2. I'll install Python manually (visit python.org)
echo.
set /p choice="  Enter 1 or 2: "

if "%choice%"=="1" (
    echo.
    echo   Running Python installer...
    echo   (A PowerShell window will open - follow the prompts)
    echo.
    powershell -ExecutionPolicy Bypass -File "install_python312.ps1"
    if errorlevel 1 (
        echo   [!] Installation failed. Please install Python 3.12 manually.
        pause
        exit /b 1
    )
    :: Refresh PATH and retry
    call refresh_path.bat 2>nul
    set PYTHON_CMD=py -3.12
    goto :python_found
) else (
    echo   Please download Python 3.12 from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    echo   Then run this script again.
    pause
    exit /b 1
)

:python_found
%PYTHON_CMD% --version
echo   [OK] Python found.
echo.

:: ---- STEP 2: Install packages ----
echo [2/4] Installing required packages (this may take a minute)...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
echo   pip upgraded.

echo   Installing core requirements...
%PYTHON_CMD% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo   [*] Retrying with individual installs...
    %PYTHON_CMD% -m pip install MetaTrader5 --quiet
    %PYTHON_CMD% -m pip install pandas --quiet
    %PYTHON_CMD% -m pip install numpy --quiet
    %PYTHON_CMD% -m pip install requests --quiet
    %PYTHON_CMD% -m pip install python-dotenv --quiet
    %PYTHON_CMD% -m pip install flask --quiet
    %PYTHON_CMD% -m pip install flask-cors --quiet
)
echo   [OK] Core packages installed.

echo   Installing AI/DeepSeek dependency (openai)...
%PYTHON_CMD% -m pip install openai --quiet
if errorlevel 1 (
    echo   [*] openai install failed — bot may still work without AI filter
) else (
    echo   [OK] openai installed.
)

:: Install trading_bot package requirements if available
if exist ..\trading_bot\requirements.txt (
    echo   Installing trading_bot dependencies...
    %PYTHON_CMD% -m pip install -r ..\trading_bot\requirements.txt --quiet
)
echo.

:: ---- STEP 3: Run connection test (optional) ----
echo [3/4] Testing MT5 connection...
echo   (Make sure MT5 terminal is running and logged in!)
echo.
if exist test_mt5.py (
    %PYTHON_CMD% test_mt5.py
    if errorlevel 1 (
        echo   [!] Connection test failed. Check that MT5 is running.
        echo   You can continue, but the bot may not work until MT5 is connected.
        echo.
    )
) else (
    echo   [SKIP] No test_mt5.py found.
)
echo.

:: ---- STEP 4: Launch bot ----
echo.
echo [4/4] Setup complete!
echo.
echo ========================================
echo   ALL DONE! The bot will start shortly.
echo ========================================
echo.
echo   Choose launch option:
echo   1. Start bot now (Super Bot v8 - recommended)
echo   2. Start dashboard (web UI at http://localhost:5000)
echo   3. Exit
echo.
set /p bot_choice="  Enter 1, 2, or 3: "

if "%bot_choice%"=="1" (
    echo.
    echo   Starting Super Bot v8...
    echo   (Press Ctrl+C to stop)
    echo.
    %PYTHON_CMD% main_super.py
) else if "%bot_choice%"=="2" (
    echo.
    echo   Starting dashboard...
    echo   Open: http://localhost:5000
    echo.
    %PYTHON_CMD% dashboard_simple.py
) else (
    echo.
    echo   You can start the bot later by running:
    echo     %PYTHON_CMD% main_super.py
    echo.
    echo   Or the dashboard:
    echo     %PYTHON_CMD% dashboard_simple.py
)

echo.
pause