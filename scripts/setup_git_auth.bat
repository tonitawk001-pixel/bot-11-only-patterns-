@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ============================================
echo   GITHUB AUTH SETUP FOR AUTO-UPDATE
echo ============================================
echo.
echo This will configure git to allow the bot to
echo auto-update from your private GitHub repo.
echo Run this ONCE on the VPS.
echo.
echo Your credentials will be stored securely.
echo ============================================
echo.

set /p GIT_TOKEN="Enter your GitHub Token: "

if "%GIT_TOKEN%"=="" (
    echo ERROR: Token cannot be empty!
    pause
    exit /b 1
)

echo.
echo [1/3] Configuring git credential helper...
git config --global credential.helper store

echo [2/3] Updating remote URL with token...
cd /d "%~dp0.."
git remote set-url final-version https://%GIT_TOKEN%@github.com/tonitawk001-pixel/mt5-bot-edited-final-verison.git

echo [3/3] Testing connection...
git fetch final-version

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo   ✅ SUCCESS! Git authentication configured.
    echo ============================================
    echo.
    echo The bot can now auto-update from GitHub.
    echo To push updates, use:
    echo   git push final-version main
    echo.
) else (
    echo.
    echo ============================================
    echo   ❌ FAILED! Check your token.
    echo ============================================
)

pause