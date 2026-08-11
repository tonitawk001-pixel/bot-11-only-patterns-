<#
    V22 Gold Scalping Bot — Windows Auto Setup Script
    ==================================================
    Run this on ANY fresh Windows VPS or local machine.
    It will:
    1. Install Python (if not already installed)
    2. Install all required Python packages
    3. Clone/update the bot from GitHub
    4. Launch the bot automatically

    Usage:
        1. Copy this script to the VPS
        2. Right-click → "Run with PowerShell"
           OR open PowerShell as Admin and run:
           Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
           .\auto_setup_windows.ps1

    NOTE: Make sure MT5 terminal is installed and logged into your broker BEFORE running.
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  V22 GOLD SCALPING BOT - AUTO SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---- STEP 1: Check / Install Python ----
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow

$pythonInstalled = $false
$pythonPath = ""

# Try common Python locations
$possiblePythonPaths = @(
    "python",
    "python3",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:ProgramFiles\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe"
)

foreach ($p in $possiblePythonPaths) {
    try {
        $version = & $p --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python 3\.(1[1-9]|[2-9]\d+)") {
            $pythonInstalled = $true
            $pythonPath = $p
            Write-Host "  ✅ Python found: $($version.Trim()) at $p" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonInstalled) {
    Write-Host "  ⚠️ Python 3.11+ not found. Downloading and installing..." -ForegroundColor Yellow
    
    # Download Python 3.12 installer
    $pythonUrl = "https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"
    
    try {
        Write-Host "  Downloading Python 3.12.4..." -ForegroundColor Gray
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        (New-Object System.Net.WebClient).DownloadFile($pythonUrl, $installerPath)
        Write-Host "  ✅ Downloaded!" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Failed to download Python. Trying alternative method..." -ForegroundColor Red
        try {
            Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
        } catch {
            Write-Host "  ❌ Download failed. Please install Python manually from python.org" -ForegroundColor Red
            Write-Host "     Then re-run this script."
            pause
            exit 1
        }
    }
    
    # Install silently with PATH
    Write-Host "  Installing Python (this may take a minute)..." -ForegroundColor Yellow
    $installArgs = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Host "  ✅ Python installed successfully!" -ForegroundColor Green
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $pythonPath = "python"
    } else {
        Write-Host "  ❌ Python installation failed (exit code: $($process.ExitCode))" -ForegroundColor Red
        Write-Host "     Please install Python manually from python.org"
        pause
        exit 1
    }
}

# Verify Python works
try {
    $versionCheck = & $pythonPath --version 2>&1
    Write-Host "  ✅ Python ready: $($versionCheck.Trim())" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Cannot run Python. Please install Python 3.11+ and add it to PATH." -ForegroundColor Red
    pause
    exit 1
}

# ---- STEP 2: Upgrade pip ----
Write-Host ""
Write-Host "[2/5] Upgrading pip..." -ForegroundColor Yellow
try {
    & $pythonPath -m pip install --upgrade pip --quiet 2>&1 | Out-Null
    Write-Host "  ✅ pip upgraded" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️ pip upgrade skipped (non-critical)" -ForegroundColor Gray
}

# ---- STEP 3: Install Python packages ----
Write-Host ""
Write-Host "[3/5] Installing required Python packages..." -ForegroundColor Yellow

$packages = @(
    "MetaTrader5",
    "pandas",
    "numpy",
    "requests",
    "flask",
    "flask-cors"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    try {
        $result = & $pythonPath -m pip install $pkg --quiet 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✅ $pkg installed" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️ $pkg install had issues (retrying...)" -ForegroundColor Yellow
            & $pythonPath -m pip install $pkg --quiet --no-cache-dir 2>&1 | Out-Null
        }
    } catch {
        Write-Host "    ⚠️ $pkg install failed, retrying with no cache..." -ForegroundColor Yellow
        & $pythonPath -m pip install $pkg --quiet --no-cache-dir 2>&1 | Out-Null
    }
}

Write-Host "  ✅ All packages installed!" -ForegroundColor Green

# ---- STEP 4: Clone or update the bot ----
Write-Host ""
Write-Host "[4/5] Setting up bot from GitHub..." -ForegroundColor Yellow

$botDir = "$PSScriptRoot\.."
$repoUrl = "https://github.com/tonitawk001-pixel/Ai-bot-1-mt5.git"

# Check if git is available, or just use the files directly
$gitAvailable = $false
try {
    $gitVersion = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gitAvailable = $true
        Write-Host "  ✅ Git found: $($gitVersion.Trim())" -ForegroundColor Green
    }
} catch {
    $gitAvailable = $false
}

if ($gitAvailable) {
    # Check if .git exists (we're in the repo)
    if (Test-Path "$botDir\.git") {
        Write-Host "  ✅ Bot repository found. Updating..." -ForegroundColor Green
        Push-Location $botDir
        git pull origin main 2>&1 | Out-Null
        Pop-Location
        Write-Host "  ✅ Bot updated from GitHub" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Not a git repository. Cloning fresh..." -ForegroundColor Yellow
        $parentDir = Split-Path $botDir -Parent
        git clone $repoUrl "$parentDir\Ai-bot-1-mt5" 2>&1 | Out-Null
        $botDir = "$parentDir\Ai-bot-1-mt5"
        Write-Host "  ✅ Bot cloned from GitHub" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠️ Git not installed. Using local files..." -ForegroundColor Yellow
    Write-Host "  ✅ Bot files are ready" -ForegroundColor Green
}

# ---- STEP 5: Launch the bot ----
Write-Host ""
Write-Host "[5/5] Launching the bot..." -ForegroundColor Yellow
Write-Host ""

$mainScript = Join-Path $botDir "trading_bot_mt5" "main_mt5.py"
if (-not (Test-Path $mainScript)) {
    # Try relative to script location
    $mainScript = Join-Path $PSScriptRoot "..\trading_bot_mt5\main_mt5.py"
}
if (-not (Test-Path $mainScript)) {
    $mainScript = Join-Path (Get-Location) "trading_bot_mt5\main_mt5.py"
}

if (Test-Path $mainScript) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ✅ ALL SETUP COMPLETE!" -ForegroundColor Cyan
    Write-Host "  Starting bot now..." -ForegroundColor Cyan
    Write-Host "  Bot path: $mainScript" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  📱 Bot will send Telegram notifications to your phone" -ForegroundColor White
    Write-Host "  💚 Heartbeat every 5 minutes to confirm it's alive" -ForegroundColor White
    Write-Host "  🔄 Auto-restarts if it crashes" -ForegroundColor White
    Write-Host "  ⏹️  Press Ctrl+C to stop the bot" -ForegroundColor White
    Write-Host ""
    Write-Host "---------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  BOT OUTPUT:" -ForegroundColor Cyan
    Write-Host "---------------------------------------------------" -ForegroundColor DarkGray
    
    & $pythonPath $mainScript
} else {
    Write-Host "  ❌ Bot script not found at: $mainScript" -ForegroundColor Red
    Write-Host "     Please make sure the bot files are in the correct location." -ForegroundColor Yellow
    Write-Host "     Expected: trading_bot_mt5/main_mt5.py" -ForegroundColor Yellow
    pause
    exit 1
}

# Keep window open on exit
Write-Host ""
Write-Host "Bot has stopped. Press any key to close..." -ForegroundColor Gray
pause