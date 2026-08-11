<#
    Python 3.12 Silent Installer for Windows
    =========================================
    Downloads and installs Python 3.12 silently with PATH enabled.
    Run this on a fresh Windows VPS before running setup_bot.bat.

    Usage: Right-click → "Run with PowerShell"
           OR: powershell -ExecutionPolicy Bypass -File install_python312.ps1
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PYTHON 3.12 INSTALLER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python 3.12 is already installed
$pythonFound = $false
try {
    $ver = & py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.12") {
        Write-Host "  ✅ Python 3.12 already installed: $($ver.Trim())" -ForegroundColor Green
        exit 0
    }
} catch { }

try {
    $ver = & python --version 2>&1
    if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.12") {
        Write-Host "  ✅ Python 3.12 already installed: $($ver.Trim())" -ForegroundColor Green
        exit 0
    }
} catch { }

Write-Host "  ⚠️ Python 3.12 not found. Downloading..." -ForegroundColor Yellow

$pythonUrl = "https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe"
$installerPath = "$env:TEMP\python-3.12.4-amd64.exe"

# Download
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $webClient = New-Object System.Net.WebClient
    Write-Host "  Downloading (this may take a moment)..." -ForegroundColor Gray
    $webClient.DownloadFile($pythonUrl, $installerPath)
    Write-Host "  ✅ Downloaded to $installerPath" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Download failed: $_" -ForegroundColor Red
    Write-Host "  Trying alternative download method..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
    } catch {
        Write-Host "  ❌ All download methods failed." -ForegroundColor Red
        Write-Host "  Please download Python 3.12 manually from: https://www.python.org/downloads/" -ForegroundColor Yellow
        pause
        exit 1
    }
}

# Install silently
Write-Host "  Installing Python 3.12 (silent mode)..." -ForegroundColor Yellow
$installArgs = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0 InstallLauncherAllUsers=1"
$process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0) {
    Write-Host "  ✅ Python 3.12 installed successfully!" -ForegroundColor Green
    Write-Host "  Refreshing PATH..." -ForegroundColor Gray
    
    # Refresh PATH in current session
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
    
    # Verify
    try {
        $ver = & py -3.12 --version 2>&1
        Write-Host "  ✅ Verified: $($ver.Trim())" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️ Python installed but 'py -3.12' not working." -ForegroundColor Yellow
        Write-Host "     Try opening a NEW Command Prompt and running: py -3.12 --version" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ❌ Installation failed (exit code: $($process.ExitCode))" -ForegroundColor Red
    Write-Host "  Please install Python 3.12 manually from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  (Make sure to check 'Add Python to PATH' during installation)" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DONE! Now run setup_bot.bat to continue." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
pause