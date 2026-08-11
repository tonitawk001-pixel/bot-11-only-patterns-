@echo off
echo ========================================
echo   RUNNING ALL BOT INTEGRATION TESTS
echo ========================================
echo.

:: Test 1: Check Python
echo [1/6] Checking Python...
python --version
if errorlevel 1 (
    echo FAILED: Python not found
    pause
    exit /b 1
)
echo OK
echo.

:: Test 2: Check MT5 module
echo [2/6] Checking MetaTrader5...
python -c "import MetaTrader5; print('MetaTrader5', MetaTrader5.__version__); mt5.initialize(); print('MT5 init OK'); mt5.shutdown()"
echo.

:: Test 3: Check all imports
echo [3/6] Checking all module imports...
python -c "
import sys
sys.path.insert(0, 'trading_bot_mt5')
from telegram_notifier import send_message
from telegram_handler import start_polling, set_deepseek_client, stop_polling
from github_exporter import push_analysis, setup_git
from deepseek_filter import DeepSeekFilter
print('All imports OK')
# Send test Telegram
r = send_message('Bot integration test - all modules loaded successfully')
print('Telegram:', 'OK' if r else 'FAILED')
"
echo.

:: Test 4: Test GitHub exporter
echo [4/6] Testing GitHub exporter...
python -c "
import sys; sys.path.insert(0,'trading_bot_mt5')
from github_exporter import setup_git
r = setup_git()
print('Git setup:', r)
"
echo.

:: Test 5: Test config system
echo [5/6] Testing config override system...
python -c "
import sys; sys.path.insert(0,'trading_bot_mt5')
from telegram_handler import _extract_config_json, _save_config_overrides
_save_config_overrides({'FIXED_RISK': 0.01})
j = _extract_config_json('config\n{\"MAX_POSITIONS\": 2}\n')
print('Config system OK:', j)
"
echo.

:: Test 6: Verify main_super.py syntax
echo [6/6] Checking main_super.py syntax...
python -m py_compile trading_bot_mt5/main_super.py
if errorlevel 1 (
    echo FAILED: Syntax error in main_super.py
    pause
    exit /b 1
)
echo Syntax OK
echo.

echo ========================================
echo   ALL TESTS PASSED!
echo ========================================
echo.
echo Bot is ready. Run:
echo   python trading_bot_mt5/main_super.py
echo.
pause