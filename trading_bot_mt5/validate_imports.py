"""Integration validation: verify the bot's imports and module loading resolve."""
import sys, os, warnings, importlib
warnings.filterwarnings('ignore')

MT5_DIR = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
ROOT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master"

for p in (MT5_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

results = []
modules = [
    "mt5_connection", "logger_mt5", "telegram_notifier", "trade_exporter",
    "github_setup", "telegram_handler", "github_exporter", "news_filter",
    "deepseek_filter", "candle_patterns", "performance_tracker", "pattern_engine",
    "trading_bot.indicators.technical_indicators",
    "trading_bot.strategy.gold_scalping_strategy",
    "trading_bot.strategy.confirmation_engine",
]

for m in modules:
    try:
        importlib.import_module(m)
        results.append((m, "OK"))
    except Exception as e:
        results.append((m, f"FAIL: {type(e).__name__}: {str(e)[:120]}"))

# Now compile (not import, to avoid side effects) the two bot entry points
for fname in ["main_super.py", "main_patterns.py"]:
    p = os.path.join(MT5_DIR, fname)
    try:
        src = open(p, encoding='utf-8').read()
        compile(src, fname, 'exec')
        results.append((fname, "COMPILE OK"))
    except Exception as e:
        results.append((fname, f"FAIL: {type(e).__name__}: {str(e)[:120]}"))

print("="*70)
print("  INTEGRATION / IMPORT VALIDATION")
print("="*70)
for name, status in results:
    print(f"  {name:<48} {status}")
print("="*70)
fails = [r for r in results if r[1].startswith("FAIL")]
print(f"\n  {len(results)-len(fails)}/{len(results)} passed, {len(fails)} failed")
