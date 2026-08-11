"""Test all import dependencies for main_super.py startup."""
import sys, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_check.txt")
sys.stdout = open(OUT, 'w', encoding='utf-8')
sys.stderr = sys.stdout

sys.path.insert(0, r'C:\visual studio code\ai bot\bot-6--master\bot-6--master')
sys.path.insert(0, r'C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5')

checks = []
def check(name, fn):
    try:
        fn()
        print(f'  OK: {name}')
        checks.append(('PASS', name))
    except Exception as e:
        print(f'  FAIL: {name} - {e}')
        checks.append(('FAIL', name))

print('Testing import chain for main_super.py...')
print()

check('mt5_connection', lambda: __import__('mt5_connection'))
check('logger_mt5', lambda: __import__('logger_mt5'))
check('telegram_notifier', lambda: __import__('telegram_notifier'))
check('trade_exporter', lambda: __import__('trade_exporter'))
check('github_setup', lambda: __import__('github_setup'))
check('news_filter', lambda: __import__('news_filter'))
check('deepseek_filter', lambda: __import__('deepseek_filter'))
check('candle_patterns', lambda: __import__('candle_patterns'))
check('performance_tracker', lambda: __import__('performance_tracker'))
check('technical_indicators', lambda: __import__('trading_bot.indicators.technical_indicators'))
check('gold_scalping_strategy', lambda: __import__('trading_bot.strategy.gold_scalping_strategy'))
check('confirmation_engine', lambda: __import__('trading_bot.strategy.confirmation_engine'))

# Instantiate key objects
print()
print('Instantiating key objects...')
try:
    from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
    s = GoldScalpingStrategy()
    print('  OK: GoldScalpingStrategy()')
except Exception as e:
    print(f'  FAIL: GoldScalpingStrategy() - {e}')

try:
    from trading_bot.strategy.confirmation_engine import evaluate_trade_signals
    print('  OK: evaluate_trade_signals (function)')
except Exception as e:
    print(f'  FAIL: evaluate_trade_signals - {e}')

try:
    from news_filter import NewsFilter
    nf = NewsFilter()
    print('  OK: NewsFilter()')
except Exception as e:
    print(f'  FAIL: NewsFilter() - {e}')

passed = sum(1 for s, _ in checks if s == 'PASS')
failed = sum(1 for s, _ in checks if s == 'FAIL')
print(f'\nResult: {passed}/{passed+failed} imports passed, {failed} failed')
print('main_super.py CAN start' if failed == 0 else 'main_super.py CANNOT start')
