"""
STARTUP SEQUENCE TEST: Run the exact initialization order of main_loop()
up to the point of entering the trade loop. Verifies the bot can actually start.
This does NOT place trades — it stops before the while loop.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "startup_test.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

sys.path.insert(0, OUT_DIR)
sys.path.insert(0, os.path.dirname(OUT_DIR))

from datetime import datetime, timezone
import MetaTrader5 as mt5

print("=" * 80)
print("  STARTUP SEQUENCE TEST (replicates main_loop init, stops before trades)")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Replicate main_loop initialization order EXACTLY
try:
    print("\n[1] Import MT5Connection...")
    from mt5_connection import MT5Connection
    _mt5_conn = MT5Connection()
    print("    OK: MT5Connection created")
    
    print("\n[2] Initialize MT5...")
    if not _mt5_conn.initialize():
        print("    FAIL: MT5 init failed")
        sys.exit(1)
    print("    OK: MT5 connected")
    
    print("\n[3] Import strategy + filters...")
    from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
    strategy = GoldScalpingStrategy()
    from news_filter import NewsFilter
    nf = NewsFilter()
    print("    OK: strategy + news filter created")
    
    print("\n[4] Get account info...")
    info = _mt5_conn.get_account_info()
    if info:
        print(f"    OK: Account {info.get('login','?')} Balance ${info['balance']:,.2f}")
    else:
        print("    WARN: no account info")
    
    print("\n[5] Get candles (H4/M15/M5/M1)...")
    for tf, count in [("H4",50),("M15",100),("M5",100),("M1",100)]:
        c = _mt5_conn.get_candles("XAUUSD", tf, count)
        ok = c is not None and len(c) > 0
        print(f"    {tf}: {'OK ('+str(len(c))+' bars)' if ok else 'FAIL'}")
    
    print("\n[6] Compute indicators on M15...")
    m15w = _mt5_conn.get_candles("XAUUSD", "M15", 100)
    m15w_ren = m15w.rename(columns=lambda x: x.lower())
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    i15 = compute_all_indicators(m15w_ren)
    has_adx = 'adx_14' in i15
    has_stoch = 'stochastic' in i15
    has_sma = 'smas' in i15
    print(f"    adx_14: {'OK' if has_adx else 'MISSING'}")
    print(f"    stochastic: {'OK' if has_stoch else 'MISSING'}")
    print(f"    smas: {'OK' if has_sma else 'MISSING'}")
    
    print("\n[7] Verify confirmation engine loads...")
    from trading_bot.strategy.confirmation_engine import evaluate_trade_signals
    print("    OK: evaluate_trade_signals available")
    
    print("\n[8] Verify regime detector inputs available...")
    m15_adx = float(i15.get('adx_14', __import__('pandas').Series([0])).iloc[-1]) if 'adx_14' in i15 else 99
    import numpy as np
    strong_trend = (not np.isnan(m15_adx)) and m15_adx >= 35
    print(f"    ADX={m15_adx:.1f}, strong_trend={strong_trend}")
    
    print("\n" + "=" * 80)
    print("  STARTUP SEQUENCE COMPLETE — bot can reach trade loop")
    print("=" * 80)
    
    _mt5_conn.shutdown()

except Exception as e:
    print(f"\n  STARTUP FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
