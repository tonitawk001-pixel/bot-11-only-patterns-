"""
VERIFY CRITICAL FIX: Confirm SMA 200 now computes and BUY path is no longer NaN-blocked.
Uses the exact candle count (300) the fixed bot now uses.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "verify_fix.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout
sys.path.insert(0, OUT_DIR)
sys.path.insert(0, os.path.dirname(OUT_DIR))

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  VERIFY CRITICAL FIX: SMA 200 + BUY path")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Connect and pull 300 candles (matches fixed bot)
if not mt5.initialize():
    print("FAIL: MT5 init"); sys.exit(1)

rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 300)
mt5.shutdown()

df = pd.DataFrame(rates)
df.columns = [c.lower() for c in df.columns]
print(f"\n[1] Candles pulled: {len(df)} (bot now uses 300)")

# Compute indicators
from trading_bot.indicators.technical_indicators import compute_all_indicators
ind = compute_all_indicators(df)

# Check SMA 200
sma200 = ind['smas']['SMA_200'].iloc[-1]
sma40 = ind['smas']['SMA_40'].iloc[-1]
print(f"\n[2] SMA 40: {sma40:.2f}")
print(f"    SMA 200: {sma200:.2f}")
print(f"    SMA 200 is NaN: {np.isnan(sma200)}")

if np.isnan(sma200):
    print("\n    FAIL: SMA 200 still NaN — BUY path still broken")
    sys.exit(1)
else:
    print("    PASS: SMA 200 computes correctly")

# Check the trend filter logic directly
from trading_bot.strategy.confirmation_engine import validate_trend_filter_buy
ma40_val = float(sma40)
ma200_val = float(sma200)
trend_ok, trend_msg = validate_trend_filter_buy(ma40_val, ma200_val)
print(f"\n[3] Trend filter (real values):")
print(f"    {trend_msg}")

# Verify the trend filter CAN now pass (not always False due to NaN)
print(f"\n[4] Test: can trend filter ever return True?")
test_cases = [
    (float(sma40), float(sma200), "real values"),
    (4400.0, 4300.0, "bullish (40>200)"),
    (4300.0, 4400.0, "bearish (40<200)"),
    (float('nan'), 4300.0, "NaN ma40"),
    (4400.0, float('nan'), "NaN ma200"),
]
for m40, m200, label in test_cases:
    ok, msg = validate_trend_filter_buy(m40, m200)
    print(f"    {label}: {'PASS' if ok else 'FAIL'} — {msg}")

print(f"\n{'='*80}")
print("  VERIFICATION COMPLETE")
print(f"{'='*80}")
print(f"\n  RESULT: {'FIX CONFIRMED — BUY path works' if not np.isnan(sma200) else 'STILL BROKEN'}")
