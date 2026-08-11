"""
LIVE VALIDATION: Run the v2.0 confirmation engine against current MT5 market data.
Output is written to validate_output.txt due to logger stdout capture.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')

# Redirect all output to file
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_output.txt")
sys.stdout = open(LOG_FILE, 'w', encoding='utf-8')
sys.stderr = sys.stdout

sys.path.insert(0, r'C:\visual studio code\ai bot\bot-6--master\bot-6--master')

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 70)
print("  LIVE VALIDATION — v2.0 Confirmation Engine")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 70)

# Connect to MT5
if not mt5.initialize():
    print("FATAL: MT5 init failed:", mt5.last_error())
    sys.exit(1)

info = mt5.account_info()
if not info:
    print("FATAL: Cannot get account info")
    mt5.shutdown()
    sys.exit(1)

print(f"\nAccount: {info.login} | Server: {info.server}")
print(f"Balance: ${info.balance:,.2f} | Equity: ${info.equity:,.2f}")

# Pull M15 data
SYMBOL = "XAUUSD"
print(f"\nPulling M15 data for {SYMBOL}...")
to_dt = datetime.now(timezone.utc)
from_dt = to_dt - timedelta(days=7)

rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, from_dt, to_dt)
if rates is None or len(rates) < 60:
    print(f"ERROR: Only {len(rates) if rates else 0} M15 candles available")
    mt5.shutdown()
    sys.exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} M15 candles: {df.index[0]} to {df.index[-1]}")

# Compute indicators
print("\nComputing indicators...")
from trading_bot.indicators.technical_indicators import compute_all_indicators
ind = compute_all_indicators(df)

# Extract latest values
def safe_last(s):
    try:
        v = float(s.iloc[-1])
        return None if np.isnan(v) else v
    except:
        return None

print("\n" + "=" * 50)
print("  CURRENT M15 MARKET STATE")
print("=" * 50)
print(f"  Price:     ${df['close'].iloc[-1]:.2f}")
print(f"  RSI(14):   {safe_last(ind['rsi']):.1f}" if safe_last(ind['rsi']) else "  RSI(14):   N/A")
print(f"  SMA 40:    {safe_last(ind['smas']['SMA_40']):.2f}" if 'smas' in ind else "  SMA 40:    N/A")
print(f"  SMA 200:   {safe_last(ind['smas']['SMA_200']):.2f}" if 'smas' in ind else "  SMA 200:   N/A")
print(f"  BB Upper:  {safe_last(ind['bb']['upper']):.2f}")
print(f"  BB Mid:    {safe_last(ind['bb']['middle']):.2f}")
print(f"  BB Lower:  {safe_last(ind['bb']['lower']):.2f}")
print(f"  MACD Line: {safe_last(ind['macd']['macd']):.4f}")
print(f"  MACD Sig:  {safe_last(ind['macd']['signal']):.4f}")
print(f"  MACD Hist: {safe_last(ind['macd']['histogram']):.4f}")

if 'stochastic' in ind and ind['stochastic'] is not None:
    print(f"  Stoch K:   {safe_last(ind['stochastic']['stoch_k']):.1f}")
    print(f"  Stoch D:   {safe_last(ind['stochastic']['stoch_d']):.1f}")
if ind.get('volume_ma') is not None:
    print(f"  Volume:    {df['tick_volume'].iloc[-1]:.0f}")
    print(f"  Vol MA:    {safe_last(ind['volume_ma']):.0f}")

# Run confirmation engine
from trading_bot.strategy.confirmation_engine import evaluate_trade_signals

print("\n" + "=" * 70)
print("  RUNNING CONFIRMATION ENGINE")
print("=" * 70)

def get_val(d, key):
    v = safe_last(d) if hasattr(d, 'iloc') else (d.get(key) if isinstance(d, dict) else None)
    return v

for direction in ["BUY", "SELL"]:
    result = evaluate_trade_signals(
        direction=direction,
        price=float(df['close'].iloc[-1]),
        ma40=safe_last(ind['smas']['SMA_40']) if 'smas' in ind else None,
        ma200=safe_last(ind['smas']['SMA_200']) if 'smas' in ind else None,
        rsi=safe_last(ind['rsi']),
        prev_rsi=safe_last(ind['rsi'].shift(1)) if len(ind['rsi']) >= 2 else None,
        bb_upper=safe_last(ind['bb']['upper']),
        bb_mid=safe_last(ind['bb']['middle']),
        bb_lower=safe_last(ind['bb']['lower']),
        macd_line=safe_last(ind['macd']['macd']),
        macd_signal=safe_last(ind['macd']['signal']),
        macd_hist=safe_last(ind['macd']['histogram']),
        prev_macd_hist=safe_last(ind['macd']['histogram'].shift(1)) if len(ind['macd']) >= 2 else None,
        stoch_k=safe_last(ind['stochastic']['stoch_k']) if ind.get('stochastic') is not None else None,
        stoch_d=safe_last(ind['stochastic']['stoch_d']) if ind.get('stochastic') is not None else None,
        prev_stoch_k=safe_last(ind['stochastic']['stoch_k'].shift(1)) if ind.get('stochastic') is not None and len(ind['stochastic']) >= 2 else None,
        prev_stoch_d=safe_last(ind['stochastic']['stoch_d'].shift(1)) if ind.get('stochastic') is not None and len(ind['stochastic']) >= 2 else None,
        volume=float(df['tick_volume'].iloc[-1]) if 'tick_volume' in df.columns else None,
        vol_ma=safe_last(ind['volume_ma']) if ind.get('volume_ma') is not None else None,
    )

# Also check open positions
positions = mt5.positions_get(symbol=SYMBOL)
if positions:
    print(f"\n  OPEN POSITIONS: {len(positions)}")
    for p in positions:
        print(f"    #{p.ticket}: {'BUY' if p.type==0 else 'SELL'} {p.volume} lots @ {p.price_open} | SL: {p.sl} | TP: {p.tp} | PnL: ${p.profit:.2f}")

# Check spread
tick = mt5.symbol_info_tick(SYMBOL)
if tick:
    spread = round(tick.ask - tick.bid, 2)
    print(f"\n  Current Spread: ${spread}")
    print(f"  Bid: ${tick.bid:.2f} | Ask: ${tick.ask:.2f}")

mt5.shutdown()
print("\n" + "=" * 70)
print("  VALIDATION COMPLETE — Engine is operational")
print("=" * 70)
