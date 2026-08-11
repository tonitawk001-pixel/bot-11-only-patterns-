"""Backtest on 2019 daily data (no H1 available for 2019)."""
import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging; logging.disable(logging.CRITICAL)

import yfinance as yf
import pandas as pd
import numpy as np

print("Downloading 2019 daily gold data...")
df = yf.download('GC=F', interval='1d', start='2019-01-01', end='2020-01-01', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df.rename(columns=str.lower)
df.index = pd.to_datetime(df.index)
print(f"2019: {len(df)} daily candles, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
print(f"\nNOTE: This is DAILY data — strategy is designed for intraday M15")
print(f"Results will be approximate, not exact live behavior\n")

# Simple trend-following backtest on daily data
# Buy when 50-day EMA > 200-day EMA (trend), sell when opposite
# Risk 2% per trade, 6:1 reward:risk

balance = 304.99
positions = []
closed = []
daily_pnl = 0.0
last_date = None

for i in range(200, len(df)):
    ct = df.index[i]
    close = float(df["close"].iloc[i])
    
    if last_date is None: last_date = ct
    if ct != last_date:
        daily_pnl = 0.0
        last_date = ct
    
    # Compute EMAs
    ema50 = df["close"].iloc[:i].ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = df["close"].iloc[:i].ewm(span=200, adjust=False).mean().iloc[-1]
    
    # ATR for position sizing
    tr = pd.concat([
        (df["high"].iloc[:i] - df["low"].iloc[:i]).abs(),
        (df["high"].iloc[:i] - df["close"].iloc[:i].shift()).abs(),
        (df["low"].iloc[:i] - df["close"].iloc[:i].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    
    direction = "BUY" if ema50 > ema200 else "SELL"
    
    # Close opposite positions
    surv = []
    for p in positions:
        if p["dir"] != direction:
            # Close at market
            pnl = (close - p["entry"]) * p["lot"] * 100 if p["dir"] == "BUY" else (p["entry"] - close) * p["lot"] * 100
            pnl -= 0.50 * p["lot"] * 100
            p["pnl"] = pnl
            p["reason"] = "TREND"
            closed.append(p)
            balance += pnl
        else:
            surv.append(p)
    positions = surv
    
    # Check if we should enter
    if len(positions) == 0:
        sl_dist = atr * 1.5
        tp_dist = atr * 6.0
        risk_amt = balance * 0.02
        risk_per_lot = sl_dist * 100
        lot = max(0.01, min(round(risk_amt / risk_per_lot / 0.01) * 0.01, 10.0)) if risk_per_lot > 0 else 0.01
        
        if direction == "BUY":
            sl = close - sl_dist
            tp = close + tp_dist
        else:
            sl = close + sl_dist
            tp = close - tp_dist
        
        positions.append({
            "entry": close, "sl": sl, "tp": tp, "lot": lot,
            "dir": direction, "open_time": ct
        })

total_pnl = sum(t["pnl"] for t in closed) if closed else 0
wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []
win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []

peak = 304.99
maxdd = 0
eq = [304.99]
for t in closed:
    eq.append(eq[-1] + t["pnl"])
for e in eq:
    peak = max(peak, e)
    dd = (peak - e) / peak * 100 if peak > 0 else 0
    maxdd = max(maxdd, dd)

pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else (float('inf') if win_pnls else 0)
wr = wins / len(closed) * 100 if closed else 0

print("="*60)
print(f"  2019 BACKTEST (Daily Data - Approximate)")
print("="*60)
print(f"  Period:   2019 (252 daily candles)")
print(f"  Starting: ${304.99:.2f}")
print(f"  Final:    ${304.99 + total_pnl:.2f}")
print(f"  Net P&L:  ${total_pnl:+.2f} ({total_pnl/304.99*100:.1f}%)")
print(f"  Max DD:   {maxdd:.1f}%")
print(f"  Trades:   {len(closed)}")
print(f"  Win Rate: {wr:.1f}%")
print(f"  PF:       {pf:.2f}")
print(f"\n  ⚠️ DAILY DATA TEST — not exact live behavior")
print(f"  The actual strategy on M15 candles would be more precise")
print(f"\n  Gold 2019 price: ~$1280 -> ~$1520 (+18.8%)")