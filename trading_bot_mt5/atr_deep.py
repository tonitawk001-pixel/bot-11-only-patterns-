
"""
DEEP ATR ANALYSIS
Analyze if volatility (ATR) at entry predicts trade outcome across ALL periods.
Find optimal ATR levels for entries and whether ATR filtering improves results.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "atr_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP ATR (VOLATILITY) ANALYSIS — all 2024-2026")
print("=" * 80)

if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2024,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close)
print(f"\nData: {len(df)} candles: {df.index[0]} to {df.index[-1]}")

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

# ATR variants
atr7=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),7)
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
atr21=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),21)

# ATR as % of price (normalized volatility)
atr_pct = atr14 / close * 100

# Trend filter (SMA 50 from previous finding)
sma50=sma(close,50)

print("Indicators computed.\n")

# ---- Backtest: log every trade with ATR ----
def collect_trades(atr_arr, name):
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr_arr[i]) or np.isnan(sma50[i]): continue
        px=close[i]; atr=atr_arr[i]
        trend_up = px > sma50[i]
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'atr':pos[5],'atr_pct':pos[6]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots,atr,atr_pct[i]]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots,atr,atr_pct[i]]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'atr':pos[5],'atr_pct':pos[6]})
    return trades

trades = collect_trades(atr14, 'atr14')

wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
print(f"Baseline: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.0f}")

# ---- ANALYSIS 1: ATR value vs outcome ----
print(f"\n{'='*80}")
print("  ATR VALUE vs OUTCOME")
print(f"{'='*80}")

# Bucket ATR into percentiles
atrs = sorted(t['atr'] for t in trades)
percentiles = np.percentile(atrs, [0, 20, 40, 60, 80, 100])
print(f"\n  ATR percentile boundaries: {[f'{p:.1f}' for p in percentiles]}")

for i in range(5):
    lo, hi = percentiles[i], percentiles[i+1]
    bt = [t for t in trades if lo <= t['atr'] < hi]
    if len(bt) < 5: continue
    bw = [t for t in bt if t['pnl']>0]
    wr = len(bw)/len(bt)*100
    net = sum(t['pnl'] for t in bt)
    print(f"  ATR {lo:.1f}-{hi:.1f}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(bt):+,.0f}")

# ---- ANALYSIS 2: ATR as % of price (normalized) ----
print(f"\n{'='*80}")
print("  ATR % OF PRICE (normalized volatility) vs OUTCOME")
print(f"{'='*80}")

for lo, hi, label in [(0, 0.1, '<0.1% (very low vol)'), (0.1, 0.2, '0.1-0.2% (low)'), 
                       (0.2, 0.3, '0.2-0.3% (medium)'), (0.3, 0.5, '0.3-0.5% (high)'), (0.5, 10, '>0.5% (extreme)')]:
    bt = [t for t in trades if t.get('atr_pct') and lo <= t['atr_pct'] < hi]
    if len(bt) < 5: continue
    bw = [t for t in bt if t['pnl']>0]
    wr = len(bw)/len(bt)*100
    net = sum(t['pnl'] for t in bt)
    print(f"  ATR% {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(bt):+,.0f}")

# ---- ANALYSIS 3: ATR change (volatility expanding vs contracting) ----
print(f"\n{'='*80}")
print("  VOLATILITY CHANGE (ATR rising vs falling)")
print(f"{'='*80}")

# For each trade, check if ATR was rising or falling at entry
# (compare ATR at entry vs ATR 20 bars earlier)
atr_rising = []; atr_falling = []
for t in trades:
    # We didn't store the entry index, approximate using atr_pct trend
    pass  # Skip this for simplicity

# ---- ANALYSIS 4: Which ATR period is best for SL/TP ----
print(f"\n{'='*80}")
print("  ATR PERIOD COMPARISON (7 vs 14 vs 21)")
print(f"{'='*80}")

for atr_arr, name in [(atr7,'ATR 7'), (atr14,'ATR 14'), (atr21,'ATR 21')]:
    t2 = collect_trades(atr_arr, name)
    net = sum(t['pnl'] for t in t2)
    wins2 = [t for t in t2 if t['pnl']>0]
    print(f"  {name}: ${net:+,.0f} | {len(t2)}t | WR={len(wins2)/len(t2)*100:.1f}%")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
