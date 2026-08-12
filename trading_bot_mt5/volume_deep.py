
"""
DEEP VOLUME ANALYSIS
Analyze if volume (tick volume) at entry predicts trade outcome across 2024-2026.
Test volume ratio, volume spikes, and volume trends.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "volume_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP VOLUME ANALYSIS — all 2024-2026")
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

atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma50=sma(close,50)

# Volume moving average (20-period)
vol_ma20 = sma(vol, 20)
vol_ma50 = sma(vol, 50)

print("Indicators computed.\n")

def collect_trades():
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        
        # Volume ratio: current volume vs 20-period average
        vn = vol[i]; vma = vol_ma20[i]
        vol_ratio = vn / vma if (vma and vma > 0 and not np.isnan(vma)) else 1.0
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'vol_ratio':pos[5]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots,vol_ratio]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots,vol_ratio]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'vol_ratio':pos[5]})
    return trades

trades = collect_trades()
print(f"Baseline: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.0f}")

# ---- Volume ratio vs outcome ----
print(f"\n{'='*80}")
print("  VOLUME RATIO (current vs 20-period avg) vs OUTCOME")
print(f"{'='*80}")

for lo, hi, label in [(0,0.5,'<0.5x (very low volume)'),(0.5,1.0,'0.5-1.0x (below avg)'),
                       (1.0,1.5,'1.0-1.5x (avg)'),(1.5,2.5,'1.5-2.5x (above avg)'),
                       (2.5,100,'>2.5x (volume spike)')]:
    bt = [t for t in trades if lo <= t['vol_ratio'] < hi]
    if len(bt) < 5: continue
    bw = [t for t in bt if t['pnl']>0]
    wr = len(bw)/len(bt)*100
    net = sum(t['pnl'] for t in bt)
    print(f"  {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(bt):+,.0f}")

# ---- Volume ratio by direction ----
print(f"\n{'='*80}")
print("  VOLUME RATIO BY DIRECTION")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    for lo, hi, label in [(0,0.7,'Low vol (<0.7x)'),(0.7,1.5,'Normal (0.7-1.5x)'),(1.5,100,'High vol (>1.5x)')]:
        bt = [t for t in dt if lo <= t['vol_ratio'] < hi]
        if len(bt) < 5: continue
        bw = [t for t in bt if t['pnl']>0]
        wr = len(bw)/len(bt)*100
        net = sum(t['pnl'] for t in bt)
        print(f"    {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f}")

# ---- Find optimal volume ratio filter ----
print(f"\n{'='*80}")
print("  VOLUME FILTER TEST: block trades below threshold")
print(f"{'='*80}")

baseline_net = sum(t['pnl'] for t in trades)
for min_ratio in [0.3, 0.5, 0.7, 1.0]:
    kept = [t for t in trades if t['vol_ratio'] >= min_ratio]
    if len(kept) < 10: continue
    net = sum(t['pnl'] for t in kept)
    wins = [t for t in kept if t['pnl']>0]
    wr = len(wins)/len(kept)*100
    diff = net - baseline_net
    print(f"  Require vol >= {min_ratio}x: {len(kept)}t WR={wr:.0f}% PnL=${net:+,.0f} (diff ${diff:+,.0f})")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
