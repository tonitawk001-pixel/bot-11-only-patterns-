
"""
FINAL VOLUME DOUBLE-CHECK: granular SELL thresholds + confirm BUY unfiltered.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "volume_final.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  FINAL VOLUME DOUBLE-CHECK")
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
vol_ma20 = sma(vol, 20)

def run(buy_min, buy_max, sell_min, sell_max):
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        vn=vol[i]; vma=vol_ma20[i]
        vr = vn/vma if (vma and vma>0 and not np.isnan(vma)) else 1.0
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        
        if trend_up:
            if not (buy_min <= vr <= buy_max): continue
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots]
        else:
            if not (sell_min <= vr <= sell_max): continue
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    if len(trades)<10: return None
    wins=[t for t in trades if t>0]
    return {'net':round(sum(trades),2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1)}

# 1. Confirm BUY is best unfiltered (test granular BUY thresholds)
print(f"\n{'='*80}")
print("  1. CONFIRM BUY UNFILTERED (SELL unfiltered)")
print(f"{'='*80}")
base = run(0,100,0,100)
print(f"  BUY no filter: ${base['net']:+,.0f} | {base['trades']}t | WR={base['wr']}%")
for bmin, bmax, label in [(0,0.7,'<0.7x'),(0.7,1.5,'0.7-1.5x'),(0,2.0,'<2.0x'),(0,3.0,'<3.0x')]:
    r = run(bmin,bmax,0,100)
    if r:
        diff = r['net']-base['net']
        print(f"  BUY {label:<12} ${r['net']:+,.0f} | {r['trades']}t | {diff:+,.0f}")

# 2. Granular SELL thresholds (BUY unfiltered)
print(f"\n{'='*80}")
print("  2. GRANULAR SELL VOLUME THRESHOLDS (BUY unfiltered)")
print(f"{'='*80}")
sell_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
for smin in sell_thresholds:
    r = run(0,100,smin,100)
    if r:
        diff = r['net']-base['net']
        print(f"  SELL vol >= {smin:.1f}x: ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | {diff:+,.0f}")

# 3. Also test SELL upper bound (block volume spikes)
print(f"\n{'='*80}")
print("  3. SELL UPPER BOUND (block high volume spikes)")
print(f"{'='*80}")
for smax in [1.5, 2.0, 2.5, 3.0]:
    r = run(0,100,0.7,smax)
    if r:
        diff = r['net']-base['net']
        print(f"  SELL 0.7-{smax}x: ${r['net']:+,.0f} | {r['trades']}t | {diff:+,.0f}")

# 4. Best overall
print(f"\n{'='*80}")
print("  4. BEST FINAL SETTING")
print(f"{'='*80}")
best = base; best_cfg = "baseline"
for smin in sell_thresholds:
    for smax in [100, 2.0, 2.5, 3.0]:
        r = run(0,100,smin,smax)
        if r and r['net'] > best['net']:
            best = r; best_cfg = f"SELL vol {smin}-{smax}x"

print(f"\n  BEST: {best_cfg}")
print(f"  ${best['net']:+,.0f} | {best['trades']}t | WR={best['wr']}%")
print(f"  vs baseline (no filter): ${best['net']-base['net']:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
