
"""
DIRECTION-SPECIFIC BOLLINGER BAND FILTER TEST
Test BB position filters for BUY and SELL separately across all 2024-2026.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "bb_direction.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DIRECTION-SPECIFIC BOLLINGER BAND TEST")
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
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2

def run(buy_min, buy_max, sell_min, sell_max):
    """BB position (0=lower band, 1=upper band) range for BUY and SELL."""
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]) or np.isnan(bm[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        bb_range = bu[i] - bl[i]
        bb_pos = (px - bl[i]) / bb_range if bb_range > 0 else 0.5
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        
        if trend_up:
            if not (buy_min <= bb_pos <= buy_max): continue
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots]
        else:
            if not (sell_min <= bb_pos <= sell_max): continue
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    if len(trades)<10: return None
    wins=[t for t in trades if t>0]
    return {'net':round(sum(trades),2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1)}

base = run(0,1,0,1)
print(f"\n  BASELINE (no BB filter): ${base['net']:+,.0f} | {base['trades']}t | WR={base['wr']}%")

# BUY-only BB filters
print(f"\n{'='*80}")
print("  BUY-ONLY BB FILTERS")
print(f"{'='*80}")
buy_tests = [
    (0.3, 1.0, "buy BB >0.3 (avoid lower)"),
    (0.5, 1.0, "buy BB >0.5 (upper half)"),
    (0.7, 1.0, "buy BB >0.7 (upper only)"),
    (0.3, 0.7, "buy BB 0.3-0.7 (middle)"),
    (0, 0.7, "buy BB <0.7 (avoid extreme)"),
]
for bmin, bmax, label in buy_tests:
    r = run(bmin, bmax, 0, 1)
    if r:
        diff = r['net']-base['net']
        print(f"  {label:<28} ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | {diff:+,.0f}")

# SELL-only BB filters
print(f"\n{'='*80}")
print("  SELL-ONLY BB FILTERS")
print(f"{'='*80}")
sell_tests = [
    (0, 0.3, "sell BB <0.3 (lower only)"),
    (0, 0.7, "sell BB <0.7 (avoid upper)"),
    (0.3, 1.0, "sell BB >0.3 (avoid lower)"),
    (0, 0.5, "sell BB <0.5 (lower half)"),
]
for smin, smax, label in sell_tests:
    r = run(0, 1, smin, smax)
    if r:
        diff = r['net']-base['net']
        print(f"  {label:<28} ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | {diff:+,.0f}")

# Best combined
print(f"\n{'='*80}")
print("  BEST COMBINED")
print(f"{'='*80}")
best = base; best_cfg = "baseline"
for bmin, bmax in [(0,1),(0.3,1),(0.5,1),(0.7,1),(0.3,0.7)]:
    for smin, smax in [(0,1),(0,0.3),(0,0.7),(0.3,1)]:
        r = run(bmin, bmax, smin, smax)
        if r and r['net'] > best['net']:
            best = r
            best_cfg = f"buy {bmin}-{bmax}, sell {smin}-{smax}"

print(f"\n  BEST: {best_cfg}")
print(f"  ${best['net']:+,.0f} | {best['trades']}t | WR={best['wr']}%")
print(f"  vs baseline: ${best['net']-base['net']:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
