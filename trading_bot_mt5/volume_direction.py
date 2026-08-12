
"""
DIRECTION-SPECIFIC VOLUME FILTER TEST
Test separate volume thresholds for BUY and SELL to find if they help.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "volume_direction.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DIRECTION-SPECIFIC VOLUME FILTER TEST")
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

def run(buy_min_vol, buy_max_vol, sell_min_vol, sell_max_vol):
    """
    buy_min_vol/buy_max_vol: volume ratio range for BUY entries (0-100)
    sell_min_vol/sell_max_vol: volume ratio range for SELL entries
    """
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        vn=vol[i]; vma=vol_ma20[i]
        vol_ratio = vn/vma if (vma and vma>0 and not np.isnan(vma)) else 1.0
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        
        if trend_up:
            # BUY volume filter
            if not (buy_min_vol <= vol_ratio <= buy_max_vol):
                continue
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots]
        else:
            # SELL volume filter
            if not (sell_min_vol <= vol_ratio <= sell_max_vol):
                continue
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    if len(trades)<10: return None
    wins=[t for t in trades if t>0]
    return {'net':round(sum(trades),2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1)}

# Baseline: no volume filter (0-100 both)
base = run(0, 100, 0, 100)
print(f"\n  BASELINE (no vol filter): ${base['net']:+,.0f} | {base['trades']}t | WR={base['wr']}%")

# Test BUY-only volume filters (SELL unfiltered)
print(f"\n{'='*80}")
print("  BUY-ONLY VOLUME FILTER (SELL unfiltered)")
print(f"{'='*80}")
buy_tests = [
    (0.5, 100, "buy vol >= 0.5x"),
    (0.7, 100, "buy vol >= 0.7x"),
    (0.7, 2.0, "buy vol 0.7-2.0x"),
    (0.7, 1.5, "buy vol 0.7-1.5x (normal)"),
    (1.0, 2.0, "buy vol 1.0-2.0x"),
]
for bmin, bmax, label in buy_tests:
    r = run(bmin, bmax, 0, 100)
    if r:
        diff = r['net'] - base['net']
        print(f"  {label:<28} ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | {diff:+,.0f}")

# Test SELL-only volume filters (BUY unfiltered)
print(f"\n{'='*80}")
print("  SELL-ONLY VOLUME FILTER (BUY unfiltered)")
print(f"{'='*80}")
sell_tests = [
    (0.5, 100, "sell vol >= 0.5x"),
    (0.7, 100, "sell vol >= 0.7x"),
    (0.7, 2.0, "sell vol 0.7-2.0x"),
    (0.7, 1.5, "sell vol 0.7-1.5x"),
    (1.0, 2.0, "sell vol 1.0-2.0x"),
]
for smin, smax, label in sell_tests:
    r = run(0, 100, smin, smax)
    if r:
        diff = r['net'] - base['net']
        print(f"  {label:<28} ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | {diff:+,.0f}")

# Best combined
print(f"\n{'='*80}")
print("  BEST COMBINED (BUY + SELL specific)")
print(f"{'='*80}")
best = base
best_label = "baseline"
for bmin, bmax in [(0,100),(0.7,2.0),(0.7,1.5)]:
    for smin, smax in [(0,100),(0.7,2.0),(0.7,1.5)]:
        r = run(bmin, bmax, smin, smax)
        if r and r['net'] > best['net']:
            best = r
            best_label = f"buy {bmin}-{bmax}x, sell {smin}-{smax}x"

print(f"\n  BEST: {best_label}")
print(f"  ${best['net']:+,.0f} | {best['trades']}t | WR={best['wr']}%")
print(f"  vs baseline: ${best['net']-base['net']:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
