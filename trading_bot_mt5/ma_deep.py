
"""
DEEP MOVING AVERAGE ANALYSIS
Find the MA period/type that generates most profit across ALL periods (2024+2025+2026).
Tests SMA vs EMA at many periods with simple trend-following + trailing stop.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "ma_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP MOVING AVERAGE ANALYSIS (all 2024-2026 data)")
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

# ATR for SL/TP
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)

# Pre-compute all MA variants
ma_variants = {}
for period in [20, 50, 100, 150, 200, 250, 300]:
    ma_variants[f'sma_{period}'] = sma(close, period)
    ma_variants[f'ema_{period}'] = ema(close, period)

# Also test MA crossover (fast vs slow)
crossover_variants = {
    'sma50_200': (sma(close,50), sma(close,200)),
    'sma20_100': (sma(close,20), sma(close,100)),
    'ema50_200': (ema(close,50), ema(close,200)),
    'ema20_100': (ema(close,20), ema(close,100)),
    'sma20_50': (sma(close,20), sma(close,50)),
}

def run_single_ma(ma_arr, use_cross, fast_arr=None):
    """Trend-following: buy when price > MA, sell when < MA. Trailing stop."""
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(ma_arr[i]): continue
        px=close[i]; atr=atr14[i]
        
        if use_cross:
            if np.isnan(fast_arr[i]): continue
            trend_up = fast_arr[i] > ma_arr[i]
        else:
            trend_up = px > ma_arr[i]
        
        if pos is not None:
            # Trailing stop: lock profit as price moves
            sl_hit=False; tp_hit=False
            if pos[0]=='B':
                if px <= pos[1]: sl_hit=True
                elif px >= pos[2]: tp_hit=True
                else:
                    new_sl = px - atr*3.0
                    if new_sl > pos[1]: pos[1] = new_sl
            else:
                if px >= pos[1]: sl_hit=True
                elif px <= pos[2]: tp_hit=True
                else:
                    new_sl = px + atr*3.0
                    if new_sl < pos[1]: pos[1] = new_sl
            
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    
    if len(trades)<10: return None
    wins=[t for t in trades if t>0]
    return {'net':round(sum(trades),2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1)}

# Test single MA
print(f"\n{'='*80}")
print("  SINGLE MA (price vs MA)")
print(f"{'='*80}")
print(f"  {'MA':<12} {'Net':>10} {'Trades':>8} {'WR':>8}")
single_results = []
for name, arr in ma_variants.items():
    r = run_single_ma(arr, False)
    if r:
        single_results.append((name, r))
        print(f"  {name:<12} ${r['net']:+,.0f}   {r['trades']:<8} {r['wr']}%")

# Test MA crossover
print(f"\n{'='*80}")
print("  MA CROSSOVER (fast vs slow)")
print(f"{'='*80}")
print(f"  {'Crossover':<14} {'Net':>10} {'Trades':>8} {'WR':>8}")
cross_results = []
for name, (fast, slow) in crossover_variants.items():
    r = run_single_ma(slow, True, fast)
    if r:
        cross_results.append((name, r))
        print(f"  {name:<14} ${r['net']:+,.0f}   {r['trades']:<8} {r['wr']}%")

# Best overall
all_results = single_results + cross_results
all_results.sort(key=lambda x: x[1]['net'], reverse=True)

print(f"\n{'='*80}")
print("  TOP 5 MOVING AVERAGE SETTINGS (across ALL 2024-2026)")
print(f"{'='*80}")
for name, r in all_results[:5]:
    print(f"  {name}: ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}%")

# Also show performance split by year for the best MA
best_name = all_results[0][0]
print(f"\n  BEST: {best_name} — ${all_results[0][1]['net']:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
