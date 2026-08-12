
"""
DEEP BOLLINGER BAND ANALYSIS — definitive double-check
Full distribution of BB position for wins vs losses, by direction.
Determines if BB has a real edge or is noise.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "bb_deepcheck.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP BOLLINGER BAND DOUBLE-CHECK")
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

# Collect ALL trades with detailed BB position, no filtering
def collect_all():
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
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'bb_pos':pos[5]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots,bb_pos]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots,bb_pos]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'bb_pos':pos[5]})
    return trades

trades = collect_all()
print(f"\nCollected {len(trades)} trades")

# FULL distribution: BB position bins (10 bins) for wins vs losses
print(f"\n{'='*80}")
print("  FULL BB POSITION DISTRIBUTION (10 bins, by direction)")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    wins = [t for t in dt if t['pnl']>0]
    loses = [t for t in dt if t['pnl']<=0]
    
    print(f"\n  {dlabel} (n={len(dt)}, wins={len(wins)}, losses={len(loses)}):")
    print(f"  {'BB pos range':<20} {'Trades':>7} {'WR':>6} {'Avg PnL':>10} {'Total PnL':>12}")
    
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        bt = [t for t in dt if lo <= t['bb_pos'] < hi]
        if len(bt) < 3: 
            print(f"  {lo:.1f}-{hi:.1f}: {len(bt)}t (too few)")
            continue
        bw = [t for t in bt if t['pnl']>0]
        wr = len(bw)/len(bt)*100
        avg = sum(t['pnl'] for t in bt)/len(bt)
        total = sum(t['pnl'] for t in bt)
        print(f"  {lo:.1f}-{hi:.1f}: {len(bt):>7} {wr:>5.0f}% {avg:>+10.0f} {total:>+12.0f}")

# Key question: is there a MONOTONIC pattern (edge) or random scatter (noise)?
print(f"\n{'='*80}")
print("  EDGE DETECTION: monotonic vs random?")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    # Win rates per bin
    bins_wr = []
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        bt = [t for t in dt if lo <= t['bb_pos'] < hi]
        if len(bt) >= 3:
            bw = [t for t in bt if t['pnl']>0]
            bins_wr.append((lo+hi)/2, len(bw)/len(bt)*100)
    
    # Check if WR is monotonic (consistent edge) or scattered (noise)
    if len(bins_wr) >= 5:
        wrs = [w for _, w in bins_wr]
        # Compute trend: correlation between bb_pos and WR
        positions = [p for p, _ in bins_wr]
        if len(set(wrs)) > 1:
            corr = np.corrcoef(positions, wrs)[0,1]
            print(f"    WR vs BB position correlation: {corr:.2f}")
            if abs(corr) > 0.6:
                print(f"    => {'STRONG EDGE (monotonic)' if corr > 0 else 'STRONG EDGE (inverse)'}")
            elif abs(corr) > 0.3:
                print(f"    => weak edge")
            else:
                print(f"    => NOISE (no consistent pattern)")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
