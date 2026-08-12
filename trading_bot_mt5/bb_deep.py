
"""
DEEP BOLLINGER BANDS ANALYSIS
Analyze BB position and width at entry vs outcome across all 2024-2026.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "bb_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP BOLLINGER BANDS ANALYSIS — all 2024-2026")
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

# Bollinger Bands
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2
bb_width = (bu - bl) / bm  # normalized BB width

print("Indicators computed.\n")

def collect_trades():
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]) or np.isnan(bm[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        
        # Position relative to BB: 0=lower band, 0.5=mid, 1=upper band
        bb_range = bu[i] - bl[i]
        bb_pos = (px - bl[i]) / bb_range if bb_range > 0 else 0.5
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'bb_pos':pos[5],'bb_width':pos[6]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots,bb_pos,bb_width[i]]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots,bb_pos,bb_width[i]]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'bb_pos':pos[5],'bb_width':pos[6]})
    return trades

trades = collect_trades()
print(f"Baseline: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.0f}")

# ---- BB POSITION vs outcome ----
print(f"\n{'='*80}")
print("  BB POSITION AT ENTRY vs OUTCOME (0=lower, 0.5=mid, 1=upper)")
print(f"{'='*80}")

for lo, hi, label in [(0,0.2,'Below lower area (<0.2)'),(0.2,0.4,'Lower half (0.2-0.4)'),
                       (0.4,0.6,'Middle (0.4-0.6)'),(0.6,0.8,'Upper half (0.6-0.8)'),
                       (0.8,1.0,'Above upper area (>0.8)')]:
    bt = [t for t in trades if lo <= t['bb_pos'] < hi]
    if len(bt) < 5: continue
    bw = [t for t in bt if t['pnl']>0]
    wr = len(bw)/len(bt)*100
    net = sum(t['pnl'] for t in bt)
    print(f"  {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(bt):+,.0f}")

# ---- BB position by direction ----
print(f"\n{'='*80}")
print("  BB POSITION BY DIRECTION")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    for lo, hi, label in [(0,0.3,'Lower (<0.3)'),(0.3,0.7,'Middle (0.3-0.7)'),(0.7,1.0,'Upper (>0.7)')]:
        bt = [t for t in dt if lo <= t['bb_pos'] < hi]
        if len(bt) < 5: continue
        bw = [t for t in bt if t['pnl']>0]
        wr = len(bw)/len(bt)*100
        net = sum(t['pnl'] for t in bt)
        print(f"    {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f}")

# ---- BB WIDTH (volatility) vs outcome ----
print(f"\n{'='*80}")
print("  BB WIDTH (band expansion) vs OUTCOME")
print(f"{'='*80}")

widths = sorted(t['bb_width'] for t in trades if t.get('bb_width') and not np.isnan(t['bb_width']))
if widths:
    pcts = np.percentile(widths, [0,20,40,60,80,100])
    print(f"\n  BB width percentiles: {[f'{p:.3f}' for p in pcts]}")
    for i in range(5):
        lo, hi = pcts[i], pcts[i+1]
        bt = [t for t in trades if t.get('bb_width') and lo <= t['bb_width'] < hi]
        if len(bt) < 5: continue
        bw = [t for t in bt if t['pnl']>0]
        wr = len(bw)/len(bt)*100
        net = sum(t['pnl'] for t in bt)
        print(f"  Width {lo:.3f}-{hi:.3f}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
