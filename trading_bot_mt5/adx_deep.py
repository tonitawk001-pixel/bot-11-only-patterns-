
"""
DEEP ADX ANALYSIS
Analyze ADX (trend strength) at entry vs outcome across all 2024-2026.
Find optimal ADX levels and whether ADX filtering improves entries.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "adx_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP ADX ANALYSIS — all 2024-2026")
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

# ADX
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
ae=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

# +DI and -DI for directional strength
pdi=ema(pdm,14); ndi=ema(ndm,14)

print("Indicators computed.\n")

def collect_trades():
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]) or np.isnan(adx14[i]): continue
        px=close[i]; atr=atr14[i]; adx=adx14[i]
        trend_up = px > sma50[i]
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'adx':pos[5],'pdi':pos[6],'ndi':pos[7]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['B',sl,tp,px,lots,adx,pdi[i],ndi[i]]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots,adx,pdi[i],ndi[i]]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'adx':pos[5],'pdi':pos[6],'ndi':pos[7]})
    return trades

trades = collect_trades()
print(f"Baseline: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.0f}")

# ---- ADX value vs outcome ----
print(f"\n{'='*80}")
print("  ADX VALUE vs OUTCOME")
print(f"{'='*80}")

for lo, hi, label in [(0,15,'ADX <15 (no trend)'),(15,20,'15-20 (weak)'),(20,25,'20-25 (developing)'),
                       (25,30,'25-30 (trend)'),(30,40,'30-40 (strong)'),(40,100,'40+ (very strong)')]:
    bt = [t for t in trades if lo <= t['adx'] < hi]
    if len(bt) < 5: continue
    bw = [t for t in bt if t['pnl']>0]
    wr = len(bw)/len(bt)*100
    net = sum(t['pnl'] for t in bt)
    print(f"  {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(bt):+,.0f}")

# ---- ADX by direction ----
print(f"\n{'='*80}")
print("  ADX BY DIRECTION")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    for lo, hi, label in [(0,20,'<20 (weak)'),(20,30,'20-30 (moderate)'),(30,100,'30+ (strong)')]:
        bt = [t for t in dt if lo <= t['adx'] < hi]
        if len(bt) < 5: continue
        bw = [t for t in bt if t['pnl']>0]
        wr = len(bw)/len(bt)*100
        net = sum(t['pnl'] for t in bt)
        print(f"    {label}: {len(bt)}t WR={wr:.0f}% PnL=${net:+,.0f}")

# ---- ADX change (rising vs falling) ----
print(f"\n{'='*80}")
print("  DIRECTIONAL STRENGTH (+DI vs -DI)")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    # +DI > -DI means bullish momentum
    bull = [t for t in dt if t.get('pdi') and t.get('ndi') and t['pdi'] > t['ndi']]
    bear = [t for t in dt if t.get('pdi') and t.get('ndi') and t['pdi'] <= t['ndi']]
    for subset, sublabel in [(bull,'+DI > -DI (bullish momentum)'),(bear,'+DI <= -DI (bearish momentum)')]:
        if len(subset) < 3: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100
        net = sum(t['pnl'] for t in subset)
        print(f"    {sublabel}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
