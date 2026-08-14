"""
MACD — deep analysis (single indicator, ignore others).
Part 1: relationship between MACD histogram/line and gold price (forward returns).
Part 2: settings sweep to find most profitable MACD filter across all 4 backtests.
Baseline = SMA50 + SELL ATR<0.3.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "macd_report.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize(): print("FATAL"); sys.exit(1)
periods = [
    ("2023 full", datetime(2023,1,1), datetime(2023,12,31)),
    ("Sep23-Apr24", datetime(2023,9,1), datetime(2024,4,30)),
    ("May24-Dec24", datetime(2024,5,1), datetime(2024,12,31)),
    ("2025-2026", datetime(2025,1,1), datetime(2026,8,12)),
]
period_dfs = {}
for pname, start, end in periods:
    r = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, start, end)
    if r is not None and len(r) > 0:
        d = pd.DataFrame(r); d['time']=pd.to_datetime(d['time'],unit='s')
        d.set_index('time',inplace=True); d.columns=[c.lower() for c in d.columns]
        period_dfs[pname]=d
mt5.shutdown()
names = list(period_dfs.keys())

def sma(a,s): return pd.Series(a).rolling(s).mean().values
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values

def compute(sub):
    close=sub['close'].values.astype(float); high=sub['high'].values.astype(float)
    low=sub['low'].values.astype(float)
    tr=np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
    atr14=ema(tr,14); atr_pct=atr14/close*100
    sma50=sma(close,50)
    ef=ema(close,12); es=ema(close,26)
    macd_line=ef-es
    signal=ema(macd_line,9)
    hist=macd_line-signal
    hist_prev=np.roll(hist,1)
    return close,high,low,atr14,atr_pct,sma50,hist,hist_prev,macd_line,signal

def run_bt(close,high,low,atr14,ma,filters=None):
    n=len(close); bal=10000; trades=[]; pos=None; START=10000; RISK=0.02
    for i in range(300,n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(ma[i]): continue
        px=close[i]; av=atr14[i]; direction='B' if px>ma[i] else 'S'
        if pos is not None:
            sl=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl or tp:
                ep=pos[1] if sl else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                pnl-=0.5*pos[4]; bal+=pnl; trades.append(pnl); pos=None
            continue
        if filters:
            blk=False
            for f in filters:
                if f(direction,i,px): blk=True; break
            if blk: continue
        sd=av*2.0; lots=max(0.01,min(5.0,round(START*RISK/(sd*100),2)))
        pos=['B',px-sd,px+sd*2,px,lots] if direction=='B' else ['S',px+sd,px-sd*2,px,lots]
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    return round(sum(trades),2) if trades else 0

data={p:compute(period_dfs[p]) for p in names}

# ============ PART 1: MACD vs PRICE RELATIONSHIP ============
print("="*90)
print("  PART 1: MACD vs GOLD PRICE RELATIONSHIP")
print("="*90)

HORIZONS=[1,4,12,48]
HL={1:"15min",4:"1h",12:"3h",48:"12h"}

for pname in names:
    close,high,low,atr14,atr_pct,sma50,hist,hist_prev,ml,sig = data[pname]
    n=len(close)
    # correlation hist vs next return
    ret1=(close[1:]-close[:-1])/close[:-1]*100
    h_aligned=hist[:-1]; r_aligned=ret1
    mask=~np.isnan(h_aligned)&~np.isnan(r_aligned)
    corr=np.corrcoef(h_aligned[mask],r_aligned[mask])[0,1]
    # hist sign vs forward return
    pos_mask = hist>0
    neg_mask = hist<0
    # forward 3h (12 bars)
    idx_pos=np.where(pos_mask)[0]; idx_pos=idx_pos[idx_pos<n-12]
    idx_neg=np.where(neg_mask)[0]; idx_neg=idx_neg[idx_neg<n-12]
    pos_ret=np.mean((close[idx_pos+12]-close[idx_pos])/close[idx_pos]*100) if len(idx_pos)>0 else np.nan
    neg_ret=np.mean((close[idx_neg+12]-close[idx_neg])/close[idx_neg]*100) if len(idx_neg)>0 else np.nan
    # hist rising vs falling
    rising = hist>hist_prev
    falling = hist<hist_prev
    idx_r=np.where(rising)[0]; idx_r=idx_r[idx_r<n-12]
    idx_f=np.where(falling)[0]; idx_f=idx_f[idx_f<n-12]
    rise_ret=np.mean((close[idx_r+12]-close[idx_r])/close[idx_r]*100) if len(idx_r)>0 else np.nan
    fall_ret=np.mean((close[idx_f+12]-close[idx_f])/close[idx_f]*100) if len(idx_f)>0 else np.nan
    print(f"\n  {pname}:")
    print(f"    corr(histogram, next-bar return) = {corr:+.4f}")
    print(f"    hist>0 -> 3h fwd {pos_ret:+.3f}%  |  hist<0 -> {neg_ret:+.3f}%")
    print(f"    hist rising -> 3h fwd {rise_ret:+.3f}%  |  hist falling -> {fall_ret:+.3f}%")

# ============ PART 2: MACD SETTINGS SWEEP ============
print(f"\n{'='*90}")
print("  PART 2: MACD SETTINGS SWEEP (diff vs baseline)")
print(f"{'='*90}")

base_totals={}
for p in names:
    close,high,low,atr14,atr_pct,sma50,hist,hp,ml,sig = data[p]
    def f(d,i,px,_ap=atr_pct): return (d=='S' and _ap[i]<0.3)
    base_totals[p]=run_bt(close,high,low,atr14,sma50,[f])
BASE=round(sum(base_totals.values()),2)
print(f"\n  BASELINE = {BASE:+,.0f}")

def eval_macd(cond):
    diffs=[]
    for p in names:
        close,high,low,atr14,atr_pct,sma50,hist,hp,ml,sig = data[p]
        def basef(d,i,px,_ap=atr_pct): return (d=='S' and _ap[i]<0.3)
        def mf(d,i,px,_c=cond,_h=hist,_hp=hp): return _c(d,i,_h,_hp)
        v=run_bt(close,high,low,atr14,sma50,[basef,mf])
        diffs.append(round(v-base_totals[p],2))
    net=round(sum(diffs),2); helps=sum(1 for d in diffs if d>0)
    return diffs,net,helps

def sweep(title, vals, mk):
    print(f"\n  {title}")
    rows=[]
    for v in vals:
        diffs,net,helps=eval_macd(mk(v)); rows.append((v,net,helps,diffs))
    rows.sort(key=lambda x:-x[1])
    print(f"    {'setting':<16} {'net':>10} {'helps':>5}   per-period")
    for v,net,helps,diffs in rows:
        print(f"    {str(v):<16} {net:>+10,.0f} {helps:>3}/4   {diffs}")
    best=[r for r in rows if r[2]>=3 and r[1]>0]
    print(f"    >>> {'BEST: '+str(best[0][0])+' net '+format(best[0][1],'+,.0f') if best else 'no robust setting'}")
    return rows

# BUY block when hist < X
sweep("BUY block when histogram < X", [0.0,0.5,1.0,2.0,5.0],
      lambda X:(lambda d,i,h,hp:(d=='B' and h[i]<X)))
# BUY block when hist falling
sweep("BUY block when hist falling", [True,False],
      lambda v:(lambda d,i,h,hp:(d=='B' and h[i]<hp[i])) if v else (lambda d,i,h,hp:(d=='B' and h[i]>hp[i])))
# SELL block when hist > X
sweep("SELL block when histogram > X", [0.0,0.5,1.0,2.0,5.0],
      lambda X:(lambda d,i,h,hp:(d=='S' and h[i]>X)))
# SELL block when hist rising
sweep("SELL block when hist rising", [True,False],
      lambda v:(lambda d,i,h,hp:(d=='S' and h[i]>hp[i])) if v else (lambda d,i,h,hp:(d=='S' and h[i]<hp[i])))
# BUY block hist<0 + SELL block hist>0
sweep("BUY hist<0 AND SELL hist>0", [True],
      lambda v:(lambda d,i,h,hp:(d=='B' and h[i]<0) or (d=='S' and h[i]>0)))

print(f"\n{'='*90}\n  DONE\n{'='*90}")
