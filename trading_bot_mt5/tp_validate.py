"""Validate bot-10 TP change: backtest full bot-10 config with 1:1 vs 1:2 TP."""
import os, sys, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
sys.stdout = open(os.path.join(OUT, "tp_validate.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(OUT, "gold_m15_3y.csv"))
close = df['close'].values.astype(float)
high = df['high'].values.astype(float)
low = df['low'].values.astype(float)
n = len(close)

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

sma50 = sma(close,50)
tr = np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
atr14 = ema(tr,14)
atr_pct = atr14/close*100
# BB
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2; bb_pos=(close-bl)/np.where(bu-bl==0,1,bu-bl)
# ADX (Wilder)
def wilder(a,p):
    a=np.asarray(a,float); out=np.empty_like(a); out[:p]=np.nan; out[p-1]=np.nanmean(a[:p])
    for i in range(p,len(a)): out[i]=(out[i-1]*(p-1)+a[i])/p
    return out
tr2=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr2[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if (up>dn and up>0) else 0
    ndm[i]=dn if (dn>up and dn>0) else 0
atr_w=wilder(tr2,14); sp=wilder(pdm,14); sn=wilder(ndm,14)
pdi=100*sp/np.where(atr_w==0,1,atr_w); ndi=100*sn/np.where(atr_w==0,1,atr_w)
dx=100*np.abs(pdi-ndi)/np.where(pdi+ndi==0,1,pdi+ndi)
adx=wilder(dx,14)

periods = [("2023 full",0,23559),("Sep23-Apr24",23559,23559+23736),("May24-Dec24",23559+23736,23559+23736+23637),("2025-2026",23559+23736+23637,n)]

def run(s,e,rr):
    bal=10000.0; START=10000.0; RISK=0.02; pos=None; trades=[]
    for i in range(300,e):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]): continue
        px=close[i]; av=atr14[i]
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                pnl-=0.5*pos[4]; bal+=pnl; trades.append(pnl); pos=None
            continue
        # bot-10 config: SMA50 trend + ATR + BB + ADX filters
        d = 'B' if px>sma50[i] else 'S'
        if d=='S' and atr_pct[i]<0.3: continue  # ATR filter
        if d=='S' and 0.25<=bb_pos[i]<=0.7: continue  # BB filter
        if not np.isnan(adx[i]) and adx[i]>=50: continue  # ADX block
        sl_dist=av*2.0; lots=max(0.01,min(5.0,round(START*RISK/(sl_dist*100),2)))
        if d=='B': pos=['B',px-sl_dist,px+sl_dist*rr,px,lots]
        else: pos=['S',px+sl_dist,px-sl_dist*rr,px,lots]
    if pos is not None:
        ep=close[e-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    return round(sum(trades),2) if trades else 0, len(trades)

print("="*80)
print("  BOT-10 CONFIG: 1:2 TP vs 1:1 TP (same SL, same filters)")
print("="*80)
for rr,label in [(2.0,"1:2 TP (old)"),(1.0,"1:1 TP (new)")]:
    per=[]; nt=0
    for pn,s,e in periods:
        pnl,cnt=run(s,e,rr); per.append(pnl); nt+=cnt
    tot=round(sum(per),2)
    wins=None
    print(f"\n  {label:<14} TOTAL {tot:>+12,.0f}  ({nt} trades)")
    print(f"    per-period: {[round(x,0) for x in per]}")
print(f"\n  Note: bot-10 backtest earlier reported +$87,951 with ADX>=50 block")
print("="*80)
