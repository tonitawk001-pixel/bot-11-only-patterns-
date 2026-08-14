"""Test bot-11 improvements: add SMA50 trend filter + raise threshold."""
import os, sys, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
sys.stdout = open(os.path.join(OUT, "bot11_improve.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(OUT, "gold_m15_3y.csv"))
close = df['close'].values.astype(float)
high = df['high'].values.astype(float)
low = df['low'].values.astype(float)
vol = df['tick_volume'].values.astype(float)
n = len(close)

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

sma50 = sma(close,50)
tr = np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
atr14 = ema(tr,14)
delta=np.diff(close,prepend=close[0]); g=np.where(delta>0,delta,0.0); l=np.where(delta<0,-delta,0.0)
ag=pd.Series(g).ewm(alpha=1/14,adjust=False).mean().values
al=pd.Series(l).ewm(alpha=1/14,adjust=False).mean().values
rsi = 100 - 100/(1+np.where(al==0,1e-10,al)/ag)
ef=ema(close,12); es=ema(close,26); ml=ef-es; sig=ema(ml,9); hist=ml-sig
hh=pd.Series(high).rolling(14).max().values; ll=pd.Series(low).rolling(14).min().values
stoch_k=(close-ll)/np.where(hh-ll==0,1,hh-ll)*100
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2; bb_pos=(close-bl)/np.where(bu-bl==0,1,bu-bl)
vol_ma=sma(vol,20); vol_ratio=vol/np.where(vol_ma==0,1,vol_ma)
pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if (up>dn and up>0) else 0
    ndm[i]=dn if (dn>up and dn>0) else 0
atr_w=ema(tr,14); pdi=100*ema(pdm,14)/np.where(atr_w==0,1,atr_w); ndi=100*ema(ndm,14)/np.where(atr_w==0,1,atr_w)

def pattern_scores(i):
    buy=0.0; sell=0.0
    if i>=2 and rsi[i]>70 and rsi[i-1]>70 and rsi[i-2]>70: buy+=3.0
    elif i>=2 and rsi[i]<30 and rsi[i-1]<30 and rsi[i-2]<30: sell+=2.0
    if rsi[i]>70 and buy<3.0: buy+=1.0
    if rsi[i]<30 and sell<2.0: sell+=1.0
    if stoch_k[i]<20: buy+=2.0
    elif stoch_k[i]>80: sell+=1.0
    if bb_pos[i]<0.10: buy+=2.0
    elif bb_pos[i]>0.90: sell+=1.0
    if vol_ratio[i]<0.7: buy+=1.0
    if hist[i]>0: buy+=1.0
    else: sell+=1.0
    if pdi[i]<ndi[i] and close[i]<sma50[i]: buy+=1.0
    return buy, sell

def run(s, e, trend_filter, threshold):
    bal=10000.0; START=10000.0; RISK=0.02; pos=None; trades=[]
    for i in range(max(300,s), e):
        if np.isnan(close[i]) or np.isnan(atr14[i]): continue
        px=close[i]; av=atr14[i]
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                pnl-=0.5*pos[4]; bal+=pnl; trades.append(pnl); pos=None
            continue
        buy, sell = pattern_scores(i)
        d = "NONE"
        if buy>=threshold and buy>sell: d="BUY"
        elif sell>=threshold and sell>buy: d="SELL"
        if d=="NONE": continue
        if trend_filter:
            if d=="BUY" and close[i]<sma50[i]: continue
            if d=="SELL" and close[i]>sma50[i]: continue
        sl_dist=av*2.0; lots=max(0.01,min(5.0,round(START*RISK/(sl_dist*100),2)))
        if d=='B': pos=['B',px-sl_dist,px+sl_dist*2.0,px,lots]
        else: pos=['S',px+sl_dist,px-sl_dist*2.0,px,lots]
    if pos is not None:
        ep=close[e-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    return round(sum(trades),2) if trades else 0, len(trades)

periods = [("2023",0,23559),("Sep23-Apr24",23559,23559+23736),("May24-Dec24",23559+23736,23559+23736+23637),("2025-26",23559+23736+23637,n)]

configs = [
    ("original (threshold 3, no trend)", False, 3.0),
    ("+ trend filter (SMA50)", True, 3.0),
    ("+ trend + threshold 4", True, 4.0),
    ("+ trend + threshold 5", True, 5.0),
    ("trend only, threshold 6", True, 6.0),
]

for name, tf, th in configs:
    per=[]; ntr=0
    for pn,s,e in periods:
        pnl,nt = run(s,e,tf,th); per.append(pnl); ntr+=nt
    tot=round(sum(per),2)
    print(f"  {name:<30} TOTAL {tot:>+12,.0f}  ({ntr} trades)  {per}")

print("\nDone. bot-10 filter bot = +$87,951 reference.")
