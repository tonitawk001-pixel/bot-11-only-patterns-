"""Sanity test: run pattern engine over 3 years, count signal distribution."""
import os, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
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

# compute indicators as the bot's compute_all_indicators would (approx)
# build i15-like dict as DataFrames/Series
sma50 = sma(close,50)
# rsi
delta=np.diff(close,prepend=close[0]); g=np.where(delta>0,delta,0.0); l=np.where(delta<0,-delta,0.0)
ag=pd.Series(g).ewm(alpha=1/14,adjust=False).mean().values
al=pd.Series(l).ewm(alpha=1/14,adjust=False).mean().values
rsi=pd.Series(100-100/(1+np.where(al==0,1e-10,al)/ag))
# macd
ef=ema(close,12); es=ema(close,26); ml=ef-es; sig=ema(ml,9)
macd=pd.DataFrame({"macd":ml,"signal":sig,"histogram":ml-sig})
# stochastic
hh=pd.Series(high).rolling(14).max().values; ll=pd.Series(low).rolling(14).min().values
sk=pd.Series((close-ll)/np.where(hh-ll==0,1,hh-ll)*100)
sd=sk.rolling(3).mean()
stoch=pd.DataFrame({"stoch_k":sk,"stoch_d":sd})
# bb
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bb=pd.DataFrame({"middle":bm,"upper":bm+bs*2,"lower":bm-bs*2})
# volume ma
vol_ma=pd.Series(vol).rolling(20).mean()
# di
pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if (up>dn and up>0) else 0
    ndm[i]=dn if (dn>up and dn>0) else 0
tr=np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
atr=ema(tr,14)
pdi=pd.Series(100*ema(pdm,14)/np.where(atr==0,1,atr))
ndi=pd.Series(100*ema(ndm,14)/np.where(atr==0,1,atr))
di=pd.DataFrame({"pdi":pdi,"ndi":ndi})
smas=pd.DataFrame({"SMA_50":sma50})

i15 = {"rsi":rsi,"macd":macd,"stochastic":stoch,"bb":bb,"volume_ma":vol_ma,"di_14":di,"smas":smas}

from pattern_engine import evaluate_patterns

buy=sell=none=0
reasons_counter={}
for i in range(50, n):
    px = close[i]
    vr = vol[i]/vol_ma[i] if vol_ma[i]>0 else None
    # evaluate using last values up to i
    i15i = {
        "rsi": rsi.iloc[:i+1],
        "macd": macd.iloc[:i+1],
        "stochastic": stoch.iloc[:i+1],
        "bb": bb.iloc[:i+1],
        "volume_ma": vol_ma.iloc[:i+1],
        "di_14": di.iloc[:i+1],
        "smas": smas.iloc[:i+1],
    }
    res = evaluate_patterns(i15i, px, vr)
    if res["direction"]=="BUY": buy+=1
    elif res["direction"]=="SELL": sell+=1
    else: none+=1
    for r in res["reasons"]:
        reasons_counter[r]=reasons_counter.get(r,0)+1

tot=buy+sell+none
print(f"BUY: {buy} ({buy/tot*100:.1f}%)")
print(f"SELL: {sell} ({sell/tot*100:.1f}%)")
print(f"NONE: {none} ({none/tot*100:.1f}%)")
print(f"\nPattern fire counts:")
for k,v in sorted(reasons_counter.items(), key=lambda x:-x[1]):
    print(f"  {k}: {v}")
