"""
PATTERN DISCOVERY — Stochastic, RSI, Volume, DI, ATR, BB (each in isolation).
Finds non-linear patterns: state, persistence, crossover, extreme, combined with trend.
Reports hit rate vs baseline (bull-market drift ~52.6%), flags consistent deviations.
Horizon = 3h (12 bars) and 12h (48 bars).
"""
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

sma50 = sma(close,50)
tr = np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
atr14 = ema(tr,14)
atr_pct = atr14/close*100

# Stochastic
hh = pd.Series(high).rolling(14).max().values
ll = pd.Series(low).rolling(14).min().values
stoch_k = (close-ll)/np.where(hh-ll==0,1,hh-ll)*100
stoch_d = pd.Series(stoch_k).rolling(3).mean().values

# RSI
delta = np.diff(close, prepend=close[0])
g = np.where(delta>0,delta,0.0); l = np.where(delta<0,-delta,0.0)
ag = pd.Series(g).ewm(alpha=1/14,adjust=False).mean().values
al = pd.Series(l).ewm(alpha=1/14,adjust=False).mean().values
rsi = 100 - 100/(1+np.where(al==0,1e-10,al)/ag)

# Volume ratio
vol_ma = sma(vol,20); vol_ratio = vol/np.where(vol_ma==0,1,vol_ma)

# DI
n_=n; pdm=np.zeros(n_); ndm=np.zeros(n_)
for i in range(1,n_):
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if (up>dn and up>0) else 0
    ndm[i]=dn if (dn>up and dn>0) else 0
atr_w=ema(tr,14)
pdi=100*ema(pdm,14)/np.where(atr_w==0,1,atr_w)
ndi=100*ema(ndm,14)/np.where(atr_w==0,1,atr_w)
di_gap=pdi-ndi

# BB position
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2; bb_pos=(close-bl)/np.where(bu-bl==0,1,bu-bl)

subs = [
    ("2023", 0, 23559),
    ("2024", 23559, 23559+23736),
    ("2025", 23559+23736, 23559+23736+23637),
    ("2026", 23559+23736+23637, n),
]

H = 12
fwd = np.full(n, np.nan)
fwd[:n-H] = (close[H:]-close[:-H])/close[:-H]*100

# baseline
base_idx = np.where(~np.isnan(fwd))[0]
BASELINE_HR = (fwd[base_idx] > 0).mean()*100
print(f"BASELINE (unconditional) hit rate = {BASELINE_HR:.2f}% up\n")

def report(name, cond):
    c = cond & ~np.isnan(fwd)
    idx = np.where(c)[0]
    if len(idx) < 100:
        return
    hr = (fwd[idx] > 0).mean()*100
    mean = fwd[idx].mean()
    dev = hr - BASELINE_HR
    hrs = []
    for sn,s,e in subs:
        m = (idx>=s)&(idx<e)
        if m.sum()>=20:
            hrs.append((fwd[idx[m]]>0).mean()*100)
    consistent = (all(h>BASELINE_HR for h in hrs)) or (all(h<BASELINE_HR for h in hrs)) if hrs else False
    flag = "***" if (abs(dev)>=2.0 and consistent) else ("**" if abs(dev)>=1.5 else "   ")
    print(f"  {flag} [{name:<38}] n={len(idx):>5} hit={hr:>5.1f}% (dev {dev:+.1f}) mean={mean:+.3f}%  sub={[f'{h:.0f}' for h in hrs]}")

def section(title):
    print(f"\n{'='*100}\n  {title}\n{'='*100}")

section("STOCHASTIC")
report("K > 80 (overbought)", stoch_k > 80)
report("K < 20 (oversold)", stoch_k < 20)
report("K > 80 for 3+ bars", np.minimum.reduce([np.roll(stoch_k>80,k) for k in range(3)]))
report("K < 20 for 3+ bars", np.minimum.reduce([np.roll(stoch_k<20,k) for k in range(3)]))
report("K crosses above D (bullish)", (stoch_k>stoch_d)&(np.roll(stoch_k,1)<=np.roll(stoch_d,1)))
report("K crosses below D (bearish)", (stoch_k<stoch_d)&(np.roll(stoch_k,1)>=np.roll(stoch_d,1)))
report("K > 50 & price>SMA50", (stoch_k>50)&(close>sma50))
report("K < 50 & price<SMA50", (stoch_k<50)&(close<sma50))
report("K < 50 & price>SMA50 (dip)", (stoch_k<50)&(close>sma50))
report("K > 50 & price<SMA50 (contrarian)", (stoch_k>50)&(close<sma50))

section("RSI")
report("RSI > 70 (overbought)", rsi > 70)
report("RSI < 30 (oversold)", rsi < 30)
report("RSI > 70 for 3+ bars", np.minimum.reduce([np.roll(rsi>70,k) for k in range(3)]))
report("RSI < 30 for 3+ bars", np.minimum.reduce([np.roll(rsi<30,k) for k in range(3)]))
report("RSI > 70 & price>SMA50 (trend OB)", (rsi>70)&(close>sma50))
report("RSI < 30 & price>SMA50 (oversold in uptrend)", (rsi<30)&(close>sma50))
report("RSI < 30 & price<SMA50 (oversold in downtrend)", (rsi<30)&(close<sma50))
report("RSI rising", rsi>np.roll(rsi,1))
report("RSI falling", rsi<np.roll(rsi,1))

section("VOLUME")
report("vol > 1.5x (high)", vol_ratio > 1.5)
report("vol > 2.0x (very high)", vol_ratio > 2.0)
report("vol < 0.7x (low)", vol_ratio < 0.7)
report("vol > 1.5x & price>SMA50", (vol_ratio>1.5)&(close>sma50))
report("vol > 1.5x & price<SMA50", (vol_ratio>1.5)&(close<sma50))
report("vol < 0.7x & price>SMA50 (quiet uptrend)", (vol_ratio<0.7)&(close>sma50))

section("DI (directional)")
report("+DI > -DI (bullish)", di_gap > 0)
report("+DI < -DI (bearish)", di_gap < 0)
report("+DI > -DI & price>SMA50", (di_gap>0)&(close>sma50))
report("+DI < -DI & price<SMA50", (di_gap<0)&(close<sma50))
report("+DI > -DI for 3+ bars", np.minimum.reduce([np.roll(di_gap>0,k) for k in range(3)]))
report("DI gap > +10 (strong bull)", di_gap > 10)
report("DI gap < -10 (strong bear)", di_gap < -10)

section("ATR (volatility)")
atr_hi = np.percentile(atr_pct[~np.isnan(atr_pct)], 75)
atr_lo = np.percentile(atr_pct[~np.isnan(atr_pct)], 25)
report(f"ATR% > {atr_hi:.2f} (high vol, 75th pct)", atr_pct > atr_hi)
report(f"ATR% < {atr_lo:.2f} (low vol, 25th pct)", atr_pct < atr_lo)
report("ATR% high & price>SMA50", (atr_pct>atr_hi)&(close>sma50))
report("ATR% low & price>SMA50", (atr_pct<atr_lo)&(close>sma50))
report("ATR% rising (expanding)", atr_pct>np.roll(atr_pct,1))

section("BOLLINGER")
report("BB pos < 0.1 (near lower)", bb_pos < 0.1)
report("BB pos > 0.9 (near upper)", bb_pos > 0.9)
report("BB pos < 0.1 & price>SMA50 (lower band in uptrend)", (bb_pos<0.1)&(close>sma50))
report("BB pos > 0.9 & price>SMA50 (upper band in uptrend)", (bb_pos>0.9)&(close>sma50))
report("BB pos < 0.1 & price<SMA50", (bb_pos<0.1)&(close<sma50))
report("BB pos in 0.25-0.75 (middle)", (bb_pos>=0.25)&(bb_pos<=0.75))

print("\n\nLEGEND: *** = consistent deviation >2% from baseline, ** = >1.5%")
print("DONE")
