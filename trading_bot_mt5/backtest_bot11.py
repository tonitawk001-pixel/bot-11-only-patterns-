"""
BACKTEST bot-11 (pattern engine) across the same 4 periods used for bot-10.
Faithful to pattern_engine.evaluate_patterns logic. Same framework: SL=ATR*2,
TP=2x (1:2), 2% risk of $10k, no compounding. Reports per-period + total.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
sys.stdout = open(os.path.join(OUT, "backtest_bot11.txt"), 'w', encoding='utf-8')
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

# Indicators (vectorized, matching pattern_engine / compute_all_indicators)
sma50 = sma(close,50)
tr = np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1)))
atr14 = ema(tr,14)
# RSI
delta=np.diff(close,prepend=close[0]); g=np.where(delta>0,delta,0.0); l=np.where(delta<0,-delta,0.0)
ag=pd.Series(g).ewm(alpha=1/14,adjust=False).mean().values
al=pd.Series(l).ewm(alpha=1/14,adjust=False).mean().values
rsi = 100 - 100/(1+np.where(al==0,1e-10,al)/ag)
# MACD
ef=ema(close,12); es=ema(close,26); ml=ef-es; sig=ema(ml,9); hist=ml-sig
# Stochastic %K
hh=pd.Series(high).rolling(14).max().values; ll=pd.Series(low).rolling(14).min().values
stoch_k=(close-ll)/np.where(hh-ll==0,1,hh-ll)*100
# BB
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
bu=bm+bs*2; bl=bm-bs*2; bb_pos=(close-bl)/np.where(bu-bl==0,1,bu-bl)
# Volume ratio
vol_ma=sma(vol,20); vol_ratio=vol/np.where(vol_ma==0,1,vol_ma)
# DI
pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if (up>dn and up>0) else 0
    ndm[i]=dn if (dn>up and dn>0) else 0
atr_w=ema(tr,14)
pdi=100*ema(pdm,14)/np.where(atr_w==0,1,atr_w)
ndi=100*ema(ndm,14)/np.where(atr_w==0,1,atr_w)

def pattern_direction(i):
    """Replicates evaluate_patterns for bar i. Returns 'BUY'/'SELL'/'NONE'."""
    buy=0.0; sell=0.0
    # RSI persistence
    if i>=2 and rsi[i]>70 and rsi[i-1]>70 and rsi[i-2]>70:
        buy+=3.0
    elif i>=2 and rsi[i]<30 and rsi[i-1]<30 and rsi[i-2]<30:
        sell+=2.0
    # RSI single
    if rsi[i]>70 and buy<3.0:
        buy+=1.0
    if rsi[i]<30 and sell<2.0:
        sell+=1.0
    # Stoch
    if stoch_k[i]<20: buy+=2.0
    elif stoch_k[i]>80: sell+=1.0
    # BB
    if bb_pos[i]<0.10: buy+=2.0
    elif bb_pos[i]>0.90: sell+=1.0
    # Volume
    if vol_ratio[i]<0.7: buy+=1.0
    # MACD
    if hist[i]>0: buy+=1.0
    else: sell+=1.0
    # DI + trend
    if pdi[i]<ndi[i] and close[i]<sma50[i]: buy+=1.0
    # decide
    if buy>=3.0 and buy>sell: return "BUY"
    if sell>=3.0 and sell>buy: return "SELL"
    return "NONE"

periods = [
    ("2023 full", 0, 23559),
    ("Sep23-Apr24", 23559, 23559+23736),
    ("May24-Dec24", 23559+23736, 23559+23736+23637),
    ("2025-2026", 23559+23736+23637, n),
]

def run_backtest(s, e):
    bal=10000.0; START=10000.0; RISK=0.02
    pos=None; trades=[]
    for i in range(max(300,s), e):
        if np.isnan(close[i]) or np.isnan(atr14[i]): continue
        px=close[i]; av=atr14[i]
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                pnl-=0.5*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        d=pattern_direction(i)
        if d=="NONE": continue
        sl_dist=av*2.0
        lots=max(0.01,min(5.0,round(START*RISK/(sl_dist*100),2)))
        if d=='B': pos=['B',px-sl_dist,px+sl_dist*2.0,px,lots]
        else: pos=['S',px+sl_dist,px-sl_dist*2.0,px,lots]
    if pos is not None:
        ep=close[e-1]
        pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    return round(sum(trades),2) if trades else 0, len(trades)

print("="*80)
print("  BOT-11 (PATTERN ENGINE) BACKTEST — same 4 periods, 1:2 RR")
print("="*80)
totals=[]
for name, s, e in periods:
    pnl, nt = run_backtest(s, e)
    totals.append(pnl)
    print(f"  {name:<14} {pnl:>+12,.2f}  ({nt} trades)")
print(f"\n  {'TOTAL':<14} {sum(totals):>+12,.2f}")
print(f"\n  (bot-10 filter backtest was +$87,951 on the same periods)")
print("="*80)
