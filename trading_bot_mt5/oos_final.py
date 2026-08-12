"""
FINAL OUT-OF-SAMPLE VALIDATION
Test current optimal config on FRESH 6-month data (Aug 2025 - Jan 2026)
Never seen during any optimization.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "oos_final.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  FINAL OUT-OF-SAMPLE: Aug 2025 - Jan 2026 (NEVER SEEN)")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

print("\nPulling Aug 2025 - Jan 2026 M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2025,8,1,tzinfo=timezone.utc), datetime(2026,1,31,tzinfo=timezone.utc))
mt5.shutdown()
if rates is None or len(rates) < 1000: print("ERROR"); sys.exit(1)

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close); hours=np.array([t.hour for t in df.index])
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

# Indicators
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values
d=np.zeros(n); d[1:]=np.diff(close); g=np.maximum(d,0); l=np.maximum(-d,0)
ag=np.full(n,np.nan); al=np.full(n,np.nan)
if n>14: ag[14]=np.mean(g[1:15]); al[14]=np.mean(l[1:15])
for i in range(15,n): ag[i]=(ag[i-1]*13+g[i])/14; al[i]=(al[i-1]*13+l[i])/14
rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
rsi14=np.full(n,np.nan); rsi14[m]=100-100/(1+rs[m])
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma40=sma(close,40); sma200=sma(close,200)
bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values; bu=bm+bs*2; bl=bm-bs*2
ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
sk=pd.Series(rk).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
ef=ema(close,12); es=ema(close,26); ml=ef-es; ms=ema(ml,9); mh=ml-ms
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
ae=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

# ---- ADAPTIVE ENGINE ----
class AF:
    def __init__(self,n,init=50): self.name=n; self.score=init; self.blocks=0; self.g=0; self.b=0
    def active(self): return self.blocks<5 or self.score>=35
    def record(self): self.blocks+=1
    def fb(self,w): 
        if w: self.score=max(10,self.score-10); self.b+=1
        else: self.score=min(100,self.score+8); self.g+=1

# V1: OLD BOT (no filters, original style)
bal1=10000; trades1=[]; pos1=None
for i in range(400,n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]; sk_i=sk[i]; sd_i=sd[i]
    psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
    atr=atr14[i] if not np.isnan(atr14[i]) else 10
    if np.isnan(ml_i): continue
    
    if pos1 is not None:
        sl_hit=(pos1[0]=='B' and px<=pos1[1]) or (pos1[0]=='S' and px>=pos1[1])
        tp_hit=(pos1[0]=='B' and px>=pos1[2]) or (pos1[0]=='S' and px<=pos1[2])
        if sl_hit or tp_hit:
            ep=pos1[1] if sl_hit else pos1[2]
            pnl=(ep-pos1[3])*100*pos1[4] if pos1[0]=='B' else (pos1[3]-ep)*100*pos1[4]
            bal1+=pnl; trades1.append(pnl); pos1=None
        continue
    
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    if trend_up and rsi>=72 and not (psk>psd and sk_i<sd_i):
        sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
        lots=max(0.01,round((bal1*0.02)/(sld*100),2))
        pos1=('B',sl,tp,px,lots)
    elif trend_up and 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i:
        sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
        lots=max(0.01,round((bal1*0.02)/(sld*100),2))
        pos1=('B',sl,tp,px,lots)
    elif not (not np.isnan(maf) and px>maf):
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml_i<ms_i and mh_i<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if checks>=3:
            sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
            lots=max(0.01,round((bal1*0.02)/(sld*100),2))
            pos1=('S',sl,tp,px,lots)

if pos1 is not None:
    ep=close[-1]; pnl=(ep-pos1[3])*100*pos1[4] if pos1[0]=='B' else (pos1[3]-ep)*100*pos1[4]
    bal1+=pnl; trades1.append(pnl)

# V2: OPTIMIZED BOT (all filters + adaptive)
adx_f=AF("ADX",55); sess_f=AF("Session",40); macd_f=AF("MACD",35); overlap_f=AF("Overlap",35)
bal2=10000; trades2=[]; pos2=None; sim2=[]

for i in range(400,n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]; sk_i=sk[i]; sd_i=sd[i]
    psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
    hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
    if np.isnan(ml_i): continue
    
    if pos2 is not None:
        sl_hit=(pos2[0]=='B' and px<=pos2[1]) or (pos2[0]=='S' and px>=pos2[1])
        tp_hit=(pos2[0]=='B' and px>=pos2[2]) or (pos2[0]=='S' and px<=pos2[2])
        if sl_hit or tp_hit:
            ep=pos2[1] if sl_hit else pos2[2]
            pnl=(ep-pos2[3])*100*pos2[4] if pos2[0]=='B' else (pos2[3]-ep)*100*pos2[4]
            bal2+=pnl; trades2.append(pnl)
            for s in sim2[-20:]:
                for fn,f in [('adx',adx_f),('session',sess_f),('macd',macd_f),('overlap',overlap_f)]:
                    if s.get(fn): f.fb(s['w'])
            sim2=[]; pos2=None
        continue
    
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    buy_ok=False; sell_ok=False
    
    if trend_up:
        if 68<=rsi<=80 and not (psk>psd and sk_i<sd_i): buy_ok=True
        elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
    
    if buy_ok:
        blocked=False; bd={}
        if adx_f.active() and adx<25: blocked=True; bd['adx']=True; adx_f.record()
        if sess_f.active() and not (8<=hr<22): blocked=True; bd['session']=True; sess_f.record()
        if blocked:
            ww=False; sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            for j in range(i+1,min(i+300,n)):
                if high[j]>=tp: ww=True; break
                if low[j]<=sl: break
            sim2.append({**{k:v for k,v in bd.items()},'w':ww})
        else:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal2*0.02)/(sld*100),2))
            pos2=('B',sl,tp,px,lots); continue
    
    # SELL
    if not (not np.isnan(maf) and px>maf):
        if ml_i > ms_i: continue  # MACD gate
        blocked=False; bd={}
        if adx_f.active() and adx<25: blocked=True; bd['adx']=True; adx_f.record()
        if sess_f.active() and not (8<=hr<22): blocked=True; bd['session']=True; sess_f.record()
        if overlap_f.active() and (13<=hr<17): blocked=True; bd['overlap']=True; overlap_f.record()
        
        checks=0
        if 45<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml_i<ms_i and mh_i<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl[i])/bl[i]>0.005: checks+=1
        if px<bm[i]: checks+=1
        
        if checks>=3:
            if blocked:
                ww=False; sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                for j in range(i+1,min(i+300,n)):
                    if low[j]<=tp: ww=True; break
                    if high[j]>=sl: break
                sim2.append({**{k:v for k,v in bd.items()},'w':ww})
            else:
                sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                lots=max(0.01,round((bal2*0.02)/(sld*100),2))
                pos2=('S',sl,tp,px,lots)

if pos2 is not None:
    ep=close[-1]; pnl=(ep-pos2[3])*100*pos2[4] if pos2[0]=='B' else (pos2[3]-ep)*100*pos2[4]
    bal2+=pnl; trades2.append(pnl)

# ---- RESULTS ----
w1=[t for t in trades1 if t>0]; w2=[t for t in trades2 if t>0]
net1=sum(trades1); net2=sum(trades2)
wr1=len(w1)/len(trades1)*100 if trades1 else 0
wr2=len(w2)/len(trades2)*100 if trades2 else 0

print(f"\n{'='*80}")
print("  OUT-OF-SAMPLE RESULTS (Aug 2025 - Jan 2026)")
print(f"{'='*80}")
print(f"\n  V1 (OLD - no filters, original logic):")
print(f"    ${net1:+,.0f} | {len(trades1)}t | WR={wr1:.0f}%")
print(f"\n  V2 (OPTIMIZED - all filters, adaptive):")
print(f"    ${net2:+,.0f} | {len(trades2)}t | WR={wr2:.0f}%")
print(f"    Filters: ADX={adx_f.score:.0f}({'ON' if adx_f.active() else 'OFF'}) Session={sess_f.score:.0f}({'ON' if sess_f.active() else 'OFF'}) MACD=ON Overlap={overlap_f.score:.0f}({'ON' if overlap_f.active() else 'OFF'})")

diff = net2 - net1
pct = diff/abs(net1)*100 if net1 != 0 else 0
print(f"\n  Difference: ${diff:+,.0f} ({pct:+.0f}%)")
print(f"  Verdict: {'UPGRADE' if diff>0 else 'DOWNGRADE' if diff<0 else 'NEUTRAL'}")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
