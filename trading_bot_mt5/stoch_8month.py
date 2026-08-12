
"""
8-MONTH VALIDATION + ADAPTIVE STOCH TEST
1. Verify Stochastic zones hold on a fresh 8-month period
2. Test if adaptive (self-adjusting) zones beat fixed zones
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "stoch_8month.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  8-MONTH VALIDATION + ADAPTIVE STOCH TEST")
print("=" * 80)

# Fresh 8-month period: Feb 2025 - Oct 2025 (not used before)
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2025,2,1,tzinfo=timezone.utc), datetime(2025,10,1,tzinfo=timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close); hours=np.array([t.hour for t in df.index])
print(f"\nData: {len(df)} candles: {df.index[0]} to {df.index[-1]}")

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

def run(mode):
    """mode: 'none' (no stoch zone), 'fixed' (35-65/50-80), 'wide' (30-70/40-80)"""
    bal=10000; trades=[]; pos=None
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]; sk_i=sk[i]; sd_i=sd[i]
        psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i) or np.isnan(sk_i): continue
        strong_trend = (not np.isnan(adx)) and adx >= 35
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append(pnl); pos=None
            continue
        
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if 68<=rsi<=80 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        
        # Stochastic zone for BUY
        if buy_ok:
            if mode == 'fixed' and not (35 <= sk_i <= 65): buy_ok=False
            elif mode == 'wide' and not (30 <= sk_i <= 70): buy_ok=False
        
        if buy_ok:
            if not strong_trend and not (8<=hr<22): buy_ok=False
            if buy_ok:
                sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos=('B',sl,tp,px,lots); continue
        
        if not (not np.isnan(maf) and px>maf):
            if not strong_trend:
                if ml_i > ms_i: continue
                if not (8<=hr<22): continue
            checks=0
            if 45<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml_i<ms_i and mh_i<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl[i])/bl[i]>0.005: checks+=1
            if px<bm[i]: checks+=1
            if checks>=3:
                # Stochastic zone for SELL
                zone_ok = True
                if mode == 'fixed': zone_ok = (50 <= sk_i <= 80)
                elif mode == 'wide': zone_ok = (40 <= sk_i <= 80)
                if zone_ok:
                    sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                    lots=max(0.01,round((bal*0.02)/(sld*100),2))
                    pos=('S',sl,tp,px,lots)
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append(pnl)
    wins=[t for t in trades if t>0]
    return {'net':round(sum(trades),2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1) if trades else 0}

# Test 3 modes
print(f"\n{'='*80}")
print("  RESULTS: 8-month fresh period (Feb-Oct 2025)")
print(f"{'='*80}")

base = run('none')
fixed = run('fixed')
wide = run('wide')

print(f"\n  {'Mode':<30} {'Net P&L':>12} {'Trades':>8} {'WR':>8}")
print(f"  {'Baseline (no Stoch zone)':<30} ${base['net']:+,.0f}   {base['trades']:<8} {base['wr']}%")
print(f"  {'Fixed zone (35-65/50-80)':<30} ${fixed['net']:+,.0f}   {fixed['trades']:<8} {fixed['wr']}%")
print(f"  {'Wide zone (30-70/40-80)':<30} ${wide['net']:+,.0f}   {wide['trades']:<8} {wide['wr']}%")

fixed_diff = fixed['net'] - base['net']
wide_diff = wide['net'] - base['net']
print(f"\n  Fixed zone vs baseline: ${fixed_diff:+,.0f} ({'IMPROVES' if fixed_diff>0 else 'HURTS'})")
print(f"  Wide zone vs baseline:  ${wide_diff:+,.0f} ({'IMPROVES' if wide_diff>0 else 'HURTS'})")

print(f"\n{'='*80}")
print("  VERDICT ON ADAPTIVE")
print(f"{'='*80}")
if fixed_diff > 0 and wide_diff > fixed_diff:
    print("  Fixed zone improves; wider zone improves MORE -> consider adaptive widening")
elif fixed_diff > 0:
    print("  Fixed zone improves and is robust -> keep fixed, no need for adaptive")
else:
    print("  Fixed zone does NOT improve on this period -> zone is regime-dependent, make adaptive")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
