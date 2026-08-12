"""
OUT-OF-SAMPLE VALIDATION: Backtest on different 6-month period
Tests if our optimizations generalize or overfit.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "oos_validation.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  OUT-OF-SAMPLE VALIDATION: Different 6-month period")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull as much history as MT5 allows (try 2024 data)
print("\nPulling historical M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)

# Test on: first 6 months available (try Jan-Jun 2025)
to_dt_oos = datetime(2025, 7, 1, tzinfo=timezone.utc)
from_dt_oos = datetime(2025, 1, 1, tzinfo=timezone.utc)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt_oos, to_dt_oos)

if rates is None or len(rates) < 1000:
    # Try 2024
    print("2025 data insufficient, trying 2024...")
    to_dt_oos = datetime(2024, 7, 1, tzinfo=timezone.utc)
    from_dt_oos = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt_oos, to_dt_oos)

if rates is None or len(rates) < 1000:
    # Try any available data
    print("Historical data limited, using earliest available 6 months")
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, 
                                  datetime(2020,1,1,tzinfo=timezone.utc), 
                                  datetime(2025,12,31,tzinfo=timezone.utc))
    if rates is not None and len(rates) > 0:
        total_months = len(rates) / (4*24*30)
        mid = len(rates) // 2
        rates = rates[:mid]  # First half

mt5.shutdown()

if rates is None or len(rates) < 500:
    print(f"ERROR: Only {len(rates) if rates else 0} candles"); sys.exit(1)

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"OOS Data: {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close); hours = np.array([t.hour for t in df.index])

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
bb_mid=sma(close,20); bb_std=pd.Series(close).rolling(20).std().values
bb_u=bb_mid+bb_std*2; bb_l=bb_mid-bb_std*2
ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
sk=pd.Series(rk).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
ef=ema(close,12); es=ema(close,26); macd_l=ef-es; macd_s=ema(macd_l,9); macd_h=macd_l-macd_s
vma20=sma(vol,20)

tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
a_e=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

print(f"Indicators ready.\n")

# ---- BACKTEST FUNCTION ----
def run_bt(config_name, use_macd_gate, use_overlap_block, use_adx_filter, use_session_filter):
    bal=10000; trades=[]; pos=None
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
        pmh=macd_h[i-1] if i>0 else mh; hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        
        if np.isnan(ml) or np.isnan(adx): continue
        
        # Pre-filters
        if use_adx_filter and adx < 25: continue
        if use_session_filter and not (8 <= hr < 22): continue
        if use_overlap_block and (13 <= hr < 17): 
            # Only block SELLs during overlap
            pass  # Handled below
        
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp'}); pos=None
            continue
        
        # BUY
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm)/bm<0.01 and mh>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        if buy_ok:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots}
            continue
        
        # SELL
        if not (not np.isnan(maf) and px>maf):
            # MACD mandatory gate
            if use_macd_gate and ml > ms_: continue
            
            # Overlap block (SELL only)
            if use_overlap_block and (13 <= hr < 17): continue
            
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml<ms_ and mh<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl)/bl>0.005: checks+=1
            if px<bm: checks+=1
            if checks>=3:
                sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})
    
    if len(trades)<3: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    
    peak=10000; mdd=0
    for e in [10000]: mdd=max(mdd,0)  # simplified
    
    return {'name':config_name, 'trades':len(trades), 'wr':round(wr,1), 'net':round(net,2), 
            'pf':round(pf,2), 'wins':len(wins), 'losses':len(loses)}

# ---- TEST ALL CONFIGS ----
configs = [
    # (name, macd_gate, overlap_block, adx_filter, session_filter)
    ("BASELINE (no filters)", False, False, False, False),
    ("+ ADX>25 only", False, False, True, False),
    ("+ Session only", False, False, False, True),
    ("+ MACD gate only", True, False, False, False),
    ("+ Overlap block only", False, True, False, False),
    ("+ ADX+Session (v2.0)", False, False, True, True),
    ("+ MACD gate + overlap", True, True, False, False),
    ("FULL v2.1 (all filters)", True, True, True, True),
]

print("=" * 80)
print("  OUT-OF-SAMPLE RESULTS")
print("=" * 80)
print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n")

results = []
for name, mg, ob, af, sf in configs:
    r = run_bt(name, mg, ob, af, sf)
    if r:
        results.append(r)
        status = "UPGRADE" if results and r['net'] > results[0]['net'] else ("BASELINE" if len(results)==1 else "downgrade")
        print(f"  {name}:")
        print(f"    ${r['net']:+,.0f} | {r['trades']}t | WR={r['wr']}% | PF={r['pf']} | W{r['wins']}/L{r['losses']}")

# Compare
print(f"\n{'='*80}")
print("  UPGRADE/DOWNGRADE SUMMARY")
print(f"{'='*80}")
baseline = results[0]
for r in results[1:]:
    diff = r['net'] - baseline['net']
    pct = diff/abs(baseline['net'])*100 if baseline['net'] != 0 else 0
    verdict = "UPGRADE" if diff > 0 else "DOWNGRADE" if diff < 0 else "NEUTRAL"
    print(f"  {r['name']}: ${diff:+,.0f} ({pct:+.0f}%) — {verdict}")

full = results[-1] if results[-1]['name'].startswith('FULL') else None
if full:
    print(f"\n  FINAL: Full v2.1 vs Baseline:")
    print(f"    Baseline: ${baseline['net']:+,.0f} {baseline['trades']}t WR={baseline['wr']}%")
    print(f"    v2.1:     ${full['net']:+,.0f} {full['trades']}t WR={full['wr']}% PF={full['pf']}")
    print(f"    Diff:     ${full['net']-baseline['net']:+,.0f}")

print(f"\n{'='*80}")
print(f"  DONE ({((datetime.now()-T0).total_seconds()):.0f}s)")
print(f"{'='*80}")
