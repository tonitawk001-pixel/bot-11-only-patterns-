"""
EXHAUSTIVE OPTIMIZATION SWEEP — 30-minute deep search
Tests: oversold BUY reversal, MA types, BB periods, ADX, session filters, entry delays
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "exhaustive_log.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  EXHAUSTIVE OPTIMIZATION — 30-min deep search")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Pull data
print("\nPulling 3 months M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=90)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
if rates is None or len(rates) < 500: print("ERROR"); sys.exit(1)

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close)

# ---- Indicator library ----
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

def compute_rsi(c,p=14):
    d=np.zeros(n); d[1:]=np.diff(c); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>p: ag[p]=np.mean(g[1:p+1]); al[p]=np.mean(l[1:p+1])
    for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+g[i])/p; al[i]=(al[i-1]*(p-1)+l[i])/p
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
    r=np.full(n,np.nan); r[m]=100-100/(1+rs[m]); return r

def compute_adx(h,l,c,p=14):
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n): 
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0
        ndm[i]=dn if dn>up and dn>0 else 0
    atr14=ema(tr,p); pdi=ema(pdm,p); ndi=ema(ndm,p)
    dx=np.full(n,np.nan); den=pdi+ndi; m=den>0; dx[m]=abs(pdi[m]-ndi[m])/den[m]*100
    return ema(dx,p)

# Pre-compute all indicator variants
print("Pre-computing indicators...")
rsi14 = compute_rsi(close,14)
rsi7 = compute_rsi(close,7)
rsi21 = compute_rsi(close,21)
atr14 = ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
adx14 = compute_adx(high,low,close,14)

# MA variants
sma40 = sma(close,40); sma200 = sma(close,200)
sma20 = sma(close,20); sma50 = sma(close,50)
ema40 = ema(close,40); ema200 = ema(close,200)

# BB variants
for period in [10,20,30]:
    mid = sma(close,period); std = pd.Series(close).rolling(period).std().values
    globals()[f'bb_u_{period}'] = mid + std*2
    globals()[f'bb_m_{period}'] = mid
    globals()[f'bb_l_{period}'] = mid - std*2

# Stoch
ll = pd.Series(low).rolling(14).min().values; hh = pd.Series(high).rolling(14).max().values
rk = np.full(n,np.nan); den = hh-ll; msk = den>0; rk[msk] = (close[msk]-ll[msk])/den[msk]*100
sk = pd.Series(rk).rolling(3).mean().values; sd = pd.Series(sk).rolling(3).mean().values

vma20 = sma(vol,20)

# MACD 12/26
ef=ema(close,12); es=ema(close,26); macd_l=ef-es; macd_s=ema(macd_l,9); macd_h=macd_l-macd_s

# Time of day info from index
hours = np.array([t.hour for t in df.index])
is_london = (hours >= 8) & (hours < 17)
is_ny = (hours >= 13) & (hours < 22)
is_overlap = is_london & is_ny
is_asian = (hours >= 23) | (hours < 8)
is_active = is_london | is_ny
print("Done.\n")

# ---- PARAMETER GRID (aggressive but bounded) ----
param_grid = {
    # Core entry (most impactful)
    'rsi_buy_min': [35, 40, 45],
    'rsi_buy_max': [55, 60, 65],
    'rsi_breakout': [65, 68, 72],
    
    # Reversal paths
    'enable_os_rev': [True, False],
    'os_rev_rsi_max': [25, 30],
    'enable_reversal': [True, False],
    'rev_rsi_min': [65, 68],
    
    # ADX filter
    'adx_min': [0, 15, 20, 25],
    
    # Session filter
    'session_filter': ['none', 'active_only'],
    
    # Risk
    'tp_rr': [2.0, 2.5],
    'sl_atr': [2.0, 3.0],
}
# Total: 3*3*3*2*2*2*2*4*2*2*2 = 3^3 * 2^6 * 4 * 2 = 27*64*4*2 = 13,824
# At ~20/s = ~11 min. Good.

keys = list(param_grid.keys()); vals = list(param_grid.values())
total = 1
for v in vals: total *= len(v)
print(f"Combinations: {total:,}\n")

# ---- Backtest ----
def backtest(p, start=400):
    bal=10000.0; eq=[bal]; trades=[]; pos=None
    cons_losses=0
    rsi_data = rsi14  # Fixed RSI 14
    bu = bb_u_20; bm = bb_m_20; bl = bb_l_20  # Fixed BB 20
    maf=sma40; mas=sma200  # Fixed SMA 40/200
    
    for i in range(start, n):
        if np.isnan(close[i]): continue
        
        px=close[i]; rsi=rsi_data[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        adx=adx14[i] if not np.isnan(adx14[i]) else 50
        maf_i=maf[i]; mas_i=mas[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
        pmh=macd_h[i-1] if i>0 else mh; bu_i=bu[i]; bm_i=bm[i]; bl_i=bl[i]
        hour_now = hours[i] if i < len(hours) else 12
        
        if np.isnan(rsi) or np.isnan(ml) or np.isnan(adx): continue
        
        # Session filter
        if p['session_filter'] == 'active_only' and not (8 <= hour_now < 22): continue
        if p['session_filter'] == 'overlap_only' and not (13 <= hour_now < 17): continue
        
        # ADX filter
        if adx < p['adx_min']: continue
        
        # Consecutive loss halt
        if cons_losses >= 3: continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['dir']=='BUY' and px<=pos['sl']) or (pos['dir']=='SELL' and px>=pos['sl'])
            tp_hit=(pos['dir']=='BUY' and px>=pos['tp']) or (pos['dir']=='SELL' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
                bal+=pnl
                trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                              'rsi_entry':pos.get('rsi_entry'),'os_rev':pos.get('os_rev',False),
                              'rev':pos.get('rev',False),'adx_entry':pos.get('adx_entry',0)})
                if pnl<=0: cons_losses+=1
                else: cons_losses=0
                eq.append(bal); pos=None; continue
            eq.append(bal+((px-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-px)*100*pos['lots']))
            continue
        eq.append(bal)
        
        # ---- OVERSOLD BUY REVERSAL (NEW) ----
        if p['enable_os_rev'] and rsi <= p['os_rev_rsi_max']:
            stoch_bull = psk <= psd and sk_i > sd_i
            near_lower = abs(px - bl_i) / bl_i < 0.01
            if stoch_bull and near_lower:
                sld = atr * p['sl_atr']
                sl = px - sld; tp = px + sld * p['tp_rr']
                lots = max(0.01, round((bal*0.02)/(sld*100),2))
                pos = {'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,
                       'rsi_entry':rsi,'os_rev':True,'rev':False,'adx_entry':adx}
                continue
        
        # ---- REGULAR BUY ----
        trend_up = not np.isnan(maf_i) and not np.isnan(mas_i) and maf_i > mas_i
        if trend_up:
            buy_ok = False
            # Breakout
            if rsi >= p['rsi_breakout']:
                if not (psk > psd and sk_i < sd_i): buy_ok = True  # no bearish cross
            # Pullback
            if not buy_ok and p['rsi_buy_min'] <= rsi <= p['rsi_buy_max']:
                near_ma_bb = (abs(px - bm_i)/bm_i < 0.01) or (abs(px - maf_i)/maf_i < 0.01)
                macd_ok = mh > pmh
                stoch_ok = psk <= psd and sk_i > sd_i
                if near_ma_bb and macd_ok and stoch_ok: buy_ok = True
            
            if buy_ok:
                sld = atr * p['sl_atr']; sl = px - sld; tp = px + sld * p['tp_rr']
                lots = max(0.01, round((bal*0.02)/(sld*100),2))
                pos = {'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,
                       'rsi_entry':rsi,'os_rev':False,'rev':False,'adx_entry':adx}
                continue
        
        # ---- REVERSAL SELL ----
        sold = False; is_rev = False
        if p['enable_reversal'] and rsi >= p['rev_rsi_min']:
            stoch_bear = psk >= psd and sk_i < sd_i
            if stoch_bear:
                sold = True; is_rev = True
        
        # ---- REGULAR SELL ----
        if not sold:
            if not np.isnan(maf_i) and px > maf_i: continue
            checks = 0
            if 30 <= rsi <= 50 and i > 0 and rsi < rsi_data[i-1]: checks += 1
            if ml < ms_ and mh < 0: checks += 1
            if psk >= psd and sk_i < sd_i: checks += 1
            if abs(px - bl_i)/bl_i > 0.005: checks += 1
            if px < bm_i: checks += 1
            if checks >= 3: sold = True  # fixed 3/5
        
        if sold:
            sld = atr * p['sl_atr']; sl = px + sld; tp = px - sld * p['tp_rr']
            lots = max(0.01, round((bal*0.02)/(sld*100),2))
            pos = {'dir':'SELL','entry':px,'sl':sl,'tp':tp,'lots':lots,
                   'rsi_entry':rsi,'os_rev':False,'rev':is_rev,'adx_entry':adx}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
        bal+=pnl; trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'eod',
                                'rsi_entry':pos.get('rsi_entry'),'os_rev':pos.get('os_rev',False),
                                'rev':pos.get('rev',False)})
    
    if len(trades) < 5: return None
    
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    peak=10000; mdd=0
    for e in eq:
        if e>peak: peak=e
        dd=(peak-e)/peak*100 if peak>0 else 0; mdd=max(mdd,dd)
    
    os_revs=[t for t in trades if t.get('os_rev')]
    revs=[t for t in trades if t.get('rev')]
    buys=[t for t in trades if t['dir']=='BUY']; sells=[t for t in trades if t['dir']=='SELL']
    
    return {
        'params':p, 'trades':len(trades), 'wr':round(wr,1), 'net':round(net,2),
        'pf':round(pf,2), 'dd':round(mdd,1),
        'os_rev_n':len(os_revs), 'os_rev_pnl':round(sum(t['pnl'] for t in os_revs),2),
        'rev_n':len(revs), 'rev_pnl':round(sum(t['pnl'] for t in revs),2),
        'buys_n':len(buys), 'buys_pnl':round(sum(t['pnl'] for t in buys),2),
        'sells_n':len(sells), 'sells_pnl':round(sum(t['pnl'] for t in sells),2),
    }

# ---- RUN ----
print("Running sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%1000==0:
        et=(datetime.now()-T0).total_seconds(); et=max(et,0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} results.\n")

# ---- ANALYSIS ----
results.sort(key=lambda x: x['net'], reverse=True)

print("="*80)
print("  TOP 15 BY NET P&L")
print("="*80)
for i,r in enumerate(results[:15]):
    p=r['params']
    rsi_lbl = "RSI14"
    print(f"\n #{i+1}: ${r['net']:+,.2f} | WR={r['wr']}% | PF={r['pf']} | DD={r['dd']}% | T={r['trades']}")
    print(f"   {rsi_lbl}: buy{p['rsi_buy_min']}-{p['rsi_buy_max']} brk>{p['rsi_breakout']} | ADX>{p['adx_min']} | Sess:{p['session_filter']}")
    print(f"   OSrev:{p['enable_os_rev']}(<{p['os_rev_rsi_max']}) Rev:{p['enable_reversal']}(>{p['rev_rsi_min']}) | RR:{p['tp_rr']} SL:{p['sl_atr']}x")
    print(f"   BUY:{r['buys_n']}t ${r['buys_pnl']:+,.2f} SELL:{r['sells_n']}t ${r['sells_pnl']:+,.2f}")
    if r['os_rev_n']: print(f"   OS-Revs:{r['os_rev_n']}t ${r['os_rev_pnl']:+,.2f}")
    if r['rev_n']: print(f"   Rev-SELLs:{r['rev_n']}t ${r['rev_pnl']:+,.2f}")

# Best with each feature
print("\n"+"="*80)
print("  FEATURE CONTRIBUTION ANALYSIS")
print("="*80)

for feature, key, true_val in [
    ("Oversold BUY Reversal", 'enable_os_rev', True),
    ("Overbought SELL Reversal", 'enable_reversal', True),
    ("ADX Filter Active", 'adx_min', lambda x: x>0),
    ("Session Filter Active", 'session_filter', lambda x: x!='none'),
]:
    with_f = [r for r in results if (r['params'][key]==true_val if not callable(true_val) else true_val(r['params'][key]))]
    without_f = [r for r in results if (r['params'][key]!=true_val if not callable(true_val) else not true_val(r['params'][key]))]
    
    best_w = max(with_f, key=lambda x: x['net']) if with_f else None
    best_wo = max(without_f, key=lambda x: x['net']) if without_f else None
    
    if best_w and best_wo:
        diff = best_w['net'] - best_wo['net']
        verdict = "IMPROVES" if diff > 0 else "HURTS"
        print(f"\n  {feature}:")
        print(f"    With:    ${best_w['net']:+,.2f} | PF={best_w['pf']} | WR={best_w['wr']}%")
        print(f"    Without: ${best_wo['net']:+,.2f} | PF={best_wo['pf']} | WR={best_wo['wr']}%")
        print(f"    Diff:    ${diff:+,.2f} — {verdict}")
    else:
        print(f"\n  {feature}: insufficient data")

# Best ADX
print("\n  ADX FILTER IMPACT:")
for adx_val in [0, 15, 20, 25]:
    adx_results = [r for r in results if r['params']['adx_min']==adx_val]
    if adx_results:
        best = max(adx_results, key=lambda x: x['net'])
        print(f"    ADX>{adx_val}: Best=${best['net']:+,.2f} PF={best['pf']} WR={best['wr']}% (n={len(adx_results)})")

# ---- OVERALL BEST ----
max_net = max(r['net'] for r in results) if results else 1
max_pf = max(r['pf'] for r in results) if results else 1
max_wr = max(r['wr'] for r in results) if results else 1
for r in results:
    r['score'] = (r['net']/max_net)*0.5 + (r['pf']/max_pf)*0.3 + (r['wr']/max_wr)*0.2
results.sort(key=lambda x: x['score'], reverse=True)
best = results[0]; bp = best['params']

print("\n"+"="*80)
print("  OPTIMAL CONFIGURATION")
print("="*80)
print(f"\n  Score: {best['score']:.3f} | ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | DD={best['dd']}%")
print(f"\n  RECOMMENDED SETTINGS:")
print(f"    RSI Period:          {bp['rsi_period']}")
print(f"    RSI Buy Range:       {bp['rsi_buy_min']} - {bp['rsi_buy_max']}")
print(f"    RSI Breakout Above:  {bp['rsi_breakout']}")
print(f"    MA Type:             {bp['ma_type']}")
print(f"    BB Period:           {bp['bb_period']}")
print(f"    ADX Minimum:         {bp['adx_min']}")
print(f"    Oversold BUY Rev:    {bp['enable_os_rev']} (RSI < {bp['os_rev_rsi_max']})")
print(f"    Overbought SELL Rev: {bp['enable_reversal']} (RSI > {bp['rev_rsi_min']})")
print(f"    Session Filter:      {bp['session_filter']}")
print(f"    SELL Checks:         {bp['sell_checks']}/5")
print(f"    Risk:Reward:         1:{bp['tp_rr']}")
print(f"    SL ATR Multiplier:   {bp['sl_atr']}x")
print(f"\n  RESULT: ${best['net']:+,.2f} from $10K over 3 months")
print(f"  OS-Reversal BUYs: {best['os_rev_n']}t ${best['os_rev_pnl']:+,.2f}")
print(f"  OB-Reversal SELLs: {best['rev_n']}t ${best['rev_pnl']:+,.2f}")

with open(os.path.join(OUT_DIR,"exhaustive_results.json"),'w') as f:
    json.dump({'timestamp':T0.isoformat(),'total':cnt,'valid':len(results),
               'best':bp,'best_stats':{k:v for k,v in best.items() if k!='params'},
               'top50':[{**r} for r in results[:50]]},f,default=str)
print(f"\nResults saved.")
print("="*80)
