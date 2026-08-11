"""
FINAL OPTIMIZATION — MFE Analysis + Exit Refinement + 10K Sweep
Tests: granular ADX, MFE-based TP, trailing stops, session variants
"""
import sys, os, json, warnings, itertools, subprocess
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUT_DIR, "final_sweep_log.txt")
sys.stdout = open(LOG_FILE, 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  FINAL OPTIMIZATION — MFE + Exit + 10K Sweep")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Pull data
print("\nPulling 3 months M15...")
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

# Indicators
print("Computing indicators...")
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

def rsi_fn(c,p=14):
    d=np.zeros(n); d[1:]=np.diff(c); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>p: ag[p]=np.mean(g[1:p+1]); al[p]=np.mean(l[1:p+1])
    for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+g[i])/p; al[i]=(al[i-1]*(p-1)+l[i])/p
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
    r=np.full(n,np.nan); r[m]=100-100/(1+rs[m]); return r

rsi14=rsi_fn(close)
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma40=sma(close,40); sma200=sma(close,200)
bb_mid=sma(close,20); bb_std=pd.Series(close).rolling(20).std().values
bb_u=bb_mid+bb_std*2; bb_l=bb_mid-bb_std*2
ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
sk=pd.Series(rk).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
vma20=sma(vol,20)
ef=ema(close,12); es=ema(close,26); macd_l=ef-es; macd_s=ema(macd_l,9); macd_h=macd_l-macd_s

# ADX
def adx_fn(h,l,c,p=14):
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
    atr14_=ema(tr,p); pdi=ema(pdm,p); ndi=ema(ndm,p)
    dx=np.full(n,np.nan); d=pdi+ndi; m=d>0; dx[m]=abs(pdi[m]-ndi[m])/d[m]*100
    return ema(dx,p)

adx14=adx_fn(high,low,close)
hours=np.array([t.hour for t in df.index])
print("Done.\n")

# ---- PARAMETER GRID (~10K) ----
param_grid = {
    # ADX granular
    'adx_min': [0, 12, 15, 18, 20, 22, 25],
    
    # Session variants
    'session': ['none', 'london_ny', 'overlap_only'],
    
    # Entry thresholds
    'rsi_buy_min': [35, 40, 45],
    'rsi_buy_max': [60, 65],
    'rsi_breakout': [68, 72],
    
    # MFE-based exit (NEW)
    'use_trail': [True, False],  # Trail instead of fixed TP
    'trail_atr': [0.5, 1.0, 1.5],  # Trail distance in ATR
    
    # Fixed exit params
    'tp_rr': [1.5, 2.0, 2.5],
    'sl_atr': [2.5, 3.0, 3.5],
    
    # Reversal paths
    'enable_reversal': [True, False],
    'rev_rsi_min': [65, 68],
}
keys=list(param_grid.keys()); vals=list(param_grid.values())
total=1
for v in vals: total*=len(v)
print(f"Combinations: {total:,}\n")

# ---- Backtest with MFE tracking ----
def backtest(p, start=400):
    bal=10000.0; eq=[bal]; trades=[]; pos=None
    cons_losses=0
    mfe_data = []  # Track max favorable excursion
    
    for i in range(start, n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        adx=adx14[i] if not np.isnan(adx14[i]) else 50
        bu_i=bb_u[i]; bm_i=bb_mid[i]; bl_i=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
        pmh=macd_h[i-1] if i>0 else mh; maf=sma40[i]; mas=sma200[i]
        hour_now=hours[i] if i<len(hours) else 12
        
        if np.isnan(rsi) or np.isnan(ml) or np.isnan(adx): continue
        
        # ADX filter
        if adx < p['adx_min']: continue
        
        # Session filter
        if p['session']=='london_ny' and not (8<=hour_now<22): continue
        if p['session']=='overlap_only' and not (13<=hour_now<17): continue
        
        if cons_losses>=3: continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['dir']=='BUY' and px<=pos['sl']) or (pos['dir']=='SELL' and px>=pos['sl'])
            tp_hit=False
            if pos.get('tp') and pos['tp']>0:
                tp_hit=(pos['dir']=='BUY' and px>=pos['tp']) or (pos['dir']=='SELL' and px<=pos['tp'])
            
            # Trail stop
            if not tp_hit and not sl_hit and p['use_trail']:
                trail_dist = atr * p['trail_atr']
                if pos['dir']=='BUY':
                    profit=px-pos['entry']
                    if profit>trail_dist*2:
                        new_sl=max(pos['sl'], px-trail_dist)
                        if new_sl>pos['sl']: pos['sl']=new_sl
                        if px<=new_sl: sl_hit=True
                else:
                    profit=pos['entry']-px
                    if profit>trail_dist*2:
                        new_sl=min(pos['sl'], px+trail_dist)
                        if new_sl<pos['sl']: pos['sl']=new_sl
                        if px>=new_sl: sl_hit=True
            
            # Track MFE
            if pos['dir']=='BUY':
                pos['mfe']=max(pos.get('mfe',0),(px-pos['entry'])*100*pos['lots'])
            else:
                pos['mfe']=max(pos.get('mfe',0),(pos['entry']-px)*100*pos['lots'])
            
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
                bal+=pnl
                trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                              'rsi_entry':pos.get('rsi_entry'),'mfe':pos.get('mfe',0),
                              'rev':pos.get('rev',False),'entry':pos['entry']})
                mfe_data.append({'dir':pos['dir'],'mfe':pos.get('mfe',0),'pnl':pnl,'reason':'sl' if sl_hit else 'tp'})
                if pnl<=0: cons_losses+=1
                else: cons_losses=0
                eq.append(bal); pos=None; continue
            eq.append(bal+((px-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-px)*100*pos['lots']))
            continue
        eq.append(bal)
        
        # ---- BUY ----
        trend_up=not np.isnan(maf) and not np.isnan(mas) and maf>mas
        if trend_up:
            buy_ok=False; buy_type=''
            if rsi>=p['rsi_breakout']:
                if not (psk>psd and sk_i<sd_i): buy_ok=True; buy_type='breakout'
            if not buy_ok and p['rsi_buy_min']<=rsi<=p['rsi_buy_max']:
                near_bb=abs(px-bm_i)/bm_i<0.01
                macd_ok=mh>pmh; stoch_ok=psk<=psd and sk_i>sd_i
                if near_bb and macd_ok and stoch_ok: buy_ok=True; buy_type='pullback'
            if buy_ok:
                sld=atr*p['sl_atr']; sl=px-sld
                tp=px+sld*p['tp_rr'] if not p['use_trail'] else px+sld*5.0  # wide TP for trail
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'mfe':0,'rev':False}
                continue
        
        # ---- SELL ----
        sold=False; is_rev=False
        if p['enable_reversal'] and rsi>=p['rev_rsi_min']:
            if psk>=psd and sk_i<sd_i: sold=True; is_rev=True
        if not sold:
            if not np.isnan(maf) and px>maf: continue
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml<ms_ and mh<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl_i)/bl_i>0.005: checks+=1
            if px<bm_i: checks+=1
            if checks>=3: sold=True
        if sold:
            sld=atr*p['sl_atr']; sl=px+sld
            tp=px-sld*p['tp_rr'] if not p['use_trail'] else px-sld*5.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'SELL','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'mfe':0,'rev':is_rev}
    
    if pos is not None:
        ep=close[-1]
        pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
        bal+=pnl; trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'eod',
                                'rsi_entry':pos.get('rsi_entry'),'mfe':pos.get('mfe',0),
                                'rev':pos.get('rev',False)})
    
    if len(trades)<5: return None
    
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    peak=10000; mdd=0
    for e in eq:
        if e>peak: peak=e
        dd=(peak-e)/peak*100 if peak>0 else 0; mdd=max(mdd,dd)
    
    # MFE analysis
    win_mfes=[t['mfe'] for t in wins if t.get('mfe')]; loss_mfes=[t['mfe'] for t in loses if t.get('mfe')]
    avg_win_mfe=sum(win_mfes)/len(win_mfes) if win_mfes else 0
    avg_loss_mfe=sum(loss_mfes)/len(loss_mfes) if loss_mfes else 0
    
    # MFE efficiency: what % of max favorable did we capture?
    mfe_captured = sum(t['pnl'] for t in wins)/(sum(win_mfes)+0.01)*100 if win_mfes else 0
    
    revs=[t for t in trades if t.get('rev')]
    buys=[t for t in trades if t['dir']=='BUY']; sells=[t for t in trades if t['dir']=='SELL']
    
    return {
        'params':p, 'trades':len(trades), 'wr':round(wr,1), 'net':round(net,2),
        'pf':round(pf,2), 'dd':round(mdd,1),
        'avg_win_mfe':round(avg_win_mfe,2), 'avg_loss_mfe':round(avg_loss_mfe,2),
        'mfe_captured_pct':round(mfe_captured,1),
        'rev_n':len(revs), 'rev_pnl':round(sum(t['pnl'] for t in revs),2),
        'buys_n':len(buys), 'buys_pnl':round(sum(t['pnl'] for t in buys),2),
        'sells_n':len(sells), 'sells_pnl':round(sum(t['pnl'] for t in sells),2),
    }

# ---- RUN ----
print("Running sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%2000==0:
        et=(datetime.now()-T0).total_seconds(); et=max(et,0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} valid results.\n")

# ---- ANALYSIS ----
results.sort(key=lambda x: x['net'], reverse=True)

print("="*80)
print("  TOP 10 BY NET P&L")
print("="*80)
for i,r in enumerate(results[:10]):
    p=r['params']
    trail_str = f"Trail:{p['trail_atr']}ATR" if p['use_trail'] else f"RR:{p['tp_rr']}"
    print(f"\n #{i+1}: ${r['net']:+,.2f} | WR={r['wr']}% | PF={r['pf']} | DD={r['dd']}% | T={r['trades']}")
    print(f"   ADX>{p['adx_min']} | Sess:{p['session']} | RSI:{p['rsi_buy_min']}-{p['rsi_buy_max']}/{p['rsi_breakout']}")
    print(f"   {trail_str} | SL:{p['sl_atr']}x | Rev:{p['enable_reversal']}>{p['rev_rsi_min']}")
    print(f"   BUY:{r['buys_n']}t ${r['buys_pnl']:+,.2f} SELL:{r['sells_n']}t ${r['sells_pnl']:+,.2f}")
    print(f"   MFE: win=${r['avg_win_mfe']:,.2f} loss=${r['avg_loss_mfe']:,.2f} captured={r['mfe_captured_pct']}%")

# Trail vs Fixed
print("\n"+"="*80)
print("  TRAILING STOP vs FIXED TP")
print("="*80)
for use_trail in [True, False]:
    tr_data=[r for r in results if r['params']['use_trail']==use_trail]
    if tr_data:
        best=max(tr_data, key=lambda x: x['net'])
        print(f"\n  {'TRAILING' if use_trail else 'FIXED TP'}:")
        print(f"    Best: ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | DD={best['dd']}%")
        print(f"    MFE captured: {best['mfe_captured_pct']}%")

# Best by MFE efficiency
print("\n"+"="*80)
print("  BEST MFE CAPTURE RATE")
print("="*80)
mfe_sorted=sorted(results, key=lambda x: x['mfe_captured_pct'], reverse=True)
for i,r in enumerate(mfe_sorted[:5]):
    p=r['params']
    print(f"  #{i+1}: MFE={r['mfe_captured_pct']}% | ${r['net']:+,.2f} | WR={r['wr']}% | {'TRAIL' if p['use_trail'] else 'FIXED'}")

# ADX sweet spot
print("\n"+"="*80)
print("  ADX SWEET SPOT")
print("="*80)
for adx_val in sorted(set(r['params']['adx_min'] for r in results)):
    adx_data=[r for r in results if r['params']['adx_min']==adx_val]
    if adx_data:
        best=max(adx_data, key=lambda x: x['net'])
        avg_net=sum(r['net'] for r in adx_data)/len(adx_data)
        print(f"  ADX>{adx_val}: Best=${best['net']:+,.2f} Avg=${avg_net:+,.2f} PF={best['pf']} T={best['trades']}")

# ---- OVERALL BEST ----
max_net=max(r['net'] for r in results) if results else 1
max_pf=max(r['pf'] for r in results) if results else 1
max_wr=max(r['wr'] for r in results) if results else 1
for r in results:
    r['score']=(r['net']/max_net)*0.5 + (r['pf']/max_pf)*0.3 + (r['wr']/max_wr)*0.2
results.sort(key=lambda x: x['score'], reverse=True)
best=results[0]; bp=best['params']

print("\n"+"="*80)
print("  OPTIMAL CONFIGURATION")
print("="*80)
print(f"\n  Score: {best['score']:.3f} | ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | DD={best['dd']}%")
print(f"  Trades: {best['trades']} | MFE captured: {best['mfe_captured_pct']}%")
print(f"\n  SETTINGS:")
print(f"    ADX Minimum:        {bp['adx_min']}")
print(f"    Session Filter:     {bp['session']}")
print(f"    RSI Buy:            {bp['rsi_buy_min']}-{bp['rsi_buy_max']} brk>{bp['rsi_breakout']}")
print(f"    Use Trail:          {bp['use_trail']} (trail: {bp['trail_atr']} ATR)")
print(f"    TP RR:              {bp['tp_rr']}")
print(f"    SL ATR:             {bp['sl_atr']}x")
print(f"    Reversal SELL:      {bp['enable_reversal']} (RSI>{bp['rev_rsi_min']})")

# Save results
RESULTS_FILE = os.path.join(OUT_DIR, "final_results.json")
with open(RESULTS_FILE,'w') as f:
    json.dump({
        'timestamp':T0.isoformat(),
        'data':f"{df.index[0]} to {df.index[-1]}",
        'total_combos':cnt,'valid':len(results),
        'best_config':bp,
        'best_stats':{k:v for k,v in best.items() if k!='params'},
        'top20':[{**r} for r in results[:20]],
        'mfe_analysis':{
            'best_mfe_captured':mfe_sorted[0]['mfe_captured_pct'] if mfe_sorted else 0,
            'trail_vs_fixed':{
                'trail_best':max([r['net'] for r in results if r['params']['use_trail']],default=0),
                'fixed_best':max([r['net'] for r in results if not r['params']['use_trail']],default=0),
            }
        }
    },f,default=str,indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("="*80)
sys.stdout.flush()

# ---- GIT PUSH ----
print("\nPushing to GitHub...")
GIT_DIR = os.path.dirname(OUT_DIR)
try:
    # Initialize git if needed
    if not os.path.exists(os.path.join(GIT_DIR, '.git')):
        subprocess.run(['git','init'], cwd=GIT_DIR, capture_output=True)
        subprocess.run(['git','remote','add','origin','https://github.com/tonitawk001-pixel/bot-7.git'],
                      cwd=GIT_DIR, capture_output=True)
    
    # Add and commit
    subprocess.run(['git','add','-A'], cwd=GIT_DIR, capture_output=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    subprocess.run(['git','commit','-m',f'Optimization sweep results — {ts}'], cwd=GIT_DIR, capture_output=True)
    
    # Push
    result = subprocess.run(['git','push','-u','origin','main'], cwd=GIT_DIR, 
                          capture_output=True, text=True, timeout=60)
    if result.returncode==0:
        print("Pushed to GitHub successfully!")
    else:
        # Try master branch
        result2 = subprocess.run(['git','push','-u','origin','master'], cwd=GIT_DIR,
                               capture_output=True, text=True, timeout=60)
        if result2.returncode==0:
            print("Pushed to GitHub (master branch)!")
        else:
            print(f"Push failed. You may need to set up credentials.")
            print(f"Error: {result.stderr[:200]}")
except Exception as e:
    print(f"Git error: {e}")
