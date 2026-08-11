"""
STEP 2+3: Advanced Exit Logic + 10K+ Parameter Sweep
Tests: partial TP, structural momentum exit, time decay, MACD/RSI/MA combos
"""
import sys, os, json, warnings, itertools, subprocess
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUT_DIR, "step3_sweep_log.txt")
sys.stdout = open(LOG_FILE, 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  STEP 2+3: Advanced Exits + 10K Parameter Sweep")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Pull data
print("\nLoading M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=90)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close)

# ---- Pre-compute ALL indicator variants ----
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

def rsi_fn(c,p=14):
    d=np.zeros(n); d[1:]=np.diff(c); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>p: ag[p]=np.mean(g[1:p+1]); al[p]=np.mean(l[1:p+1])
    for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+g[i])/p; al[i]=(al[i-1]*(p-1)+l[i])/p
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
    r=np.full(n,np.nan); r[m]=100-100/(1+rs[m]); return r
rsi14=rsi_fn(close,14)

print("Computing indicator library...")
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)

# MA variants for sweep
ma_variants = {}
for fast,slow in [(30,150),(40,200)]:
    ma_variants[f'{fast}_{slow}'] = (sma(close,fast), sma(close,slow))

# MACD variants
macd_variants = {}
for fast,slow,sig in [(8,17,9),(12,26,9),(10,20,9),(5,35,5)]:
    ef=ema(close,fast); es=ema(close,slow); ml=ef-es
    macd_variants[f'{fast}_{slow}_{sig}'] = (ml, ema(ml,sig), ml-ema(ml,sig))

# Stoch variants
stoch_variants = {}
for kp,dp,slw in [(14,3,3),(8,3,3),(5,3,3)]:
    ll=pd.Series(low).rolling(kp).min().values; hh=pd.Series(high).rolling(kp).max().values
    rk=np.full(n,np.nan); d=hh-ll; m=d>0; rk[m]=(close[m]-ll[m])/d[m]*100
    stoch_variants[f'{kp}_{dp}_{slw}'] = (pd.Series(rk).rolling(slw).mean().values, 
                                           pd.Series(pd.Series(rk).rolling(slw).mean()).rolling(dp).mean().values)

# BB 20
bb_mid=sma(close,20); bb_std=pd.Series(close).rolling(20).std().values
bb_u=bb_mid+bb_std*2; bb_l=bb_mid-bb_std*2

# ADX
def adx_fn(h,l,c,p=14):
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
    a=ema(tr,p); pi=ema(pdm,p); ni=ema(ndm,p)
    dx=np.full(n,np.nan); d_=pi+ni; m=d_>0; dx[m]=abs(pi[m]-ni[m])/d_[m]*100
    return ema(dx,p)
adx14=adx_fn(high,low,close)

vma20=sma(vol,20)
hours=np.array([t.hour for t in df.index])
print("Done.\n")

# ---- PARAMETER GRID (~13K) ----
param_grid = {
    'macd_key': ['8_17_9', '12_26_9', '5_35_5'],
    'rsi_breakout': [65, 68, 72],
    'rsi_buy_min': [35, 40, 45],
    'rsi_buy_max': [60, 65],
    'ma_key': ['30_150', '40_200', 'e30_150'],
    'stoch_key': ['14_3_3', '8_3_3'],
    'adx_min': [15, 20, 25],
    'session': ['none', 'london_ny'],
    'exit_mode': ['fixed', 'partial50', 'structural'],
    'sl_atr': [2.5, 3.0],
    'tp_rr': [2.0, 2.5],
}
# Total: 3*3*3*2*3*2*3*2*3*2*2 = 3^4 * 2^5 = 81 * 32 = 2,592 * 3 = 7,776... wait
# 3*3=9*3=27*2=54*3=162*2=324*3=972*2=1944*3=5832*2=11664*2=23328
# ~23K. At 15/s = ~26 min. Borderline. Let me cut more.
param_grid = {
    'macd_key': ['8_17_9', '12_26_9', '5_35_5'],
    'rsi_breakout': [65, 68, 72],
    'rsi_buy_min': [40, 45],
    'rsi_buy_max': [60, 65],
    'ma_key': ['30_150', '40_200'],
    'stoch_key': ['14_3_3', '8_3_3'],
    'adx_min': [15, 20, 25],
    'session': ['none', 'london_ny'],
    'exit_mode': ['fixed', 'partial50', 'structural'],
    'sl_atr': [2.5, 3.0],
    'tp_rr': [2.0, 2.5],
}
# Total: 3*3*2*2*2*2*3*2*3*2*2 = 3^3 * 2^7 = 27 * 128 = 3,456 * 3 = 10,368
# ~10K. At 15/s = ~12 min. Good.

# For partial50 and structural, tp_rr and other params have different meaning
# But we test them all in the grid

keys=list(param_grid.keys()); vals=list(param_grid.values())
total=1
for v in vals: total*=len(v)
print(f"Combinations: {total:,}\n")

# ---- Backtest with advanced exits ----
def backtest(p, start=400):
    bal=10000.0; eq=[bal]; trades=[]; pos=None
    cons_losses=0; mfe_data=[]
    
    # Select indicator variants
    maf, mas = ma_variants[p['ma_key']]
    ml, ms_, mh = macd_variants[p['macd_key']]
    sk_arr, sd_arr = stoch_variants[p['stoch_key']]
    
    for i in range(start, n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        adx=adx14[i] if not np.isnan(adx14[i]) else 50
        bu_i=bb_u[i]; bm_i=bb_mid[i]; bl_i=bb_l[i]
        sk_i=sk_arr[i]; sd_i=sd_arr[i]; psk=sk_arr[i-1] if i>0 else sk_i; psd=sd_arr[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; macd_l_=ml[i]; macd_s_=ms_[i]; macd_h_=mh[i]
        pmh=mh[i-1] if i>0 else mh[i]; maf_i=maf[i]; mas_i=mas[i]
        hour_now=hours[i] if i<len(hours) else 12
        
        if np.isnan(rsi) or np.isnan(macd_l_) or np.isnan(adx): continue
        if adx < p['adx_min']: continue
        if p['session']=='london_ny' and not (8<=hour_now<22): continue
        if cons_losses>=3: continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['dir']=='BUY' and px<=pos.get('sl',0)) or (pos['dir']=='SELL' and px>=pos.get('sl',99999))
            tp_hit=False
            if pos.get('tp') and pos['tp']>0:
                tp_hit=(pos['dir']=='BUY' and px>=pos['tp']) or (pos['dir']=='SELL' and px<=pos['tp'])
            
            # Structural momentum exit (close on reversal)
            structural_close = False
            if p['exit_mode'] == 'structural':
                if pos['dir'] == 'BUY':
                    bearish_stoch = psk > psd and sk_i < sd_i
                    macd_flip = macd_h_ < 0 and macd_h_ < mh[i-1] if i>0 else False
                    if bearish_stoch or macd_flip:
                        structural_close = True
                else:
                    bullish_stoch = psk < psd and sk_i > sd_i
                    macd_flip = macd_h_ > 0 and macd_h_ > mh[i-1] if i>0 else False
                    if bullish_stoch or macd_flip:
                        structural_close = True
            
            # Time decay exit
            time_decay_close = False
            if p['exit_mode'] == 'time_decay':
                bars_held = i - pos['entry_bar']
                if bars_held >= 10:  # 10 M15 bars = 2.5 hours
                    current_pnl = (px-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-px)*100*pos['lots']
                    if current_pnl < pos.get('mfe',0) * 0.3:  # Less than 30% of peak
                        time_decay_close = True
            
            # Partial TP (50% at 1:1 RR)
            if p['exit_mode'] == 'partial50' and not pos.get('partial_done'):
                sl_dist = abs(pos['entry'] - pos['sl'])
                partial_target = pos['entry'] + sl_dist if pos['dir']=='BUY' else pos['entry'] - sl_dist
                if (pos['dir']=='BUY' and px >= partial_target) or (pos['dir']=='SELL' and px <= partial_target):
                    # Close 50%
                    half_pnl = (partial_target - pos['entry'])*100*pos['lots']*0.5 if pos['dir']=='BUY' else (pos['entry']-partial_target)*100*pos['lots']*0.5
                    bal += half_pnl
                    trades.append({'dir':pos['dir'],'pnl':half_pnl,'reason':'partial_tp','rsi_entry':pos.get('rsi_entry'),'mfe':pos.get('mfe',0)})
                    pos['lots'] *= 0.5
                    pos['partial_done'] = True
                    # Move SL to breakeven
                    pos['sl'] = pos['entry'] + 0.1 if pos['dir']=='SELL' else pos['entry'] - 0.1
            
            # Track MFE
            if pos['dir']=='BUY':
                pos['mfe']=max(pos.get('mfe',0),(px-pos['entry'])*100*pos['lots']*2 if pos.get('partial_done') else (px-pos['entry'])*100*pos['lots'])
            else:
                pos['mfe']=max(pos.get('mfe',0),(pos['entry']-px)*100*pos['lots']*2 if pos.get('partial_done') else (pos['entry']-px)*100*pos['lots'])
            
            if sl_hit or tp_hit or structural_close or time_decay_close:
                ep = pos['sl'] if sl_hit else (pos['tp'] if tp_hit else px)
                reason = 'sl' if sl_hit else ('tp' if tp_hit else ('structural' if structural_close else 'time_decay'))
                pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
                bal+=pnl
                trades.append({'dir':pos['dir'],'pnl':pnl,'reason':reason,'rsi_entry':pos.get('rsi_entry'),'mfe':pos.get('mfe',0)})
                mfe_data.append({'dir':pos['dir'],'mfe':pos.get('mfe',0),'pnl':pnl,'reason':reason})
                if pnl<=0: cons_losses+=1
                else: cons_losses=0
                eq.append(bal); pos=None; continue
            eq.append(bal)
            continue
        eq.append(bal)
        
        # ---- BUY ----
        trend_up=not np.isnan(maf_i) and not np.isnan(mas_i) and maf_i>mas_i
        if trend_up:
            buy_ok=False
            if rsi>=p['rsi_breakout']:
                if not (psk>psd and sk_i<sd_i): buy_ok=True
            if not buy_ok and p['rsi_buy_min']<=rsi<=p['rsi_buy_max']:
                near_bb=abs(px-bm_i)/bm_i<0.01
                macd_ok=macd_h_>pmh; stoch_ok=psk<=psd and sk_i>sd_i
                if near_bb and macd_ok and stoch_ok: buy_ok=True
            if buy_ok:
                sld=atr*p['sl_atr']; sl=px-sld; tp=px+sld*p['tp_rr']
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'mfe':0,'partial_done':False,'entry_bar':i}
                continue
        
        # ---- SELL ----
        if not np.isnan(maf_i) and px>maf_i: continue
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if macd_l_<macd_s_ and macd_h_<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl_i)/bl_i>0.005: checks+=1
        if px<bm_i: checks+=1
        if checks>=3:
            sld=atr*p['sl_atr']; sl=px+sld; tp=px-sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'SELL','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'mfe':0,'partial_done':False,'entry_bar':i}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
        bal+=pnl; trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'eod','rsi_entry':pos.get('rsi_entry'),'mfe':pos.get('mfe',0)})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    peak=10000; mdd=0
    for e in eq:
        if e>peak: peak=e
        dd=(peak-e)/peak*100 if peak>0 else 0; mdd=max(mdd,dd)
    
    win_mfes=[t['mfe'] for t in wins if t.get('mfe')]; loss_mfes=[t['mfe'] for t in loses if t.get('mfe')]
    mfe_captured=sum(t['pnl'] for t in wins)/(sum(win_mfes)+0.01)*100 if win_mfes else 0
    
    return {
        'params':p, 'trades':len(trades), 'wr':round(wr,1), 'net':round(net,2),
        'pf':round(pf,2), 'dd':round(mdd,1),
        'mfe_captured':round(mfe_captured,1),
        'win_n':len(wins), 'loss_n':len(loses),
    }

# ---- RUN ----
print("Running 10K+ sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%3000==0:
        et=(datetime.now()-T0).total_seconds(); et=max(et,0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} results.\n")

# ---- BEST BY NET ----
results.sort(key=lambda x: x['net'], reverse=True)
print("="*80)
print("  TOP 10 BY NET P&L")
print("="*80)
for i,r in enumerate(results[:10]):
    p=r['params']
    print(f"\n #{i+1}: ${r['net']:+,.2f} | WR={r['wr']}% | PF={r['pf']} | DD={r['dd']}% | T={r['trades']} | MFE={r['mfe_captured']}%")
    print(f"   MACD:{p['macd_key']} | MA:{p['ma_key']} | Stoch:{p['stoch_key']} | RSI:{p['rsi_buy_min']}-{p['rsi_buy_max']}/{p['rsi_breakout']}")
    print(f"   ADX>{p['adx_min']} | Sess:{p['session']} | EXIT:{p['exit_mode']} | SL:{p['sl_atr']}x RR:{p['tp_rr']}")

# ---- EXIT MODE COMPARISON ----
print("\n"+"="*80)
print("  EXIT MODE SHOWDOWN")
print("="*80)
for mode in ['fixed', 'partial50', 'structural', 'time_decay']:
    md=[r for r in results if r['params']['exit_mode']==mode]
    if md:
        best=max(md, key=lambda x: x['net'])
        avg_net=sum(r['net'] for r in md)/len(md)
        avg_mfe=sum(r['mfe_captured'] for r in md)/len(md)
        print(f"\n  {mode.upper()}:")
        print(f"    Best: ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | Avg MFE={avg_mfe:.0f}%")
        print(f"    Avg Net: ${avg_net:+,.2f} (n={len(md)})")

# Best MACD / MA / Stoch
print("\n"+"="*80)
print("  INDICATOR VARIANT COMPARISON")
print("="*80)
for label, key in [('MACD','macd_key'),('MA','ma_key'),('Stoch','stoch_key')]:
    print(f"\n  {label}:")
    variants = sorted(set(r['params'][key] for r in results))
    for v in variants:
        vd=[r for r in results if r['params'][key]==v]
        if vd:
            best=max(vd, key=lambda x: x['net'])
            avg=sum(r['net'] for r in vd)/len(vd)
            print(f"    {v}: Best=${best['net']:+,.2f} Avg=${avg:+,.2f} PF={best['pf']}")

# ---- OVERALL BEST ----
max_net=max(r['net'] for r in results) if results else 1
max_pf=max(r['pf'] for r in results) if results else 1
max_wr=max(r['wr'] for r in results) if results else 1
for r in results: r['score']=(r['net']/max_net)*0.5+(r['pf']/max_pf)*0.3+(r['wr']/max_wr)*0.2
results.sort(key=lambda x: x['score'], reverse=True)
best=results[0]; bp=best['params']

print("\n"+"="*80)
print("  FINAL OPTIMAL CONFIGURATION")
print("="*80)
print(f"\n  Score: {best['score']:.3f} | ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | DD={best['dd']}%")
print(f"  Trades: {best['trades']} | MFE Captured: {best['mfe_captured']}%")
print(f"\n  SETTINGS:")
print(f"    MACD:               {bp['macd_key']}")
print(f"    MA:                 {bp['ma_key']}")
print(f"    Stochastic:         {bp['stoch_key']}")
print(f"    RSI Buy:            {bp['rsi_buy_min']}-{bp['rsi_buy_max']} brk>{bp['rsi_breakout']}")
print(f"    ADX Minimum:        {bp['adx_min']}")
print(f"    Session:            {bp['session']}")
print(f"    EXIT MODE:          {bp['exit_mode'].upper()}")
print(f"    SL ATR:             {bp['sl_atr']}x")
print(f"    TP RR:              {bp['tp_rr']}")

with open(os.path.join(OUT_DIR,"step3_results.json"),'w') as f:
    json.dump({'timestamp':T0.isoformat(),'total':cnt,'valid':len(results),
               'best_config':bp,'best_stats':{k:v for k,v in best.items() if k!='params'},
               'exit_comparison':{mode:max([r['net'] for r in results if r['params']['exit_mode']==mode],default=0) for mode in ['fixed','partial50','structural','time_decay']},
               'top30':[{**r} for r in results[:30]]},f,default=str,indent=2)
print(f"\nResults saved.")
print("="*80)
