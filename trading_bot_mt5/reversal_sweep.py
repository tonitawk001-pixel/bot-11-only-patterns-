"""
REVERSAL SWEEP: Tests reversal SELL path against baseline.
~3K combos. Compares reversal-enabled vs disabled.
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "reversal_sweep_log.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  REVERSAL PATH SWEEP — Tests overbought exhaustion SELL")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Pull data
print("\nPulling MT5 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=90)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
if rates is None or len(rates) < 500: print(f"ERROR"); sys.exit(1)

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float) if 'tick_volume' in df.columns else np.ones(len(df))
n = len(close)

# Indicators (same as before)
print("Computing indicators...")
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values
def fast_rsi(c,p=14):
    d=np.zeros(n); d[1:]=np.diff(c); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>p: ag[p]=np.mean(g[1:p+1]); al[p]=np.mean(l[1:p+1])
    for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+g[i])/p; al[i]=(al[i-1]*(p-1)+l[i])/p
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]; r=np.full(n,np.nan); r[m]=100-100/(1+rs[m]); return r

rsi14=fast_rsi(close)
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma40=sma(close,40); sma200=sma(close,200)
bb_mid=sma(close,20); bb_std=pd.Series(close).rolling(20).std().values; bb_u=bb_mid+bb_std*2; bb_l=bb_mid-bb_std*2
ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
sk=pd.Series(rk).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
vma20=sma(vol,20)
ef=ema(close,12); es=ema(close,26); macd_l=ef-es; macd_s=ema(macd_l,9); macd_h=macd_l-macd_s
print("Done.\n")

# Parameter grid - focus on reversal settings
param_grid = {
    'enable_reversal': [True, False],
    'rev_score_min': [2, 3, 4],  # How many checks needed for reversal
    'rev_vol_required': [True, False],  # Must volume be fading?
    'rev_stoch_required': [True, False],
    'rev_bb_required': [True, False],
    'rev_rsi_min': [65, 68, 70, 72],
    'regular_sell_checks': [3, 5],
    'tp_rr': [2.0, 2.5],
}
keys=list(param_grid.keys()); vals=list(param_grid.values())
total=1
for v in vals: total*=len(v)
print(f"Combinations: {total:,}\n")

def backtest(p, start=400):
    bal=10000.0; eq=[bal]; trades=[]; pos=None
    for i in range(start,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        smaf=sma40[i]; smas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
        pmh=macd_h[i-1] if i>0 else mh
        if np.isnan(rsi) or np.isnan(ml): continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['dir']=='BUY' and px<=pos['sl']) or (pos['dir']=='SELL' and px>=pos['sl'])
            tp_hit=(pos['dir']=='BUY' and px>=pos['tp']) or (pos['dir']=='SELL' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl = (ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
                bal+=pnl
                trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                              'rsi_entry':pos.get('rsi_entry'),'rev':pos.get('rev',False)})
                eq.append(bal); pos=None; continue
            eq.append(bal+((px-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-px)*100*pos['lots']))
            continue
        eq.append(bal)
        
        # BUY (unchanged - same as sweep-optimized)
        trend_up = not np.isnan(smaf) and not np.isnan(smas) and smaf>smas
        if not trend_up: continue
        buy_ok=False
        if rsi>=68:
            stoch_ok = not (psk>psd and sk_i<sd_i)
            if stoch_ok: buy_ok=True
        if not buy_ok and 40<=rsi<=60:
            bb_chk = abs(px-bm)/bm<0.01
            macd_ok = mh>pmh; stoch_ok = (psk<=psd and sk_i>sd_i)
            if bb_chk and macd_ok and stoch_ok: buy_ok=True
        if buy_ok:
            sld=atr*2.5; sl=px-sld; tp=px+sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'rev':False}
            continue
        
        # SELL - check reversal first, then regular
        sold=False; rev=False
        
        if p['enable_reversal'] and rsi>=p['rev_rsi_min']:
            score=1  # RSI condition met
            vol_ok=True; stoch_ok=True; bb_ok=True
            if p['rev_vol_required']: 
                vol_ok = vn < vma
                if vol_ok: score += 1
            if p['rev_stoch_required']: 
                stoch_ok = psk >= psd and sk_i < sd_i
                if stoch_ok: score += 1
            if p['rev_bb_required']: 
                bb_ok = abs(px - bu) / bu < 0.01
                if bb_ok: score += 1
            if score>=p['rev_score_min']: sold=True; rev=True
        
        if not sold:
            if not np.isnan(smaf) and px>smaf: continue
            checks=0
            if 30<=rsi<=50 and rsi<rsi14[i-1] if i>0 else True: checks+=1
            if ml<ms_ and mh<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl)/bl>0.005: checks+=1
            if px<bm: checks+=1
            if checks>=p['regular_sell_checks']: sold=True
        
        if sold:
            sld=atr*2.5; sl=px+sld; tp=px-sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'SELL','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'rev':rev}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
        bal+=pnl; trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'eod',
                                'rsi_entry':pos.get('rsi_entry'),'rev':pos.get('rev',False)})
    if len(trades)<5: return None
    
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    peak=10000; mdd=0
    for e in eq:
        if e>peak: peak=e
        dd=(peak-e)/peak*100 if peak>0 else 0; mdd=max(mdd,dd)
    
    rev_trades=[t for t in trades if t.get('rev')]
    buys=[t for t in trades if t['dir']=='BUY']; sells=[t for t in trades if t['dir']=='SELL']
    norm_sells=[t for t in sells if not t.get('rev')]
    
    return {
        'params':p, 'trades':len(trades), 'wins':len(wins), 'wr':round(wr,1),
        'net':round(net,2), 'pf':round(pf,2), 'dd':round(mdd,1),
        'rev_n':len(rev_trades), 'rev_pnl':round(sum(t['pnl'] for t in rev_trades),2),
        'sells_n':len(sells), 'sells_pnl':round(sum(t['pnl'] for t in sells),2),
        'norm_sells_n':len(norm_sells), 'norm_sells_pnl':round(sum(t['pnl'] for t in norm_sells),2),
    }

# Run
print("Running sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%500==0:
        et=(datetime.now()-T0).total_seconds(); et=max(et,0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} results.\n")

# Separate with and without reversal
with_rev = [r for r in results if r['params']['enable_reversal']]
without_rev = [r for r in results if not r['params']['enable_reversal']]

def avg(l,key): return sum(r[key] for r in l)/len(l) if l else 0

print("="*80)
print("  REVERSAL ON vs OFF COMPARISON")
print("="*80)

for label, data in [("REVERSAL ON", with_rev), ("REVERSAL OFF", without_rev)]:
    if not data: continue
    best=max(data, key=lambda x: x['net'])
    print(f"\n  {label}:")
    print(f"    Best Net: ${best['net']:+,.2f} | WR={best['wr']}% | PF={best['pf']} | DD={best['dd']}%")
    print(f"    Reversal trades: {best['rev_n']} | Rev PnL: ${best['rev_pnl']:+,.2f}")
    print(f"    All SELLs: {best['sells_n']}t ${best['sells_pnl']:+,.2f}")

# Best reversal config
print("\n"+"="*80)
print("  BEST REVERSAL CONFIGURATIONS")
print("="*80)
rev_sorted=sorted(with_rev, key=lambda x: (x['rev_pnl'], x['net']), reverse=True)
for i,r in enumerate(rev_sorted[:10]):
    p=r['params']
    print(f"  #{i+1}: Rev PnL=${r['rev_pnl']:+,.2f}({r['rev_n']}t) | Overall=${r['net']:+,.2f} | PF={r['pf']} | WR={r['wr']}%")
    print(f"    RSI>{p['rev_rsi_min']} | Score>={p['rev_score_min']} | Vol:{p['rev_vol_required']} | Stoch:{p['rev_stoch_required']} | BB:{p['rev_bb_required']} | RegSell:{p['regular_sell_checks']}chk")

# Did reversal improve?
best_with = max(with_rev, key=lambda x: x['net']) if with_rev else None
best_without = max(without_rev, key=lambda x: x['net']) if without_rev else None

print("\n"+"="*80)
print("  IMPROVEMENT ANALYSIS")
print("="*80)
if best_with and best_without:
    improvement = best_with['net'] - best_without['net']
    improvement_pct = (improvement / abs(best_without['net'])) * 100 if best_without['net'] != 0 else 0
    print(f"\n  Best WITHOUT reversal: ${best_without['net']:+,.2f} | PF={best_without['pf']}")
    print(f"  Best WITH reversal:    ${best_with['net']:+,.2f} | PF={best_with['pf']}")
    print(f"  Improvement:           ${improvement:+,.2f} ({improvement_pct:+.1f}%)")
    if improvement > 0:
        print(f"\n  >>> VERDICT: Reversal path IMPROVES the bot. Keep it.")
        bp = best_with['params']
        print(f"  >>> Optimal reversal settings: RSI>{bp['rev_rsi_min']}, Score>={bp['rev_score_min']}, "
              f"Vol:{bp['rev_vol_required']}, Stoch:{bp['rev_stoch_required']}, BB:{bp['rev_bb_required']}")
    else:
        print(f"\n  >>> VERDICT: Reversal path does NOT improve. Discard it.")

with open(os.path.join(OUT_DIR,"reversal_results.json"),'w') as f:
    json.dump({'timestamp':T0.isoformat(),'total':cnt,'valid':len(results),
               'best_with':{k:v for k,v in best_with.items() if k!='params'} if best_with else {},
               'best_without':{k:v for k,v in best_without.items() if k!='params'} if best_without else {},
               'improvement':improvement if best_with and best_without else 0},f,default=str)
print(f"\nResults saved.")
