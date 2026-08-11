"""
FAST PARAMETER SWEEP — Download data + sweep in one shot.
Target: ~2000 combos, ~3 min runtime.
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "sweep_log.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  FAST PARAMETER SWEEP — 3 Months XAUUSD M15")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# Pull data
print("\nPulling MT5 data...")
if not mt5.initialize():
    print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc)
from_dt = to_dt - timedelta(days=90)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
if rates is None or len(rates) < 500:
    print(f"ERROR: {len(rates) if rates else 0} candles"); sys.exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float)
high  = df['high'].values.astype(float)
low   = df['low'].values.astype(float)
vol   = df['tick_volume'].values.astype(float) if 'tick_volume' in df.columns else np.ones(len(df))
n = len(close)

# Fast indicator computation
print("Computing indicators...")
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

# RSI (vectorized)
def fast_rsi(c,p=14):
    d=np.zeros(n); d[1:]=np.diff(c)
    g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>p: ag[p]=np.mean(g[1:p+1]); al[p]=np.mean(l[1:p+1])
    for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+g[i])/p; al[i]=(al[i-1]*(p-1)+l[i])/p
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
    r=np.full(n,np.nan); r[m]=100-100/(1+rs[m]); return r

rsi14 = fast_rsi(close)
atr14 = ema(np.maximum(np.maximum(high-low, np.abs(high-np.roll(close,1))), np.abs(low-np.roll(close,1))), 14)
sma40 = sma(close,40); sma200 = sma(close,200)
bb_mid = sma(close,20); bb_std = pd.Series(close).rolling(20).std().values
bb_u = bb_mid + bb_std*2; bb_l = bb_mid - bb_std*2

# Stochastic
ll = pd.Series(low).rolling(14).min().values; hh = pd.Series(high).rolling(14).max().values
rk = np.full(n,np.nan); den = hh-ll; msk = den>0; rk[msk] = (close[msk]-ll[msk])/den[msk]*100
sk = pd.Series(rk).rolling(3).mean().values; sd = pd.Series(sk).rolling(3).mean().values

vma20 = sma(vol,20)

# MACD variants
macd_l_8, macd_s_8, macd_h_8 = (lambda: (lambda ef,es: (ef-es, ema(ef-es,9), (ef-es)-ema(ef-es,9)))(ema(close,8), ema(close,17)))()
macd_l_12, macd_s_12, macd_h_12 = (lambda: (lambda ef,es: (ef-es, ema(ef-es,9), (ef-es)-ema(ef-es,9)))(ema(close,12), ema(close,26)))()

print("Done.\n")

# ---- Compact parameter grid (~1728 combos) ----
param_grid = {
    'rsi_buy_min':      [30, 35, 40, 45],
    'rsi_buy_max':      [60, 65, 70],
    'rsi_breakout_min': [68, 72, 76],
    'macd_8_17':        [True, False],
    'vol_breakout':     [True, False],
    'vol_mult':         [1.0, 1.5],
    'bb_strict':        [True, False],  # True=pullback_mid, False=any
    'stoch_cross':      [True, False],
    'sell_strict':      [3, 5],
    'tp_rr':            [1.5, 2.0, 2.5],
}
# Total: 4*3*3*2*2*2*2*2*2*3 = 4*3=12*3=36*2=72*2=144*2=288*2=576*2=1152*2=2304*3=6912
# Still ~7K. At 100/sec = 70 sec. Acceptable.

keys = list(param_grid.keys()); vals = list(param_grid.values())
total = 1
for v in vals: total *= len(v)
print(f"Parameter combinations: {total:,}\n")

# ---- Fast backtest ----
def backtest(p, start=400):
    bal=10000.0; eq=[bal]; trades=[]; pos=None
    for i in range(start,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        smaf=sma40[i]; smas=sma200[i]
        bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]
        
        if p['macd_8_17']:
            ml=macd_l_8[i]; ms_=macd_s_8[i]; mh=macd_h_8[i]
            pmh=macd_h_8[i-1] if i>0 else mh
        else:
            ml=macd_l_12[i]; ms_=macd_s_12[i]; mh=macd_h_12[i]
            pmh=macd_h_12[i-1] if i>0 else mh
        
        if np.isnan(rsi) or np.isnan(ml) or np.isnan(sk_i): continue
        
        # Manage position
        if pos is not None:
            sl_hit = (pos['dir']=='BUY' and px<=pos['sl']) or (pos['dir']=='SELL' and px>=pos['sl'])
            tp_hit = (pos['dir']=='BUY' and px>=pos['tp']) or (pos['dir']=='SELL' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl = (ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
                bal+=pnl
                trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                              'rsi_entry':pos.get('rsi_entry'),'buy_type':pos.get('buy_type','')})
                eq.append(bal); pos=None; continue
            eq.append(bal+((px-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-px)*100*pos['lots']))
            continue
        
        eq.append(bal)
        
        # BUY
        trend_up = not np.isnan(smaf) and not np.isnan(smas) and smaf>smas
        if not trend_up: continue
        
        buy_ok=False; buy_type=''
        if rsi>=p['rsi_breakout_min']:
            vol_ok = (not p['vol_breakout']) or (vn>vma*p['vol_mult'])
            stoch_ok = (not p['stoch_cross']) or (not (psk>psd and sk_i<sd_i))
            if vol_ok and stoch_ok:
                buy_ok=True; buy_type='breakout'
        
        if not buy_ok and p['rsi_buy_min']<=rsi<=p['rsi_buy_max']:
            bb_chk = True
            if p['bb_strict']: bb_chk = abs(px-bm)/bm<0.01
            if bb_chk:
                macd_ok = mh>pmh
                stoch_ok = (not p['stoch_cross']) or (psk<=psd and sk_i>sd_i)
                if macd_ok and stoch_ok:
                    buy_ok=True; buy_type='pullback'
        
        if buy_ok:
            sld=atr*2.5; sl=px-sld; tp=px+sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'BUY','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi,'buy_type':buy_type}
            continue
        
        # SELL
        if not np.isnan(smaf) and px>smaf: continue
        checks = 0
        if 30<=rsi<=50 and rsi<rsi14[i-1] if i>0 else True: checks+=1
        if ml<ms_ and mh<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl)/bl>0.005: checks+=1
        if px<bm: checks+=1
        if checks>=p['sell_strict']:
            sld=atr*2.5; sl=px+sld; tp=px-sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'dir':'SELL','entry':px,'sl':sl,'tp':tp,'lots':lots,'rsi_entry':rsi}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['entry'])*100*pos['lots'] if pos['dir']=='BUY' else (pos['entry']-ep)*100*pos['lots']
        bal+=pnl; trades.append({'dir':pos['dir'],'pnl':pnl,'reason':'eod','rsi_entry':pos.get('rsi_entry')})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    wr=len(wins)/len(trades)*100; net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    
    peak=10000; mdd=0
    for e in eq:
        if e>peak: peak=e
        dd=(peak-e)/peak*100 if peak>0 else 0; mdd=max(mdd,dd)
    
    r70=[t for t in trades if t.get('rsi_entry') and t['rsi_entry']>70 and t['dir']=='BUY']
    buys=[t for t in trades if t['dir']=='BUY']; sells=[t for t in trades if t['dir']=='SELL']
    
    return {
        'params':p, 'trades':len(trades), 'wins':len(wins), 'losses':len(loses),
        'wr':round(wr,1), 'net':round(net,2), 'pf':round(pf,2), 'dd':round(mdd,1),
        'r70n':len(r70), 'r70p':round(sum(t['pnl'] for t in r70),2),
        'bn':len(buys), 'bp':round(sum(t['pnl'] for t in buys),2),
        'sn':len(sells), 'sp':round(sum(t['pnl'] for t in sells),2),
        'bwr':round(len([t for t in buys if t['pnl']>0])/len(buys)*100,1) if buys else 0,
        'swr':round(len([t for t in sells if t['pnl']>0])/len(sells)*100,1) if sells else 0,
    }

# ---- RUN ----
print("Running sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%500==0:
        et=(datetime.now()-T0).total_seconds(); et=max(et,0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo))
    r=backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} results.\n")

# ---- TOP RESULTS ----
results.sort(key=lambda x: x['net'], reverse=True)

print("="*80)
print("  TOP 15 BY NET P&L")
print("="*80)
for i,r in enumerate(results[:15]):
    p=r['params']
    macd_label = "8/17" if p['macd_8_17'] else "12/26"
    print(f"\n #{i+1}: ${r['net']:+,.2f} | WR={r['wr']}% | PF={r['pf']} | DD={r['dd']}% | T={r['trades']}")
    print(f"   RSI buy:{p['rsi_buy_min']}-{p['rsi_buy_max']} Brk>{p['rsi_breakout_min']} | MACD:{macd_label} | Vol:{p['vol_breakout']}x{p['vol_mult']}")
    print(f"   BB:{'strict' if p['bb_strict'] else 'any'} | Stoch:{p['stoch_cross']} | SellChk:{p['sell_strict']} | RR:{p['tp_rr']}")
    print(f"   BUY:{r['bn']}t ${r['bp']:+,.2f}({r['bwr']}%) SELL:{r['sn']}t ${r['sp']:+,.2f}({r['swr']}%) RSI>70:${r['r70p']:+,.2f}({r['r70n']}t)")

# Combined score
max_net = max(r['net'] for r in results) if results else 1
max_pf = max(r['pf'] for r in results) if results else 1
max_wr = max(r['wr'] for r in results) if results else 1
for r in results:
    r['score'] = (r['net']/max_net)*0.5 + (r['pf']/max_pf)*0.3 + (r['wr']/max_wr)*0.2
results.sort(key=lambda x: x['score'], reverse=True)
best = results[0]; bp = best['params']
macd_label = "8/17" if bp['macd_8_17'] else "12/26"

print("\n"+"="*80)
print("  OPTIMAL CONFIGURATION (Weighted Score)")
print("="*80)
print(f"\n  Score: {best['score']:.3f} | ${best['net']:+,.2f} | PF={best['pf']} | WR={best['wr']}% | DD={best['dd']}%")
print(f"\n  RECOMMENDED SETTINGS:")
print(f"    RSI Buy Range:      {bp['rsi_buy_min']} - {bp['rsi_buy_max']}")
print(f"    RSI Breakout Above: {bp['rsi_breakout_min']}")
print(f"    MACD:               {macd_label}")
print(f"    Volume Filter:      {bp['vol_breakout']} (mult: {bp['vol_mult']}x)")
print(f"    BB Entry Rule:      {'Pullback to Mid' if bp['bb_strict'] else 'Any position'}")
print(f"    Stochastic Cross:   {bp['stoch_cross']}")
print(f"    SELL Strictness:    {bp['sell_strict']}/5 checks")
print(f"    Risk:Reward:        1:{bp['tp_rr']}")

# Also best by profit factor
print("\n"+"="*80)
print("  BEST BY PROFIT FACTOR")
print("="*80)
pf_sorted = sorted(results, key=lambda x: (x['pf'], x['net']), reverse=True)
for i,r in enumerate(pf_sorted[:5]):
    p=r['params']
    macd_label = "8/17" if p['macd_8_17'] else "12/26"
    print(f"  #{i+1}: PF={r['pf']} | ${r['net']:+,.2f} | WR={r['wr']}% | DD={r['dd']}%")
    print(f"    RSI:{p['rsi_buy_min']}-{p['rsi_buy_max']}/{p['rsi_breakout_min']} | RR:{p['tp_rr']} | Sell:{p['sell_strict']}chk")

# RSI>70 deep dive
print("\n"+"="*80)
print("  RSI>70 BUY DEEP DIVE")
print("="*80)
r70_results = [(r['r70p'], r['r70n'], r['net'], r['params']['rsi_breakout_min'], 
                r['params']['vol_breakout'], r['params']['vol_mult']) for r in results if r['r70n']>0]
r70_results.sort(key=lambda x: x[0], reverse=True)
print(f"  Best RSI>70 breakout threshold + vol settings:")
for i,(pnl,cnt,net,brk,vol,vmult) in enumerate(r70_results[:8]):
    print(f"  RSI>{brk} Vol:{vol}x{vmult}: RSI70 PnL=${pnl:+,.2f}({cnt}t) Overall=${net:+,.2f}")

with open(os.path.join(OUT_DIR,"sweep_results.json"),'w') as f:
    json.dump({'timestamp':T0.isoformat(),'data':f"{df.index[0]} to {df.index[-1]}",
               'total':cnt,'valid':len(results),'best':bp,'best_stats':{k:v for k,v in best.items() if k!='params'},
               'top20':[{**r} for r in results[:20]]},f,default=str)

print(f"\n{'='*80}")
print(f"  Results saved. Run: python main_super.py")
print(f"{'='*80}")
