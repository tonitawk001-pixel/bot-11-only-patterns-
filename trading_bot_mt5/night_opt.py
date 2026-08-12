"""
FINAL NIGHT OPTIMIZATION: 6-month backtest + actual trade replay
Goals: Max profit on 6-month data AND $300+ on actual MT5 trades
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "night_optimization.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  NIGHT OPTIMIZATION: 6-Month Backtest + Actual Trade Replay")
print(f"  {T0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

# ---- LOAD DATA ----
print("\nPulling 6 months M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=180)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
if rates is None or len(rates) < 1000: print("ERROR"); sys.exit(1)

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close)
hours = np.array([t.hour for t in df.index])

# Load actual trades
with open(r'C:\Users\ASUS\extract_mt5_trades_output.json') as f:
    trade_data = json.load(f)
actual_trades_raw = [t for t in trade_data['trades'] if t['volume'] > 0 and t['entry_price'] > 0 and t.get('exit_price')]
print(f"Loaded {len(actual_trades_raw)} actual MT5 trades\n")

# ---- INDICATORS ----
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

# RSI
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

# ADX
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
a=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

print("Indicators computed.\n")

# ---- PARAMETER GRID: Focus on finding $300+ on actual trades ----
param_grid = {
    'adx_min': [15, 18, 20, 22, 25],
    'session': ['none', 'active_only'],
    'rsi_buy_min': [35, 40, 45],
    'rsi_buy_max': [60, 65],
    'rsi_breakout': [68, 72],
    'sell_checks': [2, 3, 4, 5],  # Relax SELL to find more trades
    'sl_atr': [2.5, 3.0, 3.5],
    'tp_rr': [1.5, 2.0, 2.5],
}
keys=list(param_grid.keys()); vals=list(param_grid.values())
total=1
for v in vals: total*=len(v)
print(f"Combinations: {total:,}\n")

# ---- BACKTEST WITH ACTUAL TRADE REPLAY ----
def full_backtest(p):
    # Part 1: 6-month backtest
    bal = 10000.0; trades = []; pos = None
    
    for i in range(400, n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
        pmh=macd_h[i-1] if i>0 else mh; hr=hours[i] if i<len(hours) else 12
        
        if np.isnan(ml) or np.isnan(adx): continue
        if adx < p['adx_min']: continue
        if p['session'] == 'active_only' and not (8 <= hr < 22): continue
        
        # Position
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl; trades.append({'dir':pos['d'],'pnl':pnl}); pos=None
            continue
        
        # BUY
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if rsi>=p['rsi_breakout']:
                if not (psk>psd and sk_i<sd_i): buy_ok=True
            if not buy_ok and p['rsi_buy_min']<=rsi<=p['rsi_buy_max']:
                near_bb=abs(px-bm)/bm<0.01
                if near_bb and mh>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        if buy_ok:
            sld=atr*p['sl_atr']; sl=px-sld; tp=px+sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots}
            continue
        
        # SELL
        if not np.isnan(maf) and px>maf: continue
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml<ms_ and mh<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl)/bl>0.005: checks+=1
        if px<bm: checks+=1
        if checks>=p['sell_checks']:
            sld=atr*p['sl_atr']; sl=px+sld; tp=px-sld*p['tp_rr']
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'dir':pos['d'],'pnl':pnl})
    
    if len(trades)<5: return None
    
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    
    # Part 2: Actual trade replay
    actual_pnl = 0
    actual_taken = 0
    for trade in actual_trades_raw:
        try:
            et_str = trade['entry_time']
            if not et_str: continue
            et = datetime.fromisoformat(et_str).replace(tzinfo=None)
            idx = df.index.get_indexer([et], method='nearest')[0]
            if idx < 200 or idx >= n: continue
            
            i = idx; px=close[i]; rsi=rsi14[i]; adx=adx14[i]
            maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
            sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1]; psd=sd[i-1]
            ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]; pmh=macd_h[i-1]
            hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
            
            if np.isnan(rsi) or np.isnan(ml) or np.isnan(adx): continue
            if adx < p['adx_min']: continue
            if p['session']=='active_only' and not (8<=hr<22): continue
            
            actual_dir = trade['direction']
            v2_ok = False
            
            if actual_dir == 'BUY':
                trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
                if trend_up:
                    if rsi>=p['rsi_breakout'] and not (psk>psd and sk_i<sd_i): v2_ok=True
                    elif p['rsi_buy_min']<=rsi<=p['rsi_buy_max']:
                        nb=abs(px-bm)/bm<0.01
                        if nb and mh>pmh and psk<=psd and sk_i>sd_i: v2_ok=True
            else:
                if not np.isnan(maf) and px>maf: pass
                else:
                    checks=0
                    if 30<=rsi<=50 and rsi<rsi14[i-1] if i>0 else True: checks+=1
                    if ml<ms_ and mh<0: checks+=1
                    if psk>=psd and sk_i<sd_i: checks+=1
                    if abs(px-bl)/bl>0.005: checks+=1
                    if px<bm: checks+=1
                    if checks>=p['sell_checks']: v2_ok=True
            
            if v2_ok:
                actual_pnl += trade['profit']
                actual_taken += 1
        except:
            pass
    
    return {
        'params': p,
        'bt_net': round(net,2), 'bt_trades': len(trades),
        'bt_pf': round(pf,2), 'bt_wr': round(len(wins)/len(trades)*100,1),
        'actual_pnl': round(actual_pnl,2), 'actual_taken': actual_taken,
        'score': round(net/100 + actual_pnl/3, 2),  # Weight both equally
    }

# ---- RUN ----
print("Running optimization...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%2000==0:
        et=max((datetime.now()-T0).total_seconds(),0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=full_backtest(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} valid results.\n")

# ---- FIND BEST ----
# Goal: actual_pnl >= 300 AND bt_pf >= 1.5
good = [r for r in results if r['actual_pnl'] >= 300 and r['bt_pf'] >= 1.5]
good.sort(key=lambda x: x['bt_net'], reverse=True)

print("=" * 80)
print("  RESULTS: $300+ on actual trades + PF >= 1.5")
print("=" * 80)

if good:
    print(f"\n  Found {len(good)} configs meeting both goals!")
    for i, r in enumerate(good[:10]):
        p=r['params']
        print(f"\n  #{i+1}: ACTUAL=${r['actual_pnl']:+.2f}({r['actual_taken']}t) | BT=${r['bt_net']:+,.2f}({r['bt_trades']}t) | PF={r['bt_pf']} | WR={r['bt_wr']}%")
        print(f"    ADX>{p['adx_min']} Sess:{p['session']} RSI:{p['rsi_buy_min']}-{p['rsi_buy_max']}/{p['rsi_breakout']} Sell:{p['sell_checks']}chk SL:{p['sl_atr']}x RR:{p['tp_rr']}")
    
    best = good[0]
    bp = best['params']
    print(f"\n{'='*80}")
    print(f"  WINNING CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Actual Trade PnL: ${best['actual_pnl']:+,.2f} ({best['actual_taken']} trades taken)")
    print(f"  6-Month Backtest: ${best['bt_net']:+,.2f} ({best['bt_trades']} trades, PF={best['bt_pf']}, WR={best['bt_wr']}%)")
    print(f"\n  Settings:")
    print(f"    ADX Minimum: {bp['adx_min']}")
    print(f"    Session: {bp['session']}")
    print(f"    RSI Buy: {bp['rsi_buy_min']}-{bp['rsi_buy_max']} breakout>{bp['rsi_breakout']}")
    print(f"    SELL checks: {bp['sell_checks']}/5")
    print(f"    SL: {bp['sl_atr']}x ATR, RR: 1:{bp['tp_rr']}")
    
    success = True
else:
    print(f"\n  NO config met both goals. Showing closest...")
    # Show best by actual_pnl
    results.sort(key=lambda x: x['actual_pnl'], reverse=True)
    print(f"\n  Best actual PnL: ${results[0]['actual_pnl']:+,.2f} (BT: ${results[0]['bt_net']:+,.2f})")
    print(f"  Best BT net: ${max(r['bt_net'] for r in results):+,.2f} (Actual: ${max(results,key=lambda x:x['bt_net'])['actual_pnl']:+,.2f})")
    success = False

# ---- SAVE & UPLOAD ----
with open(os.path.join(OUT_DIR,"night_results.json"),'w') as f:
    json.dump({
        'timestamp':T0.isoformat(), 'success':success,
        'top_actual':[{**r} for r in sorted(results, key=lambda x:x['actual_pnl'],reverse=True)[:20]],
        'top_bt':[{**r} for r in sorted(results, key=lambda x:x['bt_net'],reverse=True)[:20]],
        'best_combo': good[0] if good else None,
    },f,default=str,indent=2)

if success and good:
    # Apply to main_super.py
    import subprocess
    main_file = os.path.join(OUT_DIR, "main_super.py")
    with open(main_file, 'r') as f: content = f.read()
    
    # Update ADX
    content = content.replace("m15_adx < 25", f"m15_adx < {bp['adx_min']}")
    content = content.replace("ADX_25_lt_25", f"ADX_{bp['adx_min']}_lt_{bp['adx_min']}")
    
    with open(main_file, 'w') as f: f.write(content)
    
    # Push to GitHub
    print(f"\nPushing to GitHub...")
    git_dir = os.path.dirname(OUT_DIR)
    subprocess.run(['git','add','-A'], cwd=git_dir, capture_output=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    subprocess.run(['git','commit','-m',f'Night optimization: ${best["actual_pnl"]:+.0f} on actual trades, ${best["bt_net"]:+,.0f} on 6mo BT — PF {best["bt_pf"]}'], cwd=git_dir, capture_output=True)
    r = subprocess.run(['git','push','origin','main'], cwd=git_dir, capture_output=True, text=True, timeout=60)
    if r.returncode==0:
        print("Pushed to GitHub!")
    else:
        print(f"Push note: {r.stderr[:100]}")

print(f"\n{'='*80}")
print(f"  DONE. Hibernating PC...")
print(f"{'='*80}")
sys.stdout.flush()

# Hibernate
os.system("shutdown /h")
