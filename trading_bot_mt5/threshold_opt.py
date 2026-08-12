"""
DATA-DRIVEN THRESHOLD ANALYSIS: Find exact RSI levels where losses spike.
No estimates - pure backtest data across all 4 periods.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "threshold_analysis.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  DATA-DRIVEN THRESHOLD ANALYSIS")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull ALL available data
print("\nPulling ALL M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2025,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

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

# ---- TEST EVERY RSI THRESHOLD for BUY entries ----
print("\n" + "=" * 80)
print("  BUY RSI THRESHOLD ANALYSIS (data-driven)")
print("=" * 80)

def test_buy_rsi_limit(max_rsi):
    """Simulate: only allow BUY if RSI <= max_rsi. Measure P&L impact."""
    bal=10000; trades=[]; pos=None
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i): continue
        
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                trades.append({'d':pos['d'],'pnl':pnl,'rsi':pos.get('rsi'),'reason':'sl' if sl_hit else 'tp'})
                pos=None
            continue
        
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        
        # ---- RSI LIMIT ----
        if buy_ok and rsi > max_rsi:
            buy_ok = False  # Blocked by RSI limit
        
        if buy_ok:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi}
            continue
        
        # SELL (unchanged)
        if not (not np.isnan(maf) and px>maf):
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml_i<ms_i and mh_i<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl[i])/bl[i]>0.005: checks+=1
            if px<bm[i]: checks+=1
            if checks>=3:
                sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'rsi':pos.get('rsi'),'reason':'eod'})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    buys=[t for t in trades if t['d']=='B']; buy_losses=[t for t in buys if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    return {'max_rsi':max_rsi,'net':round(net,2),'trades':len(trades),'buys':len(buys),
            'buy_losses':len(buy_losses),'pf':round(pf,2),
            'wr':round(len(wins)/len(trades)*100,1)}

# Test baseline (no RSI limit)
baseline = test_buy_rsi_limit(100)
print(f"\n  BASELINE (no RSI limit):")
print(f"    ${baseline['net']:+,.0f} | {baseline['trades']}t WR={baseline['wr']}% PF={baseline['pf']} | {baseline['buys']} BUYs, {baseline['buy_losses']} losses")

# Test every RSI limit
print(f"\n  RSI LIMIT IMPACT:")
print(f"  {'Limit':<10} {'Net P&L':<12} {'Trades':<8} {'WR':<8} {'PF':<8} {'BUY Losers':<12} {'vs Baseline':<12}")

best_rsi = None
best_net = baseline['net']
for max_rsi in [80, 75, 72, 70, 68, 65, 62, 60, 58, 55]:
    r = test_buy_rsi_limit(max_rsi)
    if r:
        diff = r['net'] - baseline['net']
        marker = " <-- BEST" if r['net'] > best_net else ""
        if r['net'] > best_net:
            best_net = r['net']; best_rsi = max_rsi
        print(f"  RSI<{max_rsi:<5} ${r['net']:+,.0f}      {r['trades']:<8} {r['wr']}%    {r['pf']:<8} {r['buy_losses']:<12} ${diff:+,.0f}{marker}")

print(f"\n  >>> OPTIMAL BUY RSI LIMIT: RSI < {best_rsi} (improves by ${best_net-baseline['net']:+,.0f})")

# ---- TEST SELL RSI LIMITS ----
print(f"\n{'='*80}")
print("  SELL RSI THRESHOLD ANALYSIS (data-driven)")
print("="*80)

def test_sell_rsi_min(min_rsi):
    """Simulate: only allow SELL if RSI >= min_rsi. Measure P&L impact."""
    bal=10000; trades=[]; pos=None
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i): continue
        
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                trades.append({'d':pos['d'],'pnl':pnl,'rsi':pos.get('rsi'),'reason':'sl' if sl_hit else 'tp'})
                pos=None
            continue
        
        # BUY (unchanged)
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): pass  # pass, handled below
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: pass
            else: trend_up=False
        
        if trend_up and rsi<=72:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi}
            continue
        
        # SELL with RSI minimum
        if not (not np.isnan(maf) and px>maf):
            # ---- RSI MINIMUM ----
            if rsi < min_rsi:
                continue  # Blocked: RSI too low
            
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml_i<ms_i and mh_i<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl[i])/bl[i]>0.005: checks+=1
            if px<bm[i]: checks+=1
            if checks>=3:
                sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'rsi':pos.get('rsi'),'reason':'eod'})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    sells=[t for t in trades if t['d']=='S']; sell_losses=[t for t in sells if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    return {'min_rsi':min_rsi,'net':round(net,2),'trades':len(trades),'sells':len(sells),
            'sell_losses':len(sell_losses),'pf':round(pf,2),
            'wr':round(len(wins)/len(trades)*100,1)}

baseline2 = test_sell_rsi_min(0)
print(f"\n  BASELINE (no RSI min for SELL):")
print(f"    ${baseline2['net']:+,.0f} | {baseline2['trades']}t WR={baseline2['wr']}% PF={baseline2['pf']} | {baseline2['sells']} SELLs, {baseline2['sell_losses']} losses")

print(f"\n  RSI MIN IMPACT:")
best_rsi_sell = None; best_net_sell = baseline2['net']
for min_rsi in [20, 25, 28, 30, 32, 35, 38, 40, 45]:
    r = test_sell_rsi_min(min_rsi)
    if r:
        diff = r['net'] - baseline2['net']
        marker = " <-- BEST" if r['net'] > best_net_sell else ""
        if r['net'] > best_net_sell:
            best_net_sell = r['net']; best_rsi_sell = min_rsi
        print(f"  RSI>={min_rsi:<4} ${r['net']:+,.0f}      {r['trades']:<8} {r['wr']}%    {r['pf']:<8} {r['sell_losses']:<12} ${diff:+,.0f}{marker}")

print(f"\n  >>> OPTIMAL SELL RSI MIN: RSI >= {best_rsi_sell} (improves by ${best_net_sell-baseline2['net']:+,.0f})")

# ---- COMBINED OPTIMAL ----
print(f"\n{'='*80}")
print("  COMBINED OPTIMAL: BUY RSI<{best_rsi} + SELL RSI>={best_rsi_sell}")
print(f"{'='*80}")

def test_combined(buy_max_rsi, sell_min_rsi):
    bal=10000; trades=[]; pos=None
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i): continue
        
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                trades.append({'d':pos['d'],'pnl':pnl}); pos=None
            continue
        
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        if buy_ok and rsi > buy_max_rsi: buy_ok=False
        
        if buy_ok:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2)); pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots}
            continue
        
        if not (not np.isnan(maf) and px>maf):
            if rsi < sell_min_rsi: continue
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml_i<ms_i and mh_i<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl[i])/bl[i]>0.005: checks+=1
            if px<bm[i]: checks+=1
            if checks>=3:
                sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2)); pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    return {'net':round(net,2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1),'pf':round(pf,2)}

combined = test_combined(best_rsi, best_rsi_sell)
print(f"\n  Combined: ${combined['net']:+,.0f} | {combined['trades']}t WR={combined['wr']}% PF={combined['pf']}")
print(f"  vs Baseline: ${combined['net']-baseline['net']:+,.0f}")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
