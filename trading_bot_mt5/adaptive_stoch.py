"""
ADAPTIVE STOCHASTIC: Self-learning Stochastic filters.
Tests: Stoch K>80 BUY block, bullish cross SELL block, K<20 SELL block.
Adaptive: each rule learns from outcomes, enables/disables itself.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "adaptive_stoch.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  ADAPTIVE STOCHASTIC ANALYSIS")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull data
print("\nPulling 18 months M15...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2025,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close); hours=np.array([t.hour for t in df.index])

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
ll_=pd.Series(low).rolling(14).min().values; hh_=pd.Series(high).rolling(14).max().values
rk_=np.full(n,np.nan); den_=hh_-ll_; msk_=den_>0; rk_[msk_]=(close[msk_]-ll_[msk_])/den_[msk_]*100
sk=pd.Series(rk_).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
ef=ema(close,12); es=ema(close,26); ml=ef-es; ms=ema(ml,9); mh=ml-ms
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
ae=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

print(f"Got {n} candles\n")

# Adaptive filter class
class AdaptiveFilter:
    def __init__(self, name, init=50):
        self.name=name; self.score=init; self.blocks=0; self.good=0; self.bad=0
    def active(self):
        if self.blocks < 5: return True
        return self.score >= 35
    def record(self): self.blocks+=1
    def feedback(self, winner):
        if winner: self.score=max(10,self.score-10); self.bad+=1
        else: self.score=min(100,self.score+8); self.good+=1

# ---- BASELINE (no Stochastic rules) ----
def backtest(stoch_k80_buy, stoch_bull_cross_sell, stoch_k20_sell):
    bal=10000; trades=[]; pos=None; sim=[]
    
    # Adaptive filters
    buy_k80 = AdaptiveFilter("BUY K>80", 50)
    sell_bull = AdaptiveFilter("SELL bull cross", 60)
    sell_k20 = AdaptiveFilter("SELL K<20", 50)
    
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i): continue
        
        # Position mgmt
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                for s in sim[-20:]:
                    for fname,filt in [('buy_k80',buy_k80),('sell_bull',sell_bull),('sell_k20',sell_k20)]:
                        if s.get(f'blocked_{fname}'): filt.feedback(s['winner'])
                sim=[]
                trades.append({'d':pos['d'],'pnl':pnl}); pos=None
            continue
        
        # BUY signal
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if 68<=rsi<=80 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        
        if buy_ok:
            blocked=False; blocked_by={}
            
            # Stoch K>80 block for BUY
            if stoch_k80_buy and buy_k80.active() and sk_i > 80:
                blocked=True; blocked_by['buy_k80']=True; buy_k80.record()
            
            if blocked:
                would_win=False; sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
                for j in range(i+1,min(i+300,n)):
                    if high[j]>=tp: would_win=True; break
                    if low[j]<=sl: break
                sim.append({**{f'blocked_{k}':v for k,v in blocked_by.items()},'winner':would_win})
            else:
                sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots}; continue
        
        # SELL signal
        if not (not np.isnan(maf) and px>maf):
            if ml_i > ms_i: continue
            
            blocked=False; blocked_by={}
            
            # Stoch bullish cross block for SELL
            is_bull_cross = psk <= psd and sk_i > sd_i
            if stoch_bull_cross_sell and sell_bull.active() and is_bull_cross:
                blocked=True; blocked_by['sell_bull']=True; sell_bull.record()
            
            # Stoch K<20 block for SELL
            if stoch_k20_sell and sell_k20.active() and sk_i < 20:
                blocked=True; blocked_by['sell_k20']=True; sell_k20.record()
            
            checks=0
            if 45<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml_i<ms_i and mh_i<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl[i])/bl[i]>0.005: checks+=1
            if px<bm[i]: checks+=1
            
            if checks>=3:
                if blocked:
                    would_win=False; sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                    for j in range(i+1,min(i+300,n)):
                        if low[j]<=tp: would_win=True; break
                        if high[j]>=sl: break
                    sim.append({**{f'blocked_{k}':v for k,v in blocked_by.items()},'winner':would_win})
                else:
                    sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                    lots=max(0.01,round((bal*0.02)/(sld*100),2))
                    pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    return {
        'net':round(net,2),'trades':len(trades),'wr':round(len(wins)/len(trades)*100,1),'pf':round(pf,2),
        'buy_k80_score':buy_k80.score,'buy_k80_blocks':buy_k80.blocks,'buy_k80_good':buy_k80.good,'buy_k80_bad':buy_k80.bad,
        'sell_bull_score':sell_bull.score,'sell_bull_blocks':sell_bull.blocks,'sell_bull_good':sell_bull.good,'sell_bull_bad':sell_bull.bad,
        'sell_k20_score':sell_k20.score,'sell_k20_blocks':sell_k20.blocks,'sell_k20_good':sell_k20.good,'sell_k20_bad':sell_k20.bad,
    }

# ---- TEST ALL COMBINATIONS ----
configs = [
    ("BASELINE (no Stoch rules)", False, False, False),
    ("BUY K>80 block only", True, False, False),
    ("SELL bull cross block only", False, True, False),
    ("SELL K<20 block only", False, False, True),
    ("BUY K>80 + SELL bull cross", True, True, False),
    ("BUY K>80 + SELL K<20", True, False, True),
    ("SELL bull cross + SELL K<20", False, True, True),
    ("ALL THREE Stoch rules", True, True, True),
]

print("=" * 80)
print("  STOCHASTIC RULE IMPACT (Adaptive)")
print("=" * 80)
print(f"  {'Config':<35} {'Net':>10} {'Trades':>8} {'WR':>7} {'PF':>7}")

baseline_result = None
best_result = None

for name, k80, bull, k20 in configs:
    r = backtest(k80, bull, k20)
    if r:
        if baseline_result is None: baseline_result = r
        diff = r['net'] - baseline_result['net'] if baseline_result else 0
        marker = " <-- BEST" if best_result is None or r['net'] > best_result['net'] else ""
        if best_result is None or r['net'] > best_result['net']:
            best_result = r
        print(f"  {name:<35} ${r['net']:+,.0f}   {r['trades']:<8} {r['wr']}%  {r['pf']:<7} {diff:+,.0f}{marker}")

# ---- DETAIL ON BEST ----
if best_result:
    print(f"\n{'='*80}")
    print(f"  BEST CONFIG FILTER STATES")
    print(f"{'='*80}")
    for fname, score, blocks, good, bad in [
        ("BUY K>80 block", best_result['buy_k80_score'], best_result['buy_k80_blocks'], best_result['buy_k80_good'], best_result['buy_k80_bad']),
        ("SELL bull cross", best_result['sell_bull_score'], best_result['sell_bull_blocks'], best_result['sell_bull_good'], best_result['sell_bull_bad']),
        ("SELL K<20 block", best_result['sell_k20_score'], best_result['sell_k20_blocks'], best_result['sell_k20_good'], best_result['sell_k20_bad']),
    ]:
        status = "ACTIVE" if score >= 35 else "DISABLED"
        acc = f"{good}/{good+bad} correct" if good+bad > 0 else "no data"
        print(f"  {fname}: score={score:.0f} ({status}) | {blocks} blocks, {acc}")

print(f"\n{'='*80}")
print(f"  COMPLETE")
print(f"{'='*80}")
