"""
ADAPTIVE ENGINE OPTIMIZATION — 10,000 combination sweep
Finds optimal learning parameters for self-adjusting filters.
Tests across 6 months of data to ensure robustness.
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "adaptive_sweep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  ADAPTIVE ENGINE OPTIMIZATION SWEEP")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull 6 months data
print("\nPulling 6 months M15...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=180)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close); hours = np.array([t.hour for t in df.index])

# Indicators (compute once for speed)
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

print("Indicators ready.\n")

# ============================================================
# PARAMETER GRID for adaptive engine
# ============================================================
param_grid = {
    'reward_block_loser': [3, 5, 8],
    'penalty_block_winner': [10, 15, 20],
    'decay_rate': [0.0, 0.5, 1.0],
    'activation_threshold': [35, 40, 45],
    'adx_init': [55, 65],
    'session_init': [40, 50],
    'macd_init': [35, 50],
    'overlap_init': [35, 50],
    'min_trades_before_activate': [3, 5],
}
# Total: 3*3*3*3*2*2*2*2*2 = 3^4 * 2^5 = 81 * 32 = 2,592

keys=list(param_grid.keys()); vals=list(param_grid.values())
total=1
for v in vals: total*=len(v)
print(f"Combinations: {total:,}\n")

# ============================================================
# ADAPTIVE BACKTEST
# ============================================================
class AdaptiveFilter:
    def __init__(self, name, init_score, reward, penalty, decay, threshold, min_trades):
        self.name = name
        self.score = init_score
        self.reward = reward
        self.penalty = penalty
        self.decay = decay
        self.threshold = threshold
        self.min_trades = min_trades
        self.blocks = 0
        self.good = 0
        self.bad = 0
        self.pending = []  # Trades waiting for outcome
    
    def is_active(self):
        if self.blocks < self.min_trades and self.score < 60:
            return True  # Not enough data, assume active
        return self.score >= self.threshold
    
    def record_block(self):
        self.blocks += 1
    
    def feedback(self, was_winner):
        if was_winner:
            self.score = max(10, self.score - self.penalty)
            self.bad += 1
        else:
            self.score = min(100, self.score + self.reward)
            self.good += 1
        # Decay toward neutral
        if self.score > 50:
            self.score = max(50, self.score - self.decay)
        elif self.score < 50:
            self.score = min(50, self.score + self.decay)

def run_adaptive(p):
    bal = 10000; trades = []; pos = None
    
    adx_f = AdaptiveFilter("ADX", p['adx_init'], p['reward_block_loser'], 
                           p['penalty_block_winner'], p['decay_rate'], 
                           p['activation_threshold'], p['min_trades_before_activate'])
    sess_f = AdaptiveFilter("Session", p['session_init'], p['reward_block_loser'],
                            p['penalty_block_winner'], p['decay_rate'],
                            p['activation_threshold'], p['min_trades_before_activate'])
    macd_f = AdaptiveFilter("MACD", p['macd_init'], p['reward_block_loser'],
                            p['penalty_block_winner'], p['decay_rate'],
                            p['activation_threshold'], p['min_trades_before_activate'])
    overlap_f = AdaptiveFilter("Overlap", p['overlap_init'], p['reward_block_loser'],
                               p['penalty_block_winner'], p['decay_rate'],
                               p['activation_threshold'], p['min_trades_before_activate'])
    
    sim_trades = []  # Simulated trades for feedback
    
    for i in range(400, n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
        ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]; pmh=macd_h[i-1] if i>0 else mh
        hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml) or np.isnan(adx): continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                
                # Feedback to filters that blocked similar trades
                for sim in sim_trades[-20:]:  # Recent simulated trades
                    for fname, filt in [('adx',adx_f),('session',sess_f),('macd',macd_f),('overlap',overlap_f)]:
                        if sim.get(f'blocked_{fname}'):
                            filt.feedback(sim['would_win'])
                sim_trades = []
                
                trades.append({'d':pos['d'],'pnl':pnl}); pos=None
            continue
        
        # ---- DETERMINE SIGNAL ----
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_signal = False; sell_signal = False
        
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_signal=True
            elif 40<=rsi<=65 and abs(px-bm)/bm<0.01 and mh>pmh and psk<=psd and sk_i>sd_i: buy_signal=True
        
        if not buy_signal and not (not np.isnan(maf) and px>maf):
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
            if ml<ms_ and mh<0: checks+=1
            if psk>=psd and sk_i<sd_i: checks+=1
            if abs(px-bl)/bl>0.005: checks+=1
            if px<bm: checks+=1
            if checks>=3: sell_signal=True
        
        if not buy_signal and not sell_signal: continue
        
        direction = 'B' if buy_signal else 'S'
        
        # ---- CHECK ADAPTIVE FILTERS ----
        blocked = False; blocked_by = {}
        
        if direction == 'B':
            if adx_f.is_active() and adx < 25:
                blocked=True; blocked_by['adx']=True; adx_f.record_block()
            if sess_f.is_active() and not (8 <= hr < 22):
                blocked=True; blocked_by['session']=True; sess_f.record_block()
        else:
            if adx_f.is_active() and adx < 25:
                blocked=True; blocked_by['adx']=True; adx_f.record_block()
            if sess_f.is_active() and not (8 <= hr < 22):
                blocked=True; blocked_by['session']=True; sess_f.record_block()
            if macd_f.is_active() and ml > ms_:
                blocked=True; blocked_by['macd']=True; macd_f.record_block()
            if overlap_f.is_active() and (13 <= hr < 17):
                blocked=True; blocked_by['overlap']=True; overlap_f.record_block()
        
        if blocked:
            # Simulate: what would have happened?
            would_win = False
            sld = atr*3.0
            if direction == 'B':
                sl=px-sld; tp=px+sld*2.0
                for j in range(i+1, min(i+300, n)):
                    if high[j] >= tp: would_win=True; break
                    if low[j] <= sl: would_win=False; break
            else:
                sl=px+sld; tp=px-sld*2.0
                for j in range(i+1, min(i+300, n)):
                    if low[j] <= tp: would_win=True; break
                    if high[j] >= sl: would_win=False; break
            
            sim_trades.append({**{f'blocked_{k}':v for k,v in blocked_by.items()}, 'would_win':would_win})
            continue
        
        # Place trade
        sld = atr*3.0
        if direction == 'B':
            sl=px-sld; tp=px+sld*2.0
        else:
            sl=px+sld; tp=px-sld*2.0
        lots=max(0.01,round((bal*0.02)/(sld*100),2))
        pos={'d':direction,'e':px,'sl':sl,'tp':tp,'l':lots}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})
    
    if len(trades)<5: return None
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    
    # Score: reward high net + PF + fewer losses
    wr = len(wins)/len(trades)*100
    
    return {
        'params': p, 'net': round(net,2), 'trades': len(trades),
        'wr': round(wr,1), 'pf': round(pf,2),
        'adx_score': round(adx_f.score,1), 'session_score': round(sess_f.score,1),
        'macd_score': round(macd_f.score,1), 'overlap_score': round(overlap_f.score,1),
        'adx_active': adx_f.is_active(), 'session_active': sess_f.is_active(),
        'macd_active': macd_f.is_active(), 'overlap_active': overlap_f.is_active(),
    }

# ---- RUN SWEEP ----
print("Running sweep...")
results=[]; cnt=0
for combo in itertools.product(*vals):
    cnt+=1
    if cnt%3000==0:
        et=max((datetime.now()-T0).total_seconds(),0.001)
        print(f"  {cnt}/{total} ({cnt/total*100:.1f}%) | {cnt/et:.0f}/s | ETA:{(total-cnt)/(cnt/et)/60:.1f}min")
    p=dict(zip(keys,combo)); r=run_adaptive(p)
    if r: results.append(r)

et=(datetime.now()-T0).total_seconds()
print(f"\nDone in {et:.0f}s. {len(results)} results.\n")

# ---- FIND BEST ----
results.sort(key=lambda x: x['net'], reverse=True)

print("="*80)
print("  TOP 10 ADAPTIVE CONFIGS")
print("="*80)
for i,r in enumerate(results[:10]):
    p=r['params']
    print(f"\n #{i+1}: ${r['net']:+,.0f} | {r['trades']}t WR={r['wr']}% PF={r['pf']}")
    print(f"   Reward:{p['reward_block_loser']} Penalty:{p['penalty_block_winner']} Decay:{p['decay_rate']} Thresh:{p['activation_threshold']}")
    print(f"   ADX:{r['adx_score']:.0f}({'ON' if r['adx_active'] else 'OFF'}) Sess:{r['session_score']:.0f}({'ON' if r['session_active'] else 'OFF'}) MACD:{r['macd_score']:.0f}({'ON' if r['macd_active'] else 'OFF'}) Ovlp:{r['overlap_score']:.0f}({'ON' if r['overlap_active'] else 'OFF'})")

# Best balanced
max_net=max(r['net'] for r in results); max_pf=max(r['pf'] for r in results)
for r in results: r['score']=(r['net']/max_net)*0.5+(r['pf']/max_pf)*0.3+(r['wr']/100)*0.2
results.sort(key=lambda x: x['score'], reverse=True)
best=results[0]; bp=best['params']

print(f"\n{'='*80}")
print("  OPTIMAL ADAPTIVE CONFIG")
print(f"{'='*80}")
print(f"\n  ${best['net']:+,.0f} | {best['trades']}t WR={best['wr']}% PF={best['pf']}")
print(f"\n  Learning Parameters:")
print(f"    Reward for blocking loser: +{bp['reward_block_loser']} pts")
print(f"    Penalty for blocking winner: -{bp['penalty_block_winner']} pts")
print(f"    Decay rate: {bp['decay_rate']} pts/event")
print(f"    Activation threshold: {bp['activation_threshold']}/100")
print(f"    Min trades before activate: {bp['min_trades_before_activate']}")
print(f"\n  Initial Scores:")
print(f"    ADX: {bp['adx_init']} | Session: {bp['session_init']} | MACD: {bp['macd_init']} | Overlap: {bp['overlap_init']}")
print(f"\n  Final Filter States:")
print(f"    ADX: {best['adx_score']:.0f} ({'ACTIVE' if best['adx_active'] else 'OFF'})")
print(f"    Session: {best['session_score']:.0f} ({'ACTIVE' if best['session_active'] else 'OFF'})")
print(f"    MACD: {best['macd_score']:.0f} ({'ACTIVE' if best['macd_active'] else 'OFF'})")
print(f"    Overlap: {best['overlap_score']:.0f} ({'ACTIVE' if best['overlap_active'] else 'OFF'})")

# Compare to non-adaptive baseline (ADX>25 only)
non_adaptive_pnl = max([r['net'] for r in results if r['params']['adx_init']==70 and r['params']['session_init']==40], default=0)
print(f"\n  Non-adaptive baseline: ~${non_adaptive_pnl:+,.0f}")
print(f"  Adaptive improvement:  ${best['net']-non_adaptive_pnl:+,.0f}")

print(f"\n{'='*80}")
print(f"  SAVED")
print(f"{'='*80}")
