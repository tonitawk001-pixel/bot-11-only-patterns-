"""
ADAPTIVE FILTER ENGINE: Self-adjusting filters based on recent trade outcomes.
Each filter has a confidence score. When a filter blocks a trade that would have won,
its score drops. When it blocks a loser, its score rises. Filters with low scores
are temporarily disabled.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "adaptive_test.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  ADAPTIVE FILTER ENGINE — Self-learning from trade outcomes")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull BOTH 2025 and 2026 data
print("\nPulling 2025 + 2026 M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)

# 2025
r2025 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, 
    datetime(2025,1,1,tzinfo=timezone.utc), datetime(2025,7,1,tzinfo=timezone.utc))
# 2026
r2026 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2026,2,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()

def process_rates(rates):
    df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df

df2025 = process_rates(r2025) if r2025 is not None and len(r2025)>500 else None
df2026 = process_rates(r2026) if r2026 is not None and len(r2026)>500 else None

# Use whichever has more data
df = df2026 if df2026 is not None and len(df2026) > len(df2025) else df2025
print(f"Using: {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close); hours = np.array([t.hour for t in df.index])

# Indicators (same as always)
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
# ADAPTIVE FILTER ENGINE
# ============================================================
class AdaptiveFilter:
    def __init__(self, name, score=50):
        self.name = name
        self.score = score  # 0-100, starts neutral
        self.blocks = 0
        self.good_blocks = 0  # Blocked a loser
        self.bad_blocks = 0   # Blocked a winner
        self.history = []
    
    def should_block(self):
        """Filter is active if score >= 40 (adaptive threshold)"""
        return self.score >= 40
    
    def record_outcome(self, blocked, would_have_won):
        """Update score based on trade outcome"""
        if not blocked: return
        
        self.blocks += 1
        if would_have_won:
            # We blocked a winner - penalty
            self.bad_blocks += 1
            self.score = max(10, self.score - 15)
        else:
            # We blocked a loser - reward
            self.good_blocks += 1
            self.score = min(100, self.score + 5)
        
        # Decay: slowly return to neutral over time
        if self.score > 50:
            self.score -= 0.5
        elif self.score < 50:
            self.score += 0.5
        
        self.history.append(self.score)

# Create adaptive filters
adx_filter = AdaptiveFilter("ADX>25", score=70)  # Start confident (always helps)
session_filter = AdaptiveFilter("Active Hours", score=50)  # Start neutral
macd_gate = AdaptiveFilter("MACD Gate SELL", score=40)  # Start low (mixed evidence)
overlap_filter = AdaptiveFilter("Overlap SELL Block", score=40)  # Start low

# ============================================================
# BACKTEST WITH ADAPTIVE FILTERS
# ============================================================
print("Running adaptive backtest...")
bal = 10000; trades = []; pos = None
filter_log = []
LOOKBACK = 20  # How many bars to look back to assess blocked trades

for i in range(400, n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
    sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
    pmh=macd_h[i-1] if i>0 else mh; hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
    
    if np.isnan(ml) or np.isnan(adx): continue
    
    # Every 500 bars, log filter states
    if i % 2000 == 0:
        filter_log.append({
            'bar': i,
            'adx_score': round(adx_filter.score, 1),
            'session_score': round(session_filter.score, 1),
            'macd_score': round(macd_gate.score, 1),
            'overlap_score': round(overlap_filter.score, 1),
        })
    
    # Manage position
    if pos is not None:
        sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
        tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
        if sl_hit or tp_hit:
            ep=pos['sl'] if sl_hit else pos['tp']
            pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
            bal+=pnl
            won = pnl > 0
            
            # Update filter scores based on which filters blocked this trade
            for fname, blocked_flag in pos.get('blocked_by', {}).items():
                if blocked_flag:
                    filt = {'adx': adx_filter, 'session': session_filter, 
                           'macd': macd_gate, 'overlap': overlap_filter}.get(fname)
                    if filt:
                        filt.record_outcome(True, won)  # We didn't actually block, but we WOULD have
            
            trades.append({'d':pos['d'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp'}); pos=None
        continue
    
    # ---- ADAPTIVE PRE-FILTERS ----
    blocked_by = {}
    
    if adx_filter.should_block() and adx < 25:
        blocked_by['adx'] = True
    if session_filter.should_block() and not (8 <= hr < 22):
        blocked_by['session'] = True
    # Note: MACD gate and overlap block are checked per-direction below
    
    # ---- BUY ----
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    buy_ok = False
    if trend_up:
        if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok = True
        elif 40<=rsi<=65 and abs(px-bm)/bm<0.01 and mh>pmh and psk<=psd and sk_i>sd_i: buy_ok = True
    
    if buy_ok and not blocked_by:
        sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
        lots=max(0.01,round((bal*0.02)/(sld*100),2))
        pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots,'blocked_by':blocked_by}
        continue
    
    # ---- SELL ----
    if buy_ok and blocked_by:
        # Simulate: what would have happened if we took this trade?
        # (check forward bars for outcome)
        for j in range(i+1, min(i+200, n)):
            fpx = close[j]; fhigh = high[j]; flow = low[j]
            sl = px - atr*3.0; tp = px + atr*6.0
            if fhigh >= tp:
                # Would have won
                for fname in blocked_by:
                    filt = {'adx': adx_filter, 'session': session_filter,
                           'macd': macd_gate, 'overlap': overlap_filter}.get(fname)
                    if filt: filt.record_outcome(True, True)
                break
            if flow <= sl:
                for fname in blocked_by:
                    filt = {'adx': adx_filter, 'session': session_filter,
                           'macd': macd_gate, 'overlap': overlap_filter}.get(fname)
                    if filt: filt.record_outcome(True, False)
                break
    
    # SELL signal
    if not blocked_by and not (not np.isnan(maf) and px>maf):
        # Adaptive MACD gate
        if macd_gate.should_block() and ml > ms_: continue
        # Adaptive overlap block
        if overlap_filter.should_block() and (13 <= hr < 17): continue
        
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml<ms_ and mh<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl)/bl>0.005: checks+=1
        if px<bm: checks+=1
        if checks>=3:
            sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots,'blocked_by':{}}

if pos is not None:
    ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
    bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})

wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*80}")
print("  ADAPTIVE FILTER RESULTS")
print(f"{'='*80}")
print(f"\n  Net PnL: ${sum(t['pnl'] for t in trades):+,.0f}")
print(f"  Trades: {len(trades)} | WR: {len(wins)/len(trades)*100:.0f}%")
print(f"  Wins: {len(wins)} | Losses: {len(loses)}")

print(f"\n  FILTER LEARNING:")
for filt in [adx_filter, session_filter, macd_gate, overlap_filter]:
    print(f"\n  {filt.name}:")
    print(f"    Final score: {filt.score:.0f}/100 ({'ACTIVE' if filt.should_block() else 'DISABLED'})")
    print(f"    Total blocks: {filt.blocks}")
    print(f"    Good blocks (blocked losers): {filt.good_blocks}")
    print(f"    Bad blocks (blocked winners): {filt.bad_blocks}")
    if filt.bad_blocks > 0 and filt.good_blocks > 0:
        ratio = filt.good_blocks / (filt.good_blocks + filt.bad_blocks) * 100
        print(f"    Accuracy: {ratio:.0f}%")

# Also test non-adaptive for comparison
print(f"\n{'='*80}")
print("  COMPARISON: Adaptive vs Fixed")
print(f"{'='*80}")

# Non-adaptive baseline (all fixed ON)
bal2=10000; trades2=[]; pos2=None
for i in range(400,n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
    sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]; pmh=macd_h[i-1] if i>0 else mh
    hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
    
    if np.isnan(ml) or np.isnan(adx): continue
    if adx < 25: continue  # Only ADX fixed (proven universal)
    
    if pos2 is not None:
        sl_hit=(pos2['d']=='B' and px<=pos2['sl']) or (pos2['d']=='S' and px>=pos2['sl'])
        tp_hit=(pos2['d']=='B' and px>=pos2['tp']) or (pos2['d']=='S' and px<=pos2['tp'])
        if sl_hit or tp_hit:
            ep=pos2['sl'] if sl_hit else pos2['tp']
            pnl=(ep-pos2['e'])*100*pos2['l'] if pos2['d']=='B' else (pos2['e']-ep)*100*pos2['l']
            bal2+=pnl; trades2.append({'d':pos2['d'],'pnl':pnl}); pos2=None
        continue
    
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    if trend_up:
        buy_ok=False
        if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok=True
        elif 40<=rsi<=65 and abs(px-bm)/bm<0.01 and mh>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        if buy_ok:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            lots=max(0.01,round((bal2*0.02)/(sld*100),2))
            pos2={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots}; continue
    
    if not (not np.isnan(maf) and px>maf):
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml<ms_ and mh<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl)/bl>0.005: checks+=1
        if px<bm: checks+=1
        if checks>=3:
            sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
            lots=max(0.01,round((bal2*0.02)/(sld*100),2))
            pos2={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots}

if pos2 is not None:
    ep=close[-1]; pnl=(ep-pos2['e'])*100*pos2['l'] if pos2['d']=='B' else (pos2['e']-ep)*100*pos2['l']
    bal2+=pnl; trades2.append({'d':pos2['d'],'pnl':pnl,'reason':'eod'})

adapt_pnl = sum(t['pnl'] for t in trades)
fixed_pnl = sum(t['pnl'] for t in trades2)
w2 = [t for t in trades2 if t['pnl']>0]

print(f"\n  Adaptive (all filters self-learning): ${adapt_pnl:+,.0f} | {len(trades)}t")
print(f"  Fixed ADX>25 only:                   ${fixed_pnl:+,.0f} | {len(trades2)}t WR={len(w2)/len(trades2)*100:.0f}%")
print(f"  Difference:                           ${adapt_pnl-fixed_pnl:+,.0f}")

print(f"\n{'='*80}")
print(f"  FILTER SCORE HISTORY (how they learned):")
print(f"{'='*80}")
for log in filter_log:
    print(f"  Bar {log['bar']}: ADX={log['adx_score']} Session={log['session_score']} MACD={log['macd_score']} Overlap={log['overlap_score']}")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
