"""
DEEP STOCHASTIC ANALYSIS: Every trade from 18 months, analyzed through Stochastic lens.
Discovers: optimal K/D levels, false signal patterns, reversal prediction accuracy.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "stochastic_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  DEEP STOCHASTIC ANALYSIS")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull all data
print("\nPulling 18 months M15...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2025,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles")

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
ef=ema(close,12); es=ema(close,26); ml=ef-es; ms=ema(ml,9); mh=ml-ms

# ADX
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
ae=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

# Multiple Stochastic variants
stoch_variants = {}
for kp, dp, slw, name in [(14,3,3,'14-3-3'), (8,3,3,'8-3-3'), (5,3,3,'5-3-3'), (21,5,3,'21-5-3')]:
    ll=pd.Series(low).rolling(kp).min().values; hh=pd.Series(high).rolling(kp).max().values
    rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
    sk_=pd.Series(rk).rolling(slw).mean().values
    sd_=pd.Series(sk_).rolling(dp).mean().values
    stoch_variants[name] = (sk_, sd_)

# Default: 14-3-3
sk, sd = stoch_variants['14-3-3']

# ---- BACKTEST: Log every trade with full Stochastic data ----
print("Running backtest with Stochastic logging...")

all_trades = []
bal=10000; pos=None

for i in range(400,n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]
    sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
    hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
    
    # Get all Stochastic variants at this bar
    stoch_snapshot = {}
    for name, (sk_arr, sd_arr) in stoch_variants.items():
        stoch_snapshot[f'K_{name}'] = sk_arr[i] if not np.isnan(sk_arr[i]) else None
        stoch_snapshot[f'D_{name}'] = sd_arr[i] if not np.isnan(sd_arr[i]) else None
        stoch_snapshot[f'Kprev_{name}'] = sk_arr[i-1] if i>0 and not np.isnan(sk_arr[i-1]) else None
        stoch_snapshot[f'Dprev_{name}'] = sd_arr[i-1] if i>0 and not np.isnan(sd_arr[i-1]) else None
    
    if np.isnan(ml_i): continue
    
    if pos is not None:
        sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
        tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
        if sl_hit or tp_hit:
            ep=pos['sl'] if sl_hit else pos['tp']
            pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
            bal+=pnl
            all_trades.append({
                'd':pos['d'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                'rsi':pos['rsi'],'adx':pos['adx'],'macd_bull':pos['macd_bull'],
                'bb':pos['bb'],'hour':pos['hour'],'atr':pos['atr'],
                **pos.get('stoch',{})
            }); pos=None
        continue
    
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    buy_ok=False; buy_type=''
    if trend_up:
        if 68<=rsi<=80 and not (psk>psd and sk_i<sd_i):
            buy_ok=True; buy_type='breakout'
        elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i:
            buy_ok=True; buy_type='pullback'
    
    vn=vol[i]; vma_v=sma(vol,20)
    bb_label='ABOVE' if px>bm[i] else 'BELOW'
    
    if buy_ok:
        sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
        lots=max(0.01,round((bal*0.02)/(sld*100),2))
        pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi,'adx':adx,
             'macd_bull':ml_i>ms_i,'bb':bb_label,'hour':hr,'atr':atr,
             'stoch':stoch_snapshot}
        continue
    
    if not (not np.isnan(maf) and px>maf):
        if ml_i > ms_i: continue  # MACD gate
        checks=0
        if 45<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml_i<ms_i and mh_i<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl[i])/bl[i]>0.005: checks+=1
        if px<bm[i]: checks+=1
        if checks>=3:
            sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
            lots=max(0.01,round((bal*0.02)/(sld*100),2))
            pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi,'adx':adx,
                 'macd_bull':ml_i>ms_i,'bb':bb_label,'hour':hr,'atr':atr,
                 'stoch':stoch_snapshot}

if pos is not None:
    ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
    bal+=pnl; all_trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})

wins=[t for t in all_trades if t['pnl']>0]; loses=[t for t in all_trades if t['pnl']<=0]
print(f"Backtest: {len(all_trades)} trades, ${sum(t['pnl'] for t in all_trades):+,.0f}")
print(f"Wins: {len(wins)}, Losses: {len(loses)}")

# ============================================================
# ANALYSIS 1: Stochastic K levels at entry — win rate by bucket
# ============================================================
print(f"\n{'='*80}")
print("  ANALYSIS 1: STOCHASTIC K AT ENTRY — Win Rate by Level")
print(f"{'='*80}")

for direction, label in [('B','BUY'),('S','SELL')]:
    d_trades = [t for t in all_trades if t['d']==direction and t.get('K_14-3-3') is not None]
    if not d_trades: continue
    
    print(f"\n  {label}:")
    for lo, hi, zone in [(0,20,'Oversold <20'),(20,35,'Low 20-35'),(35,50,'Mid-low 35-50'),
                          (50,65,'Mid-high 50-65'),(65,80,'High 65-80'),(80,100,'Overbought >80')]:
        zt = [t for t in d_trades if lo <= t['K_14-3-3'] < hi]
        if len(zt) < 2: continue
        zw = [t for t in zt if t['pnl']>0]
        wr = len(zw)/len(zt)*100; net = sum(t['pnl'] for t in zt)
        print(f"    Stoch K {zone}: {len(zt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(zt):+,.0f}")

# ANALYSIS 2: K-D relationship
print(f"\n{'='*80}")
print("  ANALYSIS 2: K vs D RELATIONSHIP")
print(f"{'='*80}")

for direction, label in [('B','BUY'),('S','SELL')]:
    d_trades = [t for t in all_trades if t['d']==direction and t.get('K_14-3-3') is not None and t.get('D_14-3-3') is not None]
    if not d_trades: continue
    
    print(f"\n  {label}:")
    
    # K above D (bullish) vs K below D (bearish)
    k_above = [t for t in d_trades if t['K_14-3-3'] > t['D_14-3-3']]
    k_below = [t for t in d_trades if t['K_14-3-3'] <= t['D_14-3-3']]
    
    for subset, sublabel in [(k_above,'K > D'), (k_below,'K <= D')]:
        if len(subset)<2: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100; net = sum(t['pnl'] for t in subset)
        print(f"    {sublabel}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(subset):+,.0f}")

# ANALYSIS 3: Cross detection accuracy
print(f"\n{'='*80}")
print("  ANALYSIS 3: CROSS DETECTION — Did the cross predict correctly?")
print(f"{'='*80}")

for direction, label in [('B','BUY'),('S','SELL')]:
    d_trades = [t for t in all_trades if t['d']==direction and t.get('K_14-3-3') is not None 
               and t.get('D_14-3-3') is not None and t.get('Kprev_14-3-3') is not None
               and t.get('Dprev_14-3-3') is not None]
    if not d_trades: continue
    
    print(f"\n  {label}:")
    
    # Bullish cross: K was below D, now above
    bull_cross = [t for t in d_trades if t['Kprev_14-3-3'] <= t['Dprev_14-3-3'] and t['K_14-3-3'] > t['D_14-3-3']]
    # Bearish cross: K was above D, now below
    bear_cross = [t for t in d_trades if t['Kprev_14-3-3'] >= t['Dprev_14-3-3'] and t['K_14-3-3'] < t['D_14-3-3']]
    # No cross
    no_cross = [t for t in d_trades if t not in bull_cross and t not in bear_cross]
    
    for subset, sublabel in [(bull_cross,'Bullish cross'),(bear_cross,'Bearish cross'),(no_cross,'No cross')]:
        if len(subset)<2: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100; net = sum(t['pnl'] for t in subset)
        print(f"    {sublabel}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(subset):+,.0f}")

# ANALYSIS 4: K-distance (how far apart are K and D?)
print(f"\n{'='*80}")
print("  ANALYSIS 4: K-D GAP — Does divergence predict outcome?")
print(f"{'='*80}")

for direction, label in [('B','BUY'),('S','SELL')]:
    d_trades = [t for t in all_trades if t['d']==direction and t.get('K_14-3-3') is not None and t.get('D_14-3-3') is not None]
    if not d_trades: continue
    
    print(f"\n  {label}:")
    for lo, hi, zone in [(0,5,'Tight (0-5)'),(5,15,'Medium (5-15)'),(15,50,'Wide (15+)')]:
        zt = [t for t in d_trades if lo <= abs(t['K_14-3-3']-t['D_14-3-3']) < hi]
        if len(zt)<2: continue
        zw = [t for t in zt if t['pnl']>0]
        wr = len(zw)/len(zt)*100; net = sum(t['pnl'] for t in zt)
        print(f"    K-D gap {zone}: {len(zt)}t WR={wr:.0f}% PnL=${net:+,.0f} avg=${net/len(zt):+,.0f}")

# ANALYSIS 5: Stoch variant comparison
print(f"\n{'='*80}")
print("  ANALYSIS 5: STOCHASTIC VARIANT COMPARISON (14-3-3 vs 8-3-3 vs 5-3-3)")
print(f"{'='*80}")

# Simulate: which variant gives best cross signals?
for variant in ['14-3-3','8-3-3','5-3-3']:
    k_key = f'K_{variant}'; d_key = f'D_{variant}'
    kp_key = f'Kprev_{variant}'; dp_key = f'Dprev_{variant}'
    
    v_trades = [t for t in all_trades if t.get(k_key) is not None]
    if len(v_trades) < 10: continue
    
    # Check cross accuracy
    correct_cross = 0; wrong_cross = 0
    for t in v_trades:
        if t.get(kp_key) is None or t.get(dp_key) is None: continue
        if t['d'] == 'B':
            # We want bullish cross
            had_cross = t[kp_key] <= t[dp_key] and t[k_key] > t[d_key]
        else:
            had_cross = t[kp_key] >= t[dp_key] and t[k_key] < t[d_key]
        if had_cross:
            if t['pnl'] > 0: correct_cross += 1
            else: wrong_cross += 1
    
    total_cross = correct_cross + wrong_cross
    if total_cross > 0:
        print(f"\n  {variant}: {total_cross} trades with cross, {correct_cross}/{total_cross} correct ({correct_cross/total_cross*100:.0f}%)")

# ANALYSIS 6: Most predictive loss pattern
print(f"\n{'='*80}")
print("  ANALYSIS 6: TOP LOSS PATTERNS (Stochastic-specific)")
print(f"{'='*80}")

for direction, label in [('B','BUY'),('S','SELL')]:
    dl = [t for t in loses if t['d']==direction and t.get('K_14-3-3') is not None]
    if not dl: continue
    
    print(f"\n  {label} losses ({len(dl)} total):")
    
    patterns = {}
    for t in dl:
        k = t.get('K_14-3-3',0); d = t.get('D_14-3-3',0)
        if k > 80: patterns['K>80 (overbought)'] = patterns.get('K>80 (overbought)',0)+1
        if k < 20: patterns['K<20 (oversold)'] = patterns.get('K<20 (oversold)',0)+1
        if abs(k-d) > 15: patterns['K-D gap >15'] = patterns.get('K-D gap >15',0)+1
        if abs(k-d) < 3: patterns['K-D tight <3'] = patterns.get('K-D tight <3',0)+1
    
    for pattern, cnt in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pattern}: {cnt}/{len(dl)} losses ({cnt/len(dl)*100:.0f}%)")

print(f"\n{'='*80}")
print(f"  COMPLETE ({(datetime.now()-T0).total_seconds():.0f}s)")
print(f"{'='*80}")
