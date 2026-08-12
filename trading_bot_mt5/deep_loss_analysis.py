"""
DEEP PATTERN ANALYSIS — 6-month backtest loss reduction
Finds every pattern that separates winners from losers.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "deep_patterns.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  DEEP PATTERN ANALYSIS — Loss Reduction")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull 6 months
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

# 6-month backtest with current rules
print("Running baseline backtest...")
bal=10000; trades=[]; pos=None
for i in range(400,n):
    if np.isnan(close[i]): continue
    px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
    if np.isnan(rsi): continue
    maf=sma40[i]; mas=sma200[i]; bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
    sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1] if i>0 else sk_i; psd=sd[i-1] if i>0 else sd_i
    vn=vol[i]; vma=vma20[i]; ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]
    pmh=macd_h[i-1] if i>0 else mh; hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
    
    if np.isnan(ml) or np.isnan(adx): continue
    if adx<25: continue
    if not (8<=hr<22): continue
    
    if pos is not None:
        sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
        tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
        if sl_hit or tp_hit:
            ep=pos['sl'] if sl_hit else pos['tp']
            pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
            bal+=pnl
            trades.append({'d':pos['d'],'pnl':pnl,'reason':'sl' if sl_hit else 'tp',
                          'rsi':pos['rsi'],'adx':pos['adx'],'macd_bull':pos['macd_bull'],
                          'stoch_k':pos['stoch_k'],'stoch_d':pos['stoch_d'],
                          'bb_pos':pos['bb_pos'],'atr':pos['atr'],'vol_ratio':pos.get('vol_ratio',1),
                          'hour':pos['hour'],'entry':pos['e'],'bars_held':i-pos['bar'],
                          'mfe_pct':pos.get('mfe_pct',0)})
            pos=None
        else:
            # Track MFE
            if pos['d']=='B':
                mfe=max(pos.get('mfe',0),(high[i]-pos['e'])/pos['e']*100)
            else:
                mfe=max(pos.get('mfe',0),(pos['e']-low[i])/pos['e']*100)
            pos['mfe']=mfe
        continue
    
    # BUY
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
    if trend_up:
        buy_ok=False; buy_type=''
        if rsi>=72:
            if not (psk>psd and sk_i<sd_i): buy_ok=True; buy_type='breakout'
        if not buy_ok and 40<=rsi<=65:
            nb=abs(px-bm)/bm<0.01
            if nb and mh>pmh and psk<=psd and sk_i>sd_i: buy_ok=True; buy_type='pullback'
        
        if buy_ok:
            sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
            bb_label = 'ABOVE' if px>bm else 'BELOW'
            pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':max(0.01,round((bal*0.02)/(sld*100),2)),
                 'rsi':rsi,'adx':adx,'macd_bull':ml>ms_,'stoch_k':sk_i,'stoch_d':sd_i,
                 'bb_pos':bb_label,'atr':atr,'vol_ratio':vn/vma if vma>0 else 1,'hour':hr,'bar':i,'mfe':0}
            continue
    
    # SELL
    if not (not np.isnan(maf) and px>maf):
        # MACD mandatory gate
        if ml>ms_: continue  # NEW: MACD bullish = no SELL
        
        checks=0
        if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
        if ml<ms_ and mh<0: checks+=1
        if psk>=psd and sk_i<sd_i: checks+=1
        if abs(px-bl)/bl>0.005: checks+=1
        if px<bm: checks+=1
        
        if checks>=3:
            sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
            bb_label = 'ABOVE' if px>bm else 'BELOW'
            pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':max(0.01,round((bal*0.02)/(sld*100),2)),
                 'rsi':rsi,'adx':adx,'macd_bull':ml>ms_,'stoch_k':sk_i,'stoch_d':sd_i,
                 'bb_pos':bb_label,'atr':atr,'vol_ratio':vn/vma if vma>0 else 1,'hour':hr,'bar':i,'mfe':0}

if pos is not None:
    ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
    bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod','rsi':pos['rsi']})

wins=[t for t in trades if t['pnl']>0]
loses=[t for t in trades if t['pnl']<=0]
print(f"Backtest: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.2f}, WR={len(wins)/len(trades)*100:.1f}%")
print(f"Wins: {len(wins)}, Losses: {len(loses)}")

# ============================================================
# DEEP PATTERN ANALYSIS ON BACKTEST
# ============================================================
print(f"\n{'='*80}")
print("  PATTERN ANALYSIS: What separates winners from losers?")
print(f"{'='*80}")

# For each feature, compute win rate for different buckets
features = [
    ('rsi', 'RSI at Entry', [(0,40),(40,55),(55,65),(65,80)]),
    ('adx', 'ADX at Entry', [(0,20),(20,30),(30,45),(45,100)]),
    ('bb_pos', 'BB Position', ['ABOVE','BELOW']),
    ('hour', 'Hour of Day', [(0,8),(8,13),(13,17),(17,22)]),
    ('stoch_k', 'Stochastic K', [(0,30),(30,50),(50,70),(70,100)]),
    ('atr', 'ATR', [(0,7),(7,10),(10,15),(15,100)]),
    ('vol_ratio', 'Volume Ratio', [(0,0.5),(0.5,1.0),(1.0,2.0),(2.0,100)]),
    ('bars_held', 'Bars Held', [(0,10),(10,40),(40,100),(100,1000)]),
    ('macd_bull', 'MACD Bullish', [True, False]),
]

for key, label, buckets in features:
    print(f"\n  {label}:")
    for bucket in buckets:
        if isinstance(bucket, tuple):
            lo,hi = bucket
            subset = [t for t in trades if t.get(key) is not None and lo <= t[key] < hi]
            bname = f"{lo}-{hi}"
        else:
            subset = [t for t in trades if t.get(key) == bucket]
            bname = str(bucket)
        
        if len(subset)<2: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100
        pnl = sum(t['pnl'] for t in subset)
        print(f"    {bname}: {len(subset)}t WR={wr:.0f}% PnL=${pnl:+,.0f} avg=${pnl/len(subset):+,.0f}")

# Direction breakdown
print(f"\n{'='*80}")
print("  DIRECTION PATTERNS")
print(f"{'='*80}")
for d in ['B','S']:
    dt = [t for t in trades if t['d']==d]
    dw = [t for t in dt if t['pnl']>0]
    dl = [t for t in dt if t['pnl']<=0]
    print(f"\n  {d}UY:" if d=='B' else f"\n  SELL:")
    print(f"    Total: {len(dt)}t WR={len(dw)/len(dt)*100:.0f}% PnL=${sum(t['pnl'] for t in dt):+,.0f}")
    
    # What makes BUY losses?
    if dl:
        print(f"    Loss patterns:")
        loss_attrs = []
        for t in dl:
            attrs = []
            if t.get('bb_pos')=='ABOVE': attrs.append('BB=above')
            if t.get('rsi',0)>65: attrs.append(f'RSI={t["rsi"]:.0f}>65')
            if t.get('macd_bull')==False: attrs.append('MACD=bear')
            if t.get('adx',0)<25: attrs.append(f'ADX={t["adx"]:.0f}<25')
            if t.get('hour',12)<8: attrs.append(f'hour={t["hour"]}')
            if t.get('stoch_k',0)>80: attrs.append(f'StochK={t["stoch_k"]:.0f}>80')
            loss_attrs.extend(attrs)
        from collections import Counter
        for attr, cnt in Counter(loss_attrs).most_common(5):
            print(f"      {attr}: {cnt}/{len(dl)} losses")

# ============================================================
# FILTER TEST: Apply single rules and measure impact
# ============================================================
print(f"\n{'='*80}")
print("  CANDIDATE FILTERS — Impact on 6-month backtest")
print(f"{'='*80}")

candidate_filters = {
    'MACD mandatory for SELL': lambda t: t['d']!='S' or t.get('macd_bull')==False,
    'No BUY when Stoch K > 80': lambda t: t['d']!='B' or (t.get('stoch_k',0) <= 80),
    'No BUY with MACD bearish': lambda t: t['d']!='B' or t.get('macd_bull') != False,
    'No SELL with RSI < 35': lambda t: t['d']!='S' or (t.get('rsi',50) >= 35),
    'No BUY when ATR > 12': lambda t: t['d']!='B' or (t.get('atr',0) <= 12),
    'No trade with vol ratio < 0.3': lambda t: t.get('vol_ratio',1) >= 0.3,
    'No SELL in overlap (13-17h)': lambda t: t['d']!='S' or not (13 <= t.get('hour',0) < 17),
    'BB mandatory for BUY (must be below mid)': lambda t: t['d']!='B' or t.get('bb_pos')=='BELOW',
}

baseline_pnl = sum(t['pnl'] for t in trades)
baseline_wr = len(wins)/len(trades)*100
print(f"\n  BASELINE: {len(trades)}t PnL=${baseline_pnl:+,.0f} WR={baseline_wr:.0f}%")

improvements = []
for name, fn in candidate_filters.items():
    kept = [t for t in trades if fn(t)]
    blocked = [t for t in trades if not fn(t)]
    if len(kept) < 5: continue
    
    kept_w = [t for t in kept if t['pnl']>0]
    kept_l = [t for t in kept if t['pnl']<=0]
    blocked_w = [t for t in blocked if t['pnl']>0]
    blocked_l = [t for t in blocked if t['pnl']<=0]
    
    kept_pnl = sum(t['pnl'] for t in kept)
    blocked_pnl = sum(t['pnl'] for t in blocked)
    improvement = kept_pnl - baseline_pnl
    win_kept_pct = len(kept_w)/len(wins)*100 if wins else 0
    loss_blocked_pct = len(blocked_l)/len(loses)*100 if loses else 0
    
    improvements.append((improvement, name, win_kept_pct, loss_blocked_pct, kept_pnl, len(kept_w), len(blocked_l)))

improvements.sort(reverse=True)

for imp, name, wk, lb, kpnl, kw, bl in improvements:
    verdict = "KEEP" if imp > 0 else "SKIP"
    print(f"\n  {verdict}: {name}")
    print(f"    PnL change: ${imp:+,.0f} (new: ${kpnl:+,.0f})")
    print(f"    Win retention: {wk:.0f}% ({kw}/{len(wins)} kept)")
    print(f"    Loss reduction: {lb:.0f}% ({bl}/{len(loses)} blocked)")

# ============================================================
# COMBINED BEST FILTERS
# ============================================================
print(f"\n{'='*80}")
print("  COMBINED FILTERS (apply all positive-impact rules)")
print(f"{'='*80}")

# Apply all filters that improved
positive = [(name, fn) for imp, name, wk, lb, kpnl, kw, bl in improvements if imp > 0]
combined_kept = trades
for name, fn in positive:
    combined_kept = [t for t in combined_kept if fn(t)]

if len(combined_kept) >= 5:
    cw = [t for t in combined_kept if t['pnl']>0]
    cl = [t for t in combined_kept if t['pnl']<=0]
    cp = sum(t['pnl'] for t in combined_kept)
    
    print(f"\n  Combined result: {len(combined_kept)} trades, PnL=${cp:+,.0f}, WR={len(cw)/len(combined_kept)*100:.0f}%")
    print(f"  Original: {len(trades)} trades, PnL=${baseline_pnl:+,.0f}, WR={baseline_wr:.0f}%")
    print(f"  Improvement: ${cp-baseline_pnl:+,.0f}")
    print(f"  Filters applied: {', '.join(n for n,_ in positive)}")

print(f"\n{'='*80}")
print(f"  ANALYSIS COMPLETE ({((datetime.now()-T0).total_seconds()):.0f}s)")
print(f"{'='*80}")
