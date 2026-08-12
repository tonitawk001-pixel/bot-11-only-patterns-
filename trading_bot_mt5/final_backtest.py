"""
FINAL COMPREHENSIVE BACKTEST + PATTERN ANALYSIS
1. Apply adaptive engine to bot
2. Test on never-seen month (May 2026)
3. Analyze ALL loss patterns from all backtests
4. Find new adaptive behaviors
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(OUT_DIR, "final_analysis.txt")
sys.stdout = open(LOG, 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  FINAL COMPREHENSIVE ANALYSIS")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# ---- Pull data: 4 different periods ----
print("\nPulling all available M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)

periods = {
    'May 2026 (NEW - unseen)': (datetime(2026,5,1,tzinfo=timezone.utc), datetime(2026,6,1,tzinfo=timezone.utc)),
    'Jun-Jul 2026': (datetime(2026,6,1,tzinfo=timezone.utc), datetime(2026,8,1,tzinfo=timezone.utc)),
    'Feb-May 2026': (datetime(2026,2,1,tzinfo=timezone.utc), datetime(2026,5,31,tzinfo=timezone.utc)),
    'Jan-Jun 2025 (OOS)': (datetime(2025,1,1,tzinfo=timezone.utc), datetime(2025,7,1,tzinfo=timezone.utc)),
}

datasets = {}
for name, (start, end) in periods.items():
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, start, end)
    if rates is not None and len(rates) > 500:
        df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        df.columns = [c.lower() for c in df.columns]
        datasets[name] = df
        print(f"  {name}: {len(df)} candles")

mt5.shutdown()

# ---- Backtest engine ----
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

class AdaptiveFilter:
    def __init__(self, name, init_score=50):
        self.name = name; self.score = init_score
        self.blocks = 0; self.good = 0; self.bad = 0
    def is_active(self):
        if self.blocks < 5: return True
        return self.score >= 35
    def record(self):
        self.blocks += 1
    def feedback(self, was_winner):
        if was_winner: self.score = max(10, self.score - 10); self.bad += 1
        else: self.score = min(100, self.score + 8); self.good += 1

def compute_indicators(df):
    close=df['close'].values.astype(float); high=df['high'].values.astype(float)
    low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
    n=len(close); hours=np.array([t.hour for t in df.index])
    
    d=np.zeros(n); d[1:]=np.diff(close); g=np.maximum(d,0); l=np.maximum(-d,0)
    ag=np.full(n,np.nan); al=np.full(n,np.nan)
    if n>14: ag[14]=np.mean(g[1:15]); al[14]=np.mean(l[1:15])
    for i in range(15,n): ag[i]=(ag[i-1]*13+g[i])/14; al[i]=(al[i-1]*13+l[i])/14
    rs=np.full(n,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
    rsi14=np.full(n,np.nan); rsi14[m]=100-100/(1+rs[m])
    
    atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
    sma40_=sma(close,40); sma200_=sma(close,200)
    bm=sma(close,20); bs=pd.Series(close).rolling(20).std().values
    bu=bm+bs*2; bl=bm-bs*2
    ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
    rk=np.full(n,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
    sk_=pd.Series(rk).rolling(3).mean().values; sd_=pd.Series(sk_).rolling(3).mean().values
    ef=ema(close,12); es=ema(close,26); ml=ef-es; ms=ema(ml,9); mh=ml-ms
    vma=sma(vol,20)
    
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
        up=high[i]-high[i-1]; dn=low[i-1]-low[i]
        pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
    ae=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
    dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
    adx14=ema(dx,14)
    
    return close,high,low,vol,hours,rsi14,atr14,sma40_,sma200_,bu,bm,bl,sk_,sd_,ml,ms,mh,vma,adx14

def backtest_period(df, period_name):
    close,high,low,vol,hours,rsi14,atr14,sma40_,sma200_,bu,bm,bl,sk_,sd_,ml,ms,mh,vma,adx14 = compute_indicators(df)
    n=len(close)
    
    # Adaptive filters
    adx_f = AdaptiveFilter("ADX>25", 55)
    sess_f = AdaptiveFilter("Session", 40)
    macd_f = AdaptiveFilter("MACD Gate", 35)
    overlap_f = AdaptiveFilter("Overlap", 35)
    
    bal=10000; trades=[]; pos=None; sim_trades=[]
    
    for i in range(400,n):
        if np.isnan(close[i]): continue
        px=close[i]; rsi=rsi14[i]; adx=adx14[i] if not np.isnan(adx14[i]) else 50
        if np.isnan(rsi): continue
        maf=sma40_[i]; mas=sma200_[i]
        sk_i=sk_[i]; sd_i=sd_[i]; psk=sk_[i-1] if i>0 else sk_i; psd=sd_[i-1] if i>0 else sd_i
        ml_i=ml[i]; ms_i=ms[i]; mh_i=mh[i]; pmh=mh[i-1] if i>0 else mh_i
        hr=hours[i]; atr=atr14[i] if not np.isnan(atr14[i]) else 10
        if np.isnan(ml_i) or np.isnan(adx): continue
        
        # Manage position
        if pos is not None:
            sl_hit=(pos['d']=='B' and px<=pos['sl']) or (pos['d']=='S' and px>=pos['sl'])
            tp_hit=(pos['d']=='B' and px>=pos['tp']) or (pos['d']=='S' and px<=pos['tp'])
            if sl_hit or tp_hit:
                ep=pos['sl'] if sl_hit else pos['tp']
                pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
                bal+=pnl
                # Feedback
                for sim in sim_trades[-20:]:
                    for fname, filt in [('adx',adx_f),('session',sess_f),('macd',macd_f),('overlap',overlap_f)]:
                        if sim.get(f'blocked_{fname}'): filt.feedback(sim['would_win'])
                sim_trades=[]
                trades.append({'d':pos['d'],'pnl':pnl,'rsi':pos.get('rsi'),'adx':pos.get('adx'),
                              'macd_bull':pos.get('macd_bull'),'bb':pos.get('bb'),'hour':pos.get('hour'),
                              'stoch_k':pos.get('sk'),'atr':pos.get('atr'),'vol_r':pos.get('vr')}); pos=None
            continue
        
        # BUY signal
        trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
        buy_ok=False
        if trend_up:
            if rsi>=72 and not (psk>psd and sk_i<sd_i): buy_ok=True
            elif 40<=rsi<=65 and abs(px-bm[i])/bm[i]<0.01 and mh_i>pmh and psk<=psd and sk_i>sd_i: buy_ok=True
        
        vn=vol[i]; vma_v=vma[i]
        bb_label='ABOVE' if px>bm[i] else 'BELOW'
        
        if buy_ok:
            blocked=False; blocked_by={}
            if adx_f.is_active() and adx<25: blocked=True; blocked_by['adx']=True; adx_f.record()
            if sess_f.is_active() and not (8<=hr<22): blocked=True; blocked_by['session']=True; sess_f.record()
            
            if blocked:
                would_win=False; sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
                for j in range(i+1,min(i+300,n)):
                    if high[j]>=tp: would_win=True; break
                    if low[j]<=sl: break
                sim_trades.append({**{f'blocked_{k}':v for k,v in blocked_by.items()},'would_win':would_win})
            else:
                sld=atr*3.0; sl=px-sld; tp=px+sld*2.0
                lots=max(0.01,round((bal*0.02)/(sld*100),2))
                pos={'d':'B','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi,'adx':adx,
                     'macd_bull':ml_i>ms_i,'bb':bb_label,'hour':hr,'sk':sk_i,'atr':atr,'vr':vn/vma_v if vma_v>0 else 1}
            continue
        
        # SELL signal
        if not (not np.isnan(maf) and px>maf):
            blocked=False; blocked_by={}
            if adx_f.is_active() and adx<25: blocked=True; blocked_by['adx']=True; adx_f.record()
            if sess_f.is_active() and not (8<=hr<22): blocked=True; blocked_by['session']=True; sess_f.record()
            if macd_f.is_active() and ml_i>ms_i: blocked=True; blocked_by['macd']=True; macd_f.record()
            if overlap_f.is_active() and (13<=hr<17): blocked=True; blocked_by['overlap']=True; overlap_f.record()
            
            checks=0
            if 30<=rsi<=50 and i>0 and rsi<rsi14[i-1]: checks+=1
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
                    sim_trades.append({**{f'blocked_{k}':v for k,v in blocked_by.items()},'would_win':would_win})
                else:
                    sld=atr*3.0; sl=px+sld; tp=px-sld*2.0
                    lots=max(0.01,round((bal*0.02)/(sld*100),2))
                    pos={'d':'S','e':px,'sl':sl,'tp':tp,'l':lots,'rsi':rsi,'adx':adx,
                         'macd_bull':ml_i>ms_i,'bb':bb_label,'hour':hr,'sk':sk_i,'atr':atr,'vr':vn/vma_v if vma_v>0 else 1}
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos['e'])*100*pos['l'] if pos['d']=='B' else (pos['e']-ep)*100*pos['l']
        bal+=pnl; trades.append({'d':pos['d'],'pnl':pnl,'reason':'eod'})
    
    if len(trades)<3: return None,None,None
    
    wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
    net=sum(t['pnl'] for t in trades)
    pf=abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in loses)) if loses and sum(t['pnl'] for t in loses)!=0 else 0
    
    return {
        'period':period_name, 'net':round(net,2), 'trades':len(trades),
        'wr':round(len(wins)/len(trades)*100,1), 'pf':round(pf,2),
        'adx_score':adx_f.score,'session_score':sess_f.score,
        'macd_score':macd_f.score,'overlap_score':overlap_f.score,
    }, wins, loses

# ---- RUN ALL PERIODS ----
print("\n" + "=" * 80)
print("  BACKTEST RESULTS ACROSS ALL PERIODS")
print("=" * 80)

all_results = []
all_losses = []

for name, df in datasets.items():
    result, wins, loses = backtest_period(df, name)
    if result:
        all_results.append(result)
        if loses: all_losses.extend([{**l, 'period':name} for l in loses])
        verdict = "PROFITABLE" if result['net']>0 else "UNPROFITABLE"
        print(f"\n  {name}:")
        print(f"    ${result['net']:+,.0f} | {result['trades']}t | WR={result['wr']}% | PF={result['pf']} — {verdict}")
        print(f"    Filters: ADX={result['adx_score']:.0f} Sess={result['session_score']:.0f} MACD={result['macd_score']:.0f} Ovlp={result['overlap_score']:.0f}")

# ---- CROSS-PERIOD LOSS PATTERN ANALYSIS ----
print(f"\n{'='*80}")
print(f"  CROSS-PERIOD LOSS PATTERNS ({len(all_losses)} total losses)")
print(f"{'='*80}")

if all_losses:
    # Most common loss characteristics
    from collections import Counter
    
    print(f"\n  Losses by direction:")
    for d,lbl in [('B','BUY'),('S','SELL')]:
        dl=[t for t in all_losses if t.get('d')==d]
        print(f"    {lbl}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Losses by RSI at entry:")
    for lo,hi,lbl in [(0,35,'0-35'),(35,50,'35-50'),(50,65,'50-65'),(65,100,'65+')]:
        dl=[t for t in all_losses if t.get('rsi') and lo<=t['rsi']<hi]
        if dl: print(f"    RSI {lbl}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Losses by ADX at entry:")
    for lo,hi,lbl in [(0,25,'<25'),(25,40,'25-40'),(40,100,'40+')]:
        dl=[t for t in all_losses if t.get('adx') and lo<=t['adx']<hi]
        if dl: print(f"    ADX {lbl}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Losses by BB position:")
    for pos in ['ABOVE','BELOW']:
        dl=[t for t in all_losses if t.get('bb')==pos]
        if dl: print(f"    BB {pos}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Losses by MACD:")
    for val,lbl in [(True,'Bullish'),(False,'Bearish')]:
        dl=[t for t in all_losses if t.get('macd_bull')==val]
        if dl: print(f"    MACD {lbl}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Losses by Hour:")
    for lo,hi,lbl in [(0,8,'Asian 0-8'),(8,13,'London AM'),(13,17,'Overlap'),(17,22,'NY PM')]:
        dl=[t for t in all_losses if t.get('hour') and lo<=t['hour']<hi]
        if dl: print(f"    {lbl}: {len(dl)} losses, avg=${sum(t['pnl'] for t in dl)/len(dl):+,.0f}")
    
    print(f"\n  Top 3 Loss Patterns (intersection):")
    # Losses with: BB ABOVE + MACD Bearish + RSI>50
    pattern1=[t for t in all_losses if t.get('bb')=='ABOVE' and t.get('macd_bull')==False]
    print(f"    BB=ABOVE + MACD=Bearish: {len(pattern1)} losses, ${sum(t['pnl'] for t in pattern1):+,.0f}")
    pattern2=[t for t in all_losses if t.get('d')=='S' and t.get('bb')=='ABOVE']
    print(f"    SELL + BB=ABOVE: {len(pattern2)} losses, ${sum(t['pnl'] for t in pattern2):+,.0f}")
    pattern3=[t for t in all_losses if t.get('rsi') and t['rsi']>60 and t.get('d')=='B']
    print(f"    BUY + RSI>60: {len(pattern3)} losses, ${sum(t['pnl'] for t in pattern3):+,.0f}")

# ---- FINAL VERDICT ----
print(f"\n{'='*80}")
print("  FINAL VERDICT")
print(f"{'='*80}")

profitable = sum(1 for r in all_results if r['net']>0)
total_periods = len(all_results)
total_net = sum(r['net'] for r in all_results)

print(f"\n  Profitable periods: {profitable}/{total_periods}")
print(f"  Total net across all periods: ${total_net:+,.0f}")
print(f"  Average per period: ${total_net/total_periods:+,.0f}" if total_periods else "")

if profitable == total_periods:
    print(f"\n  >>> ALL PERIODS PROFITABLE — Bot is market-agnostic")
elif profitable >= total_periods * 0.75:
    print(f"\n  >>> MOST periods profitable — Bot is robust with some regime sensitivity")
else:
    print(f"\n  >>> Bot needs more work — only {profitable}/{total_periods} profitable")

print(f"\n{'='*80}")
print(f"  COMPLETE ({(datetime.now()-T0).total_seconds():.0f}s)")
print(f"{'='*80}")
