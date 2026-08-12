"""
BACKTEST V2.0 AGAINST ACTUAL MT5 TRADES
Replays every historical trade through the optimized engine.
Shows: which trades would have been blocked, which would have been taken,
and the resulting P&L difference.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "backtest_actual_trades.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  V2.0 BACKTEST vs ACTUAL MT5 TRADES")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Load actual trades
with open(r'C:\Users\ASUS\extract_mt5_trades_output.json') as f:
    trade_data = json.load(f)
actual_trades = [t for t in trade_data['trades'] if t['volume'] > 0 and t['entry_price'] > 0 and t.get('exit_price')]
print(f"\nLoaded {len(actual_trades)} closed trades from MT5")

# Pull 6 months of M15 data (need enough history for SMA 200)
print("Pulling M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=180)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")

close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float); vol = df['tick_volume'].values.astype(float)
n = len(close)
hours = np.array([t.hour for t in df.index])

# Indicators (same as v2.0)
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

# RSI 14
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

print("Indicators ready.\n")

# ---- V2.0 Decision Engine (exact replica) ----
def v2_decide(price, rsi, maf, mas, bu, bm, bl, sk_i, sd_i, psk, psd, vn, vma, ml, ms_, mh, pmh, px, adx, hour_now):
    """Returns 'BUY', 'SELL', or 'NONE' based on v2.0 rules."""
    
    # ADX filter
    if adx < 25: return 'NONE'
    
    # Session filter
    if not (8 <= hour_now < 22): return 'NONE'
    
    # BUY check
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf > mas
    if trend_up:
        # Option 1: Breakout
        if rsi >= 72:
            if not (psk > psd and sk_i < sd_i):  # no bearish cross
                return 'BUY'
        # Option 2: Pullback
        if 40 <= rsi <= 65:
            near_bb = abs(price - bm) / bm < 0.01
            macd_ok = mh > pmh
            stoch_ok = psk <= psd and sk_i > sd_i
            if near_bb and macd_ok and stoch_ok:
                return 'BUY'
    
    # SELL check
    if not np.isnan(maf) and price > maf:
        return 'NONE'  # Above MA, no sell
    
    checks = 0
    if 30 <= rsi <= 50 and rsi < rsi14[i-1] if i>0 else True: checks += 1
    if ml < ms_ and mh < 0: checks += 1
    if psk >= psd and sk_i < sd_i: checks += 1
    if abs(price - bl) / bl > 0.005: checks += 1
    if price < bm: checks += 1
    if checks >= 3:
        return 'SELL'
    
    return 'NONE'

# ---- Analyze each actual trade ----
print("=" * 80)
print("  TRADE-BY-TRADE ANALYSIS")
print("=" * 80)

results = []
for trade in actual_trades:
    try:
        et_str = trade['entry_time']
        if not et_str: continue
        et = datetime.fromisoformat(et_str).replace(tzinfo=None)
        
        # Find the bar index closest to entry time
        idx = df.index.get_indexer([et], method='nearest')[0]
        if idx < 200 or idx >= n: continue  # Need enough history
        
        i = idx
        px = close[i]; rsi = rsi14[i]; adx = adx14[i]
        maf=sma40[i]; mas=sma200[i]
        bu=bb_u[i]; bm=bb_mid[i]; bl=bb_l[i]
        sk_i=sk[i]; sd_i=sd[i]; psk=sk[i-1]; psd=sd[i-1]
        vn=vol[i]; vma=vma20[i]
        ml=macd_l[i]; ms_=macd_s[i]; mh=macd_h[i]; pmh=macd_h[i-1]
        hour_now = hours[i]
        
        v2_signal = v2_decide(px, rsi, maf, mas, bu, bm, bl, sk_i, sd_i, psk, psd, vn, vma, ml, ms_, mh, pmh, px, adx, hour_now)
        actual_dir = trade['direction']
        actual_pnl = trade['profit']
        
        # Determine why blocked
        reasons = []
        if adx < 25: reasons.append(f"ADX={adx:.0f}<25")
        if not (8 <= hour_now < 22): reasons.append(f"outside hours ({hour_now}h)")
        if actual_dir == 'BUY' and v2_signal != 'BUY':
            if not (not np.isnan(maf) and not np.isnan(mas) and maf>mas): reasons.append("no trend")
            elif rsi >= 72: reasons.append(f"RSI={rsi:.0f} breakout but Stoch bearish")
            elif 40 <= rsi <= 65: 
                nb = abs(px-bm)/bm>=0.01
                mo = not (mh>pmh)
                so = not (psk<=psd and sk_i>sd_i)
                if nb: reasons.append("not near BB mid")
                if mo: reasons.append("MACD not rising")
                if so: reasons.append("no Stoch cross")
            else: reasons.append(f"RSI={rsi:.0f} outside zones")
        if actual_dir == 'SELL' and v2_signal != 'SELL':
            if not np.isnan(maf) and px>maf: reasons.append("price above MA40")
            c=0
            if not (30<=rsi<=50): reasons.append(f"RSI={rsi:.0f} not 30-50")
            if not (ml<ms_ and mh<0): reasons.append("MACD not bearish")
            if not (psk>=psd and sk_i<sd_i): reasons.append("no Stoch bearish cross")
            if not (abs(px-bl)/bl>0.005): reasons.append("on lower BB")
            if not (px<bm): reasons.append("above BB mid")
        
        would_take = v2_signal == actual_dir
        avoided_loss = -actual_pnl if not would_take and actual_pnl <= 0 else 0
        kept_profit = actual_pnl if would_take and actual_pnl > 0 else 0
        
        results.append({
            'ticket': trade['ticket'],
            'direction': actual_dir,
            'entry': trade['entry_price'],
            'exit': trade['exit_price'],
            'actual_pnl': actual_pnl,
            'v2_signal': v2_signal,
            'would_take': would_take,
            'avoided_loss': avoided_loss,
            'kept_profit': kept_profit,
            'reasons': reasons,
            'date': et_str[:16],
            'rsi': round(rsi,1) if not np.isnan(rsi) else None,
            'adx': round(adx,1) if not np.isnan(adx) else None,
            'exit_reason': trade['exit_reason'],
        })
    except Exception as e:
        print(f"  Error on trade {trade.get('ticket','?')}: {e}")

# ---- RESULTS ----
taken = [r for r in results if r['would_take']]
blocked = [r for r in results if not r['would_take']]
blocked_losses = [r for r in blocked if r['actual_pnl'] <= 0]
blocked_wins = [r for r in blocked if r['actual_pnl'] > 0]

total_avoided = sum(r['avoided_loss'] for r in results)
total_kept = sum(r['kept_profit'] for r in results)

actual_total = sum(r['actual_pnl'] for r in results)
v2_total = sum(r['actual_pnl'] for r in taken)  # Only trades v2 would take

print(f"\n  Summary:")
print(f"  Total trades analyzed: {len(results)}")
print(f"  V2 would TAKE:   {len(taken)} trades | PnL: ${sum(r['actual_pnl'] for r in taken):,.2f}")
print(f"  V2 would BLOCK:  {len(blocked)} trades | Actual PnL: ${sum(r['actual_pnl'] for r in blocked):,.2f}")
print(f"    Blocked LOSERS: {len(blocked_losses)} trades | Losses avoided: ${sum(r['actual_pnl'] for r in blocked_losses):,.2f}")
print(f"    Blocked WINNERS: {len(blocked_wins)} trades | Profit missed: ${sum(r['actual_pnl'] for r in blocked_wins):,.2f}")
print(f"\n  Actual total PnL:   ${actual_total:,.2f}")
print(f"  V2 filtered PnL:    ${v2_total:,.2f}")
print(f"  Improvement:        ${v2_total - actual_total:+,.2f}")

print(f"\n  >>> TRADES V2 WOULD BLOCK (avoided losses):")
for r in sorted(blocked_losses, key=lambda x: x['actual_pnl']):
    print(f"  {r['direction']} ${r['actual_pnl']:+,.2f} | RSI={r['rsi']} ADX={r['adx']} | {', '.join(r['reasons'])} | {r['date']}")

print(f"\n  >>> TRADES V2 WOULD BLOCK (missed profits):")
for r in sorted(blocked_wins, key=lambda x: x['actual_pnl'], reverse=True):
    print(f"  {r['direction']} ${r['actual_pnl']:+,.2f} | RSI={r['rsi']} ADX={r['adx']} | {', '.join(r['reasons'])} | {r['date']}")

print(f"\n  >>> TRADES V2 WOULD TAKE:")
for r in sorted(taken, key=lambda x: x['actual_pnl'], reverse=True):
    print(f"  {r['direction']} ${r['actual_pnl']:+,.2f} | RSI={r['rsi']} ADX={r['adx']} | exit:{r['exit_reason']} | {r['date']}")

print(f"\n{'='*80}")
print(f"  FINAL VERDICT")
print(f"{'='*80}")
print(f"  Actual PnL:   ${actual_total:,.2f}")
print(f"  V2 PnL:       ${v2_total:,.2f}")
print(f"  Difference:   ${v2_total - actual_total:+,.2f}")
if v2_total > actual_total:
    print(f"  >>> V2 IMPROVES by ${v2_total - actual_total:,.2f}")
else:
    print(f"  >>> V2 would have been worse by ${actual_total - v2_total:,.2f}")
print(f"{'='*80}")
