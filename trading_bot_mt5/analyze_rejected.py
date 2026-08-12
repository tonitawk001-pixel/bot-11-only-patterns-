"""
ANALYZE TODAY'S REJECTED TRADES
For every 15-min bar today where the bot considered a trade but rejected it,
simulate what would have happened if it took the trade anyway.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "rejected_analysis.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  TODAY'S REJECTED TRADES ANALYSIS")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Pull today's M15 data + some history for indicators
print("\nPulling M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc)
from_dt = to_dt - timedelta(days=30)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()

df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
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

# ADX
tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
for i in range(1,n):
    tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    up=high[i]-high[i-1]; dn=low[i-1]-low[i]
    pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
a_e=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

print(f"Got {len(df)} bars: {df.index[0]} to {df.index[-1]}")

# Today's date
today = to_dt.strftime('%Y-%m-%d')
today_bars = df[df.index.strftime('%Y-%m-%d') == today]
print(f"Today's bars: {len(today_bars)} ({today_bars.index[0]} to {today_bars.index[-1]})")

# Simulate each bar of today
print("\n" + "=" * 80)
print("  REJECTED TRADE SIMULATION")
print("=" * 80)

rejected_trades = []
total_rejected_pnl = 0
total_taken_pnl = 0

for bar_time, bar in today_bars.iterrows():
    idx = df.index.get_indexer([bar_time], method='nearest')[0]
    if idx < 200: continue
    
    px = close[idx]; rsi = rsi14[idx]; adx = adx14[idx]
    maf = sma40[idx]; mas = sma200[idx]
    bu = bb_u[idx]; bm = bb_mid[idx]; bl = bb_l[idx]
    sk_i = sk[idx]; sd_i = sd[idx]; psk = sk[idx-1]; psd = sd[idx-1]
    ml = macd_l[idx]; ms_ = macd_s[idx]; mh = macd_h[idx]; pmh = macd_h[idx-1]
    hr = hours[idx]; atr = atr14[idx] if not np.isnan(atr14[idx]) else 5
    vn = vol[idx]; vma = vma20[idx]
    
    if np.isnan(rsi) or np.isnan(ml) or np.isnan(adx): continue
    
    # Pre-filters (same as main_super.py)
    if adx < 25:
        blocked_reason = f"ADX={adx:.0f}<25"
    elif not (8 <= hr < 22):
        blocked_reason = f"outside hours ({hr}h)"
    else:
        blocked_reason = None
    
    # Check BUY signal
    buy_signal = False; buy_type = ''
    trend_up = not np.isnan(maf) and not np.isnan(mas) and maf > mas
    if trend_up:
        if rsi >= 72:
            if not (psk > psd and sk_i < sd_i): buy_signal = True; buy_type = 'breakout'
        if not buy_signal and 40 <= rsi <= 65:
            nb = abs(px - bm) / bm < 0.01
            if nb and mh > pmh and psk <= psd and sk_i > sd_i: buy_signal = True; buy_type = 'pullback'
    
    # Check SELL signal
    sell_signal = False
    if not (not np.isnan(maf) and px > maf):
        checks = 0
        reasons = []
        if 30 <= rsi <= 50:
            if idx > 0 and rsi < rsi14[idx-1]: checks += 1; reasons.append('rsi')
        if ml < ms_ and mh < 0: checks += 1; reasons.append('macd')
        if psk >= psd and sk_i < sd_i: checks += 1; reasons.append('stoch')
        if abs(px - bl) / bl > 0.005: checks += 1; reasons.append('location')
        if px < bm: checks += 1; reasons.append('bb')
        if checks >= 3:
            sell_signal = True
    
    direction = 'BUY' if buy_signal else ('SELL' if sell_signal else None)
    if direction is None: continue
    
    # Was it approved or rejected?
    approved = (blocked_reason is None)
    
    # Simulate the trade if it was taken
    entry_px = px
    sl = entry_px - atr * 3.0 if direction == 'BUY' else entry_px + atr * 3.0
    tp = entry_px + (atr * 3.0) * 2.0 if direction == 'BUY' else entry_px - (atr * 3.0) * 2.0
    lots = 0.01
    
    # Find exit by scanning forward bars
    exit_px = None; exit_reason = 'open'
    for j in range(idx + 1, n):
        future_px_high = high[j]; future_px_low = low[j]; future_px_close = close[j]
        future_time = df.index[j]
        if future_time.strftime('%Y-%m-%d') != today: break  # Only same day
        
        if direction == 'BUY':
            if future_px_high >= tp: exit_px = tp; exit_reason = 'tp'; break
            if future_px_low <= sl: exit_px = sl; exit_reason = 'sl'; break
        else:
            if future_px_low <= tp: exit_px = tp; exit_reason = 'tp'; break
            if future_px_high >= sl: exit_px = sl; exit_reason = 'sl'; break
    
    if exit_px is not None:
        if direction == 'BUY': pnl = (exit_px - entry_px) * 100 * lots
        else: pnl = (entry_px - exit_px) * 100 * lots
        
        result = {
            'time': str(bar_time)[:16],
            'direction': direction,
            'type': buy_type if direction == 'BUY' else 'sell_3of5',
            'entry': round(entry_px, 2),
            'exit': round(exit_px, 2),
            'pnl': round(pnl, 2),
            'exit_reason': exit_reason,
            'approved': approved,
            'blocked_reason': blocked_reason,
            'rsi': round(rsi, 1),
            'adx': round(adx, 1),
        }
        
        if not approved:
            rejected_trades.append(result)
            total_rejected_pnl += pnl
        else:
            total_taken_pnl += pnl

# Print results
print(f"\n  Rejected trades today: {len(rejected_trades)}")
print(f"  Total PnL if ALL were taken: ${total_rejected_pnl + total_taken_pnl:+,.2f}")
print(f"  Rejected trades PnL (blocked by bot): ${total_rejected_pnl:+,.2f}")
print(f"  Approved trades PnL: ${total_taken_pnl:+,.2f}")

if rejected_trades:
    print(f"\n  --- DETAIL: Each rejected trade ---")
    for t in rejected_trades:
        verdict = "GOOD BLOCK" if t['pnl'] < 0 else "MISSED PROFIT"
        print(f"  {t['time']} | {t['direction']} {t['type']} | RSI={t['rsi']} ADX={t['adx']}")
        print(f"    Entry: ${t['entry']} Exit: ${t['exit']} ({t['exit_reason']}) PnL: ${t['pnl']:+,.2f}")
        print(f"    Blocked: {t['blocked_reason']} -> {verdict}")

print(f"\n  >>> VERDICT:")
if total_rejected_pnl < 0:
    print(f"  Bot CORRECTLY blocked losing trades (${total_rejected_pnl:+,.2f} avoided)")
elif total_rejected_pnl > 0:
    print(f"  Bot missed ${total_rejected_pnl:+,.2f} in profit by being too strict")
else:
    print(f"  No rejected trades had a clear outcome")

print("=" * 80)
