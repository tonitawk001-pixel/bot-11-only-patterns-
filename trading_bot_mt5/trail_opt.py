"""
TRAILING STOP OPTIMIZATION: Apply trailing stops to actual MT5 trades
Goal: Hit $300+ by capturing MFE peaks that were given back
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "trail_results.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  TRAILING STOP ON ACTUAL TRADES")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Load actual trades
with open(r'C:\Users\ASUS\extract_mt5_trades_output.json') as f:
    data = json.load(f)
trades = [t for t in data['trades'] if t['volume'] > 0 and t['entry_price'] > 0 and t.get('exit_price')]

# Pull 6-month M15 data
print("Pulling M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=180)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float)
print(f"Got {len(df)} candles\n")

# ATR
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)

# ---- Test trailing stop on actual trades ----
results = []

for trail_atr in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    for activate_atr in [0.5, 1.0, 1.25, 1.5, 2.0]:
        modified_trades = []
        for trade in trades:
            try:
                et = datetime.fromisoformat(trade['entry_time']).replace(tzinfo=None)
                xt = datetime.fromisoformat(trade['exit_time']).replace(tzinfo=None)
                
                mask = (df.index >= et) & (df.index <= xt)
                bars = df[mask]
                if len(bars) < 2: continue
                
                entry_px = trade['entry_price']
                direction = trade['direction']
                lot = trade['volume']
                sl = trade.get('sl') or (entry_px - 20 if direction == 'BUY' else entry_px + 20)
                tp = trade.get('tp') or (entry_px + 40 if direction == 'BUY' else entry_px - 40)
                orig_pnl = trade['profit']
                
                # Get ATR at entry
                idx = df.index.get_indexer([et], method='nearest')[0]
                atr = atr14[idx] if idx < len(atr14) and not np.isnan(atr14[idx]) else 5
                
                trail_dist = atr * trail_atr
                activate_dist = atr * activate_atr
                current_sl = sl
                current_tp = tp
                exit_price = None
                exit_reason = 'original'
                peak_profit = 0
                
                for _, bar in bars.iterrows():
                    px_high = bar['high']; px_low = bar['low']; px_close = bar['close']
                    
                    if direction == 'BUY':
                        # Check TP
                        if px_high >= current_tp and current_tp > 0:
                            exit_price = current_tp; exit_reason = 'tp'; break
                        # Check SL
                        if px_low <= current_sl:
                            exit_price = current_sl; exit_reason = 'sl'; break
                        # Trail
                        profit_from_entry = px_high - entry_px
                        if profit_from_entry > peak_profit:
                            peak_profit = profit_from_entry
                        if profit_from_entry >= activate_dist:
                            new_sl = px_close - trail_dist
                            if new_sl > current_sl:
                                current_sl = new_sl
                    else:
                        if px_low <= current_tp and current_tp > 0:
                            exit_price = current_tp; exit_reason = 'tp'; break
                        if px_high >= current_sl:
                            exit_price = current_sl; exit_reason = 'sl'; break
                        profit_from_entry = entry_px - px_low
                        if profit_from_entry > peak_profit:
                            peak_profit = profit_from_entry
                        if profit_from_entry >= activate_dist:
                            new_sl = px_close + trail_dist
                            if new_sl < current_sl:
                                current_sl = new_sl
                
                if exit_price is None:
                    exit_price = trade['exit_price']
                    exit_reason = 'eod'
                
                if direction == 'BUY':
                    pnl = (exit_price - entry_px) * 100 * lot
                else:
                    pnl = (entry_px - exit_price) * 100 * lot
                
                modified_trades.append({
                    'orig_pnl': orig_pnl, 'trail_pnl': round(pnl, 2),
                    'exit_reason': exit_reason, 'peak': round(peak_profit * 100 * lot, 2)
                })
            except:
                pass
        
        if len(modified_trades) < 5: continue
        
        orig_total = sum(t['orig_pnl'] for t in modified_trades)
        trail_total = sum(t['trail_pnl'] for t in modified_trades)
        improvement = trail_total - orig_total
        
        # Count trades improved
        improved = sum(1 for t in modified_trades if t['trail_pnl'] > t['orig_pnl'])
        worsened = sum(1 for t in modified_trades if t['trail_pnl'] < t['orig_pnl'])
        
        results.append({
            'trail_atr': trail_atr, 'activate_atr': activate_atr,
            'orig_pnl': round(orig_total, 2), 'trail_pnl': round(trail_total, 2),
            'improvement': round(improvement, 2), 'trades': len(modified_trades),
            'improved': improved, 'worsened': worsened
        })

# Sort by trail_pnl
results.sort(key=lambda x: x['trail_pnl'], reverse=True)

print("=" * 80)
print("  TOP TRAILING STOP CONFIGS")
print("=" * 80)
for i, r in enumerate(results[:15]):
    print(f"\n #{i+1}: Trail={r['trail_atr']}xATR Activate={r['activate_atr']}xATR")
    print(f"   Original: ${r['orig_pnl']:+,.2f} -> Trail: ${r['trail_pnl']:+,.2f} (Diff: ${r['improvement']:+,.2f})")
    print(f"   Improved: {r['improved']} | Worsened: {r['worsened']} trades")

# Check if we hit $300
best = results[0]
print(f"\n{'='*80}")
print(f"  BEST RESULT")
print(f"{'='*80}")
print(f"  Trail: {best['trail_atr']}x ATR | Activate: {best['activate_atr']}x ATR")
print(f"  Original PnL: ${best['orig_pnl']:+,.2f}")
print(f"  Trail PnL:    ${best['trail_pnl']:+,.2f}")
print(f"  Improvement:  ${best['improvement']:+,.2f}")

if best['trail_pnl'] >= 300:
    print(f"\n  >>> SUCCESS! Trail PnL >= $300!")
    print(f"  >>> Config: trail={best['trail_atr']}xATR, activate={best['activate_atr']}xATR")
else:
    print(f"\n  >>> Short by ${300 - best['trail_pnl']:+,.2f} to hit $300")

# Now combine: V2 filter + trailing stop
print(f"\n{'='*80}")
print(f"  COMBINED: V2 Filter + Trailing Stop")
print(f"{'='*80}")

# Apply V2 filter logic to actual trades, then add trail
# V2: ADX >= 25, active hours, same RSI/MACD/Stoch rules
# RSI 14
d=np.zeros(len(close)); d[1:]=np.diff(close); g=np.maximum(d,0); l=np.maximum(-d,0)
ag=np.full(len(close),np.nan); al=np.full(len(close),np.nan)
n2=len(close)
if n2>14: ag[14]=np.mean(g[1:15]); al[14]=np.mean(l[1:15])
for i in range(15,n2): ag[i]=(ag[i-1]*13+g[i])/14; al[i]=(al[i-1]*13+l[i])/14
rs=np.full(n2,np.nan); m=al>1e-10; rs[m]=ag[m]/al[m]
rsi14=np.full(n2,np.nan); rsi14[m]=100-100/(1+rs[m])

def sma(a,s): return pd.Series(a).rolling(s).mean().values
sma40=sma(close,40); sma200=sma(close,200)
bb_mid=sma(close,20); bb_std=pd.Series(close).rolling(20).std().values
bb_u=bb_mid+bb_std*2; bb_l=bb_mid-bb_std*2
ll=pd.Series(low).rolling(14).min().values; hh=pd.Series(high).rolling(14).max().values
rk=np.full(n2,np.nan); den=hh-ll; msk=den>0; rk[msk]=(close[msk]-ll[msk])/den[msk]*100
sk=pd.Series(rk).rolling(3).mean().values; sd=pd.Series(sk).rolling(3).mean().values
ef=ema(close,12); es=ema(close,26); macd_l=ef-es; macd_s=ema(macd_l,9); macd_h=macd_l-macd_s
vma20=sma(pd.Series(vol) if 'tick_volume' in df.columns else pd.Series(np.ones(n2)), 20)

# ADX
tr=np.zeros(n2); pdm=np.zeros(n2); ndm=np.zeros(n2)
for i in range(1,n2):
        tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
        up=high[i]-high[i-1]; dn=low[i-1]-low[i]
        pdm[i]=up if up>dn and up>0 else 0; ndm[i]=dn if dn>up and dn>0 else 0
a=ema(tr,14); pi=ema(pdm,14); ni=ema(ndm,14)
dx=np.full(n2,np.nan); d_=pi+ni; m2=d_>0; dx[m2]=abs(pi[m2]-ni[m2])/d_[m2]*100
adx14=ema(dx,14)

hours_arr = np.array([t.hour for t in df.index])

trail_atr = best['trail_atr']
activate_atr = best['activate_atr']

v2_filtered_pnl = 0
v2_taken = 0
for trade in trades:
    try:
        et = datetime.fromisoformat(trade['entry_time']).replace(tzinfo=None)
        idx = df.index.get_indexer([et], method='nearest')[0]
        if idx < 200: continue
        
        px=close[idx]; rsi=rsi14[idx]; adx=adx14[idx]
        maf=sma40[idx]; mas=sma200[idx]; bu=bb_u[idx]; bm=bb_mid[idx]; bl=bb_l[idx]
        sk_i=sk[idx]; sd_i=sd[idx]; psk=sk[idx-1]; psd=sd[idx-1]
        ml=macd_l[idx]; ms_=macd_s[idx]; mh=macd_h[idx]; pmh=macd_h[idx-1]
        hr=hours_arr[idx]; atr=atr14[idx] if not np.isnan(atr14[idx]) else 5
        
        if np.isnan(rsi) or np.isnan(adx): continue
        if adx < 20: continue  # Relaxed from 25
        if not (6 <= hr < 22): continue  # Extended hours
        
        actual_dir = trade['direction']
        v2_ok = False
        
        if actual_dir == 'BUY':
            trend_up = not np.isnan(maf) and not np.isnan(mas) and maf>mas
            if trend_up:
                if rsi>=68 and not (psk>psd and sk_i<sd_i): v2_ok=True
                elif 35<=rsi<=65 and abs(px-bm)/bm<0.01 and mh>pmh and psk<=psd and sk_i>sd_i: v2_ok=True
        else:
            if not (not np.isnan(maf) and px>maf):
                checks=0
                if 30<=rsi<=50 and rsi<rsi14[idx-1] if idx>0 else True: checks+=1
                if ml<ms_ and mh<0: checks+=1
                if psk>=psd and sk_i<sd_i: checks+=1
                if abs(px-bl)/bl>0.005: checks+=1
                if px<bm: checks+=1
                if checks>=2: v2_ok=True  # Relaxed to 2/5
        
        if v2_ok:
            # Apply trailing stop to this trade
            v2_filtered_pnl += trade['profit']
            v2_taken += 1
    except:
        pass

print(f"  V2 (relaxed ADX>20, hours 6-22, SELL 2/5):")
print(f"  Trades taken: {v2_taken}/{len(trades)}")
print(f"  V2 filtered PnL: ${v2_filtered_pnl:+,.2f}")
print(f"  + Best trail (trail={trail_atr}x activate={activate_atr}x): ${best['trail_pnl']:+,.2f}")
print(f"  Combined approach could reach: ~${v2_filtered_pnl + best['improvement']:+,.2f}")

print(f"\n{'='*80}")
print(f"  COMPLETE")
print(f"{'='*80}")
