
"""
DEEP MACD ANALYSIS
Analyze MACD line/signal/histogram at entry vs outcome across 2024-2026.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "macd_deep.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  DEEP MACD ANALYSIS — all 2024-2026")
print("=" * 80)

if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2024,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close)
print(f"\nData: {len(df)} candles: {df.index[0]} to {df.index[-1]}")

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma50=sma(close,50)

# MACD (12,26,9)
ef=ema(close,12); es=ema(close,26); macd_line=ef-es; macd_signal=ema(macd_line,9); macd_hist=macd_line-macd_signal

print("Indicators computed.\n")

def collect_trades():
    bal=10000; trades=[]; pos=None
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma50[i]) or np.isnan(macd_line[i]): continue
        px=close[i]; atr=atr14[i]
        trend_up = px > sma50[i]
        
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl
                trades.append({'d':pos[0],'pnl':pnl,'macd_bull':pos[5],'macd_hist':pos[6],'hist_rising':pos[7]}); pos=None
            continue
        
        if trend_up:
            sl=px-atr*2.0; tp=px+atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            # macd_bull = line > signal; hist_rising = histogram increasing
            pos=['B',sl,tp,px,lots, macd_line[i]>macd_signal[i], macd_hist[i], macd_hist[i]>macd_hist[i-1] if i>0 else True]
        else:
            sl=px+atr*2.0; tp=px-atr*4.0
            lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
            pos=['S',sl,tp,px,lots, macd_line[i]>macd_signal[i], macd_hist[i], macd_hist[i]>macd_hist[i-1] if i>0 else True]
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'d':pos[0],'pnl':pnl,'macd_bull':pos[5],'macd_hist':pos[6],'hist_rising':pos[7]})
    return trades

trades = collect_trades()
print(f"Baseline: {len(trades)} trades, ${sum(t['pnl'] for t in trades):+,.0f}")

# ---- MACD bullish/bearish at entry ----
print(f"\n{'='*80}")
print("  MACD DIRECTION (line vs signal) at entry")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    bull = [t for t in dt if t['macd_bull']]
    bear = [t for t in dt if not t['macd_bull']]
    print(f"\n  {dlabel}:")
    for subset, sublabel in [(bull,'MACD bullish (line>signal)'),(bear,'MACD bearish (line<signal)')]:
        if len(subset) < 3: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100
        net = sum(t['pnl'] for t in subset)
        print(f"    {sublabel}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f}")

# ---- MACD histogram sign ----
print(f"\n{'='*80}")
print("  MACD HISTOGRAM (momentum) at entry")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    print(f"\n  {dlabel}:")
    for label, fn in [('Hist > 0 (positive momentum)', lambda t: t['macd_hist']>0),
                       ('Hist < 0 (negative momentum)', lambda t: t['macd_hist']<0)]:
        subset = [t for t in dt if fn(t)]
        if len(subset) < 3: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100
        net = sum(t['pnl'] for t in subset)
        print(f"    {label}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f}")

# ---- MACD histogram rising/falling ----
print(f"\n{'='*80}")
print("  MACD HISTOGRAM TREND (rising vs falling)")
print(f"{'='*80}")

for direction, dlabel in [('B','BUY'),('S','SELL')]:
    dt = [t for t in trades if t['d']==direction]
    rising = [t for t in dt if t['hist_rising']]
    falling = [t for t in dt if not t['hist_rising']]
    print(f"\n  {dlabel}:")
    for subset, sublabel in [(rising,'Hist rising'),(falling,'Hist falling')]:
        if len(subset) < 3: continue
        sw = [t for t in subset if t['pnl']>0]
        wr = len(sw)/len(subset)*100
        net = sum(t['pnl'] for t in subset)
        print(f"    {sublabel}: {len(subset)}t WR={wr:.0f}% PnL=${net:+,.0f}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
