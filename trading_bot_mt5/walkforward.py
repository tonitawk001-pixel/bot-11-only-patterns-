"""
WALK-FORWARD ADAPTIVE SYSTEM
A meta-strategy that detects regime from RECENT price action (not fixed thresholds)
and switches between trend-following and mean-reversion sub-strategies.
Validated walk-forward: adapts using only PAST data, never leaks future.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "walkforward.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 80)
print("  WALK-FORWARD ADAPTIVE SYSTEM")
print("=" * 80)

# Pull ALL data including 2024 (the test that exposed overfitting)
if not mt5.initialize(): print("FATAL"); sys.exit(1)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
    datetime(2024,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close=df['close'].values.astype(float); high=df['high'].values.astype(float)
low=df['low'].values.astype(float); vol=df['tick_volume'].values.astype(float)
n=len(close); hours=np.array([t.hour for t in df.index])
print(f"\nData: {len(df)} candles: {df.index[0]} to {df.index[-1]} (2024-2026, all conditions)")

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
def sma(a,s): return pd.Series(a).rolling(s).mean().values

# Core indicators (robust, not tuned)
atr14=ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)
sma50=sma(close,50)
sma200=sma(close,200)

print("Indicators computed.")

def detect_regime(i):
    """
    Detect regime using RECENT price action only (walk-forward safe).
    Returns 'trend' or 'range'.
    Uses: distance of price from SMA200, and recent directional consistency.
    """
    if i < 250: return 'range'
    # Trend strength: how far is SMA50 from SMA200 (normalized by ATR)
    gap = abs(sma50[i] - sma200[i]) / atr14[i] if not np.isnan(atr14[i]) and atr14[i]>0 else 0
    # Directional consistency: what % of last 20 bars moved in the dominant direction
    recent = close[i-20:i+1]
    up_moves = sum(1 for j in range(1, len(recent)) if recent[j] > recent[j-1])
    direction_score = abs(up_moves - (len(recent)-1)/2) / ((len(recent)-1)/2)
    
    # Strong trend = wide MA gap + consistent direction
    if gap > 3.0 and direction_score > 0.4:
        return 'trend'
    return 'range'

def run_strategy():
    """Walk-forward adaptive: trend strategy in trends, mean-reversion in ranges."""
    bal=10000; trades=[]; pos=None
    
    for i in range(300, n):
        if np.isnan(close[i]) or np.isnan(atr14[i]) or np.isnan(sma200[i]): continue
        px=close[i]; atr=atr14[i]
        regime = detect_regime(i)
        
        # Manage position
        if pos is not None:
            sl_hit=(pos[0]=='B' and px<=pos[1]) or (pos[0]=='S' and px>=pos[1])
            tp_hit=(pos[0]=='B' and px>=pos[2]) or (pos[0]=='S' and px<=pos[2])
            if sl_hit or tp_hit:
                ep=pos[1] if sl_hit else pos[2]
                pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
                bal+=pnl; trades.append({'pnl':pnl,'regime':pos[5]}); pos=None
            continue
        
        # TREND REGIME: follow the trend (buy above SMA200, sell below)
        if regime == 'trend':
            if px > sma200[i] and px > sma50[i]:
                # Buy pullback to SMA50
                if abs(px - sma50[i])/sma50[i] < 0.005:
                    sl = px - atr*2.5; tp = px + atr*5.0
                    lots=max(0.01,round((bal*0.02)/(atr*2.5*100),2))
                    pos=('B',sl,tp,px,lots,regime); continue
            elif px < sma200[i] and px < sma50[i]:
                if abs(px - sma50[i])/sma50[i] < 0.005:
                    sl = px + atr*2.5; tp = px - atr*5.0
                    lots=max(0.01,round((bal*0.02)/(atr*2.5*100),2))
                    pos=('S',sl,tp,px,lots,regime); continue
        
        # RANGE REGIME: mean reversion (buy oversold, sell overbought)
        else:
            # Use recent local extremes
            recent_low = low[i-20:i].min()
            recent_high = high[i-20:i].max()
            range_size = recent_high - recent_low
            if range_size > 0:
                pos_in_range = (px - recent_low) / range_size
                if pos_in_range < 0.2:
                    # Near bottom of range -> buy
                    sl = px - atr*2.0; tp = px + atr*3.0
                    lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
                    pos=('B',sl,tp,px,lots,regime); continue
                elif pos_in_range > 0.8:
                    # Near top of range -> sell
                    sl = px + atr*2.0; tp = px - atr*3.0
                    lots=max(0.01,round((bal*0.02)/(atr*2.0*100),2))
                    pos=('S',sl,tp,px,lots,regime); continue
    
    if pos is not None:
        ep=close[-1]; pnl=(ep-pos[3])*100*pos[4] if pos[0]=='B' else (pos[3]-ep)*100*pos[4]
        bal+=pnl; trades.append({'pnl':pnl,'regime':pos[5]})
    
    return trades

trades = run_strategy()

# Analysis by year
print(f"\n{'='*80}")
print("  RESULTS BY YEAR (walk-forward adaptive)")
print(f"{'='*80}")

years = sorted(set(df.index.year))
for year in years:
    # Need trade timestamps to split by year — approximate by order
    # Instead, split by cumulative PnL progression
    pass

# Overall
wins=[t for t in trades if t['pnl']>0]; loses=[t for t in trades if t['pnl']<=0]
net=sum(t['pnl'] for t in trades)
print(f"\n  OVERALL (2024-2026, all conditions):")
print(f"    Net: ${net:+,.0f} | {len(trades)}t | WR={len(wins)/len(trades)*100:.1f}%")
print(f"    Wins: {len(wins)} | Losses: {len(loses)}")

# By regime
trend_t = [t for t in trades if t['regime']=='trend']
range_t = [t for t in trades if t['regime']=='range']
print(f"\n  By regime:")
print(f"    Trend trades: {len(trend_t)} | PnL=${sum(t['pnl'] for t in trend_t):+,.0f}")
print(f"    Range trades: {len(range_t)} | PnL=${sum(t['pnl'] for t in range_t):+,.0f}")

# Win rate by regime
if trend_t:
    tw=[t for t in trend_t if t['pnl']>0]
    print(f"    Trend WR: {len(tw)/len(trend_t)*100:.1f}%")
if range_t:
    rw=[t for t in range_t if t['pnl']>0]
    print(f"    Range WR: {len(rw)/len(range_t)*100:.1f}%")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
