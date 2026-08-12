"""
PATTERN DISCOVERY — MACD (indicator #1, in isolation).
Reads the relationship between MACD and gold price over 3 years.
Finds NON-LINEAR PATTERNS (persistence, crossover, slope, extremes), not filters.
Reports forward-return hit rate and mean for each pattern, checked across 4 sub-periods
for consistency.
"""
import os, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(OUT, "gold_m15_3y.csv"))
close = df['close'].values.astype(float)
high = df['high'].values.astype(float)
low = df['low'].values.astype(float)
n = len(close)

def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values

# MACD
ef = ema(close,12); es = ema(close,26)
macd_line = ef - es
signal = ema(macd_line,9)
hist = macd_line - signal
hist_prev = np.roll(hist,1)
hist_prev[0] = hist[0]

# SMA50 for trend
sma50 = pd.Series(close).rolling(50).mean().values

# Sub-periods for consistency check
subs = [
    ("2023", 0, 23559),
    ("2024", 23559, 23559+23736),
    ("2025", 23559+23736, 23559+23736+23637),
    ("2026", 23559+23736+23637, n),
]

H = 12  # 3h forward
fwd = np.full(n, np.nan)
fwd[:n-H] = (close[H:] - close[:-H]) / close[:-H] * 100

def report(name, cond):
    """cond: bool array. Report hit rate & mean forward return, overall + per sub-period."""
    c = cond & ~np.isnan(fwd) & ~np.isnan(hist)
    idx = np.where(c)[0]
    if len(idx) < 50:
        print(f"  [{name}] too few samples ({len(idx)})")
        return
    overall_hr = (fwd[idx] > 0).mean()*100
    overall_mean = fwd[idx].mean()
    # consistency across sub-periods
    hrs = []
    for sn, s, e in subs:
        m = (idx >= s) & (idx < e)
        if m.sum() >= 20:
            hrs.append((fwd[idx[m]] > 0).mean()*100)
    consistent = (min(hrs) > 50) or (max(hrs) < 50) if hrs else False
    sign = "UP" if overall_mean > 0 else "DOWN"
    print(f"  [{name}] n={len(idx):>5} hit={overall_hr:>5.1f}% mean={overall_mean:+.3f}%  sub={[f'{h:.0f}' for h in hrs]}  {'CONSISTENT '+sign if consistent else ''}")

print("="*90)
print("  MACD PATTERN DISCOVERY (3 years M15, 3h forward)")
print("="*90)

# 1. Histogram sign states
report("hist > 0", hist > 0)
report("hist < 0", hist < 0)
report("hist > 0 & price>SMA50", (hist > 0) & (close > sma50))
report("hist < 0 & price<SMA50", (hist < 0) & (close < sma50))
report("hist > 0 & price<SMA50 (contrarian)", (hist > 0) & (close < sma50))
report("hist < 0 & price>SMA50 (contrarian)", (hist < 0) & (close > sma50))

# 2. Histogram rising/falling
rising = hist > hist_prev
falling = hist < hist_prev
report("hist rising", rising)
report("hist falling", falling)
report("hist rising & price>SMA50", rising & (close > sma50))
report("hist falling & price<SMA50", falling & (close < sma50))

# 3. Zero crossover
cross_up = (hist > 0) & (hist_prev <= 0)   # histogram crosses above zero
cross_dn = (hist < 0) & (hist_prev >= 0)   # histogram crosses below zero
report("hist crosses ABOVE zero (bullish cross)", cross_up)
report("hist crosses BELOW zero (bearish cross)", cross_dn)

# 4. Persistence (N consecutive bars same state)
for N in [2,3,5,8]:
    up_persist = np.ones(n, bool)
    dn_persist = np.ones(n, bool)
    for k in range(N):
        up_persist &= np.roll(hist > 0, k)
        dn_persist &= np.roll(hist < 0, k)
    up_persist[:N] = False; dn_persist[:N] = False
    report(f"hist>0 for {N}+ bars (persistence)", up_persist)
    report(f"hist<0 for {N}+ bars (persistence)", dn_persist)

# 5. Histogram magnitude extremes (percentile-based)
h_abs = np.abs(hist[~np.isnan(hist)])
p90 = np.percentile(h_abs, 90)
p95 = np.percentile(h_abs, 95)
report(f"hist extreme + (top 5% positive, >{np.percentile(hist,95):.1f})", hist > np.percentile(hist,95))
report(f"hist extreme - (bottom 5% negative, <{np.percentile(hist,5):.1f})", hist < np.percentile(hist,5))

# 6. MACD line vs signal (classic crossover already = hist sign flip, but check line slope)
macd_rising = macd_line > np.roll(macd_line,1)
macd_falling = macd_line < np.roll(macd_line,1)
report("macd line rising", macd_rising)
report("macd line falling", macd_falling)

print("\nDONE")
