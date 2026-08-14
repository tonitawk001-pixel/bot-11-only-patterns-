"""Edge-case validation of the ADX regime + extreme-trend filter logic."""
import numpy as np

def adx_filter(m15_adx, raw_direction):
    """Replicates the ADX logic from main_super.py (after fix)."""
    blocked_by = None
    # strong_trend
    strong_trend = (not np.isnan(m15_adx)) and m15_adx >= 35
    if strong_trend:
        pass  # aggressive mode
    else:
        if not np.isnan(m15_adx) and m15_adx < 20:
            blocked_by = f"ADX_{m15_adx:.0f}_lt_20"
            raw_direction = "NONE"
    # extreme trend block (>= 50)
    if raw_direction != "NONE" and not np.isnan(m15_adx) and m15_adx >= 50:
        blocked_by = f"extreme_trend_adx_{m15_adx:.0f}_ge_50"
        raw_direction = "NONE"
    return strong_trend, raw_direction, blocked_by

cases = [
    ("NaN ADX (unavailable)", float('nan'), "BUY"),
    ("ADX=15 (chop)", 15.0, "BUY"),
    ("ADX=30 (mild)", 30.0, "BUY"),
    ("ADX=40 (trend)", 40.0, "BUY"),
    ("ADX=55 (extreme)", 55.0, "BUY"),
    ("ADX=20 (boundary)", 20.0, "BUY"),
    ("ADX=50 (boundary)", 50.0, "BUY"),
    ("ADX=99 (old sentinel)", 99.0, "BUY"),
]

print(f"{'case':<24} {'strong_trend':<13} {'direction':<8} {'blocked_by'}")
print("-"*70)
for name, adx, d in cases:
    st, direction, blocked = adx_filter(adx, d)
    print(f"{name:<24} {str(st):<13} {direction:<8} {blocked}")

print("\nExpected correct behavior:")
print("  - NaN/99 unavailable -> no block (skip ADX filtering), strong_trend False for NaN")
print("  - ADX<20 -> chop block")
print("  - 20<=ADX<50 -> no block")
print("  - ADX>=50 -> extreme trend block")
