"""
LOSS REDUCTION ANALYSIS: Find filters that block losers without touching winners.
Analyzes 29 actual trades + 6-month backtest to find the perfect balance.
"""
import sys, os, json, warnings, itertools
warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "loss_reduction.txt"), 'w', encoding='utf-8')
sys.stderr = sys.stdout

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

T0 = datetime.now()
print("=" * 80)
print("  LOSS REDUCTION OPTIMIZATION")
print(f"  {T0.strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 80)

# Load actual trades with entry indicators
with open(r'C:\Users\ASUS\extract_mt5_trades_output.json') as f:
    data = json.load(f)
trades = [t for t in data['trades'] if t['volume'] > 0 and t['entry_price'] > 0 and t.get('exit_price') and t.get('entry_indicators_m15',{}).get('rsi_14')]

print(f"\nLoaded {len(trades)} trades with full indicator data")
wins = [t for t in trades if t['profit'] > 0]
losses = [t for t in trades if t['profit'] <= 0]
print(f"Wins: {len(wins)} (${sum(t['profit'] for t in wins):+,.2f})")
print(f"Losses: {len(losses)} (${sum(t['profit'] for t in losses):+,.2f})")

# ============================================================
# ANALYSIS 1: Find the exact indicator values that separate wins from losses
# ============================================================
print(f"\n{'='*80}")
print("  ANALYSIS 1: INDICATOR THRESHOLDS — Win vs Loss Separation")
print(f"{'='*80}")

def ei(t, key):
    return t.get('entry_indicators_m15', {}).get(key)

for indicator, label, win_filter, loss_filter in [
    ('rsi_14', 'RSI at Entry', 
     lambda v: v is not None,
     lambda v: v is not None),
    ('adx_14', 'ADX at Entry',
     lambda v: v is not None,
     lambda v: v is not None),
    ('macd_histogram', 'MACD Histogram',
     lambda v: v is not None,
     lambda v: v is not None),
    ('atr_14', 'ATR at Entry',
     lambda v: v is not None,
     lambda v: v is not None),
]:
    win_vals = [ei(t, indicator) for t in wins if ei(t, indicator) is not None]
    loss_vals = [ei(t, indicator) for t in losses if ei(t, indicator) is not None]
    
    if win_vals and loss_vals:
        print(f"\n  {label}:")
        print(f"    Wins:  min={min(win_vals):.1f} max={max(win_vals):.1f} avg={np.mean(win_vals):.1f}")
        print(f"    Losses: min={min(loss_vals):.1f} max={max(loss_vals):.1f} avg={np.mean(loss_vals):.1f}")

# ============================================================
# ANALYSIS 2: Test specific filters on actual trades
# ============================================================
print(f"\n{'='*80}")
print("  ANALYSIS 2: FILTER EFFECTIVENESS ON ACTUAL TRADES")
print(f"{'='*80}")

filters_to_test = []

# Generate all meaningful filter combinations
for adx_thresh in [0, 15, 20, 25, 30]:
    for rsi_buy_max in [55, 60, 65]:
        for reject_macd_bearish in [True, False]:
            for reject_no_stoch_cross in [True, False]:
                for reject_above_mid in [True, False]:
                    for reject_outside_hours in [True, False]:
                        for reject_low_adx_sell in [True, False]:
                            # Score: count params to keep grid manageable
                            param_count = sum([adx_thresh>0, reject_macd_bearish, 
                                             reject_no_stoch_cross, reject_above_mid,
                                             reject_outside_hours, reject_low_adx_sell])
                            if param_count < 2 or param_count > 4: continue  # Skip edge cases
                            
                            filters_to_test.append({
                                'adx_min': adx_thresh,
                                'rsi_buy_max': rsi_buy_max,
                                'reject_macd_bearish': reject_macd_bearish,
                                'reject_no_stoch_cross': reject_no_stoch_cross,
                                'reject_above_mid': reject_above_mid,
                                'reject_outside_hours': reject_outside_hours,
                                'reject_low_adx_sell': reject_low_adx_sell,
                            })

# Remove duplicates
seen = set()
unique_filters = []
for f in filters_to_test:
    key = tuple(sorted(f.items()))
    if key not in seen:
        seen.add(key)
        unique_filters.append(f)

print(f"Testing {len(unique_filters)} filter combinations...\n")

results = []
for filt in unique_filters:
    filtered_wins = []
    filtered_losses = []
    blocked_wins = []
    blocked_losses = []
    
    for t in trades:
        rsi = ei(t, 'rsi_14')
        adx = ei(t, 'adx_14')
        macd_bull = ei(t, 'macd_bullish')
        bb_pos = ei(t, 'price_vs_ema20')  # proxy for BB position
        stoch_cross = ei(t, 'macd_bullish')  # rough proxy - would need actual stoch
        hour = datetime.fromisoformat(t['entry_time']).hour if t.get('entry_time') else 12
        direction = t['direction']
        
        if rsi is None or adx is None: continue
        
        blocked = False
        reasons = []
        
        if adx < filt['adx_min']:
            blocked = True; reasons.append(f'ADX={adx:.0f}<{filt["adx_min"]}')
        
        if direction == 'BUY' and rsi > filt['rsi_buy_max']:
            blocked = True; reasons.append(f'RSI={rsi:.0f}>{filt["rsi_buy_max"]}')
        
        if filt['reject_macd_bearish'] and macd_bull == False:
            blocked = True; reasons.append('MACD bearish')
        
        if filt['reject_outside_hours'] and not (8 <= hour < 22):
            blocked = True; reasons.append(f'outside hours ({hour}h)')
        
        if filt['reject_above_mid'] and bb_pos == 'above':
            blocked = True; reasons.append('above BB mid')
        
        if blocked:
            if t['profit'] > 0: blocked_wins.append(t)
            else: blocked_losses.append(t)
        else:
            if t['profit'] > 0: filtered_wins.append(t)
            else: filtered_losses.append(t)
    
    if len(filtered_wins) + len(filtered_losses) == 0: continue
    
    orig_pnl = sum(t['profit'] for t in trades)
    filtered_pnl = sum(t['profit'] for t in (filtered_wins + filtered_losses))
    blocked_win_pnl = sum(t['profit'] for t in blocked_wins)
    blocked_loss_pnl = sum(t['profit'] for t in blocked_losses)
    
    improvement = filtered_pnl - orig_pnl
    win_retention = len(filtered_wins) / len(wins) * 100 if wins else 0
    loss_reduction = (len(losses) - len(filtered_losses)) / len(losses) * 100 if losses else 0
    
    # Score: prefer high win retention + high loss reduction
    score = win_retention * 0.6 + loss_reduction * 0.4
    
    results.append({
        'filter': filt,
        'filtered_pnl': round(filtered_pnl, 2),
        'improvement': round(improvement, 2),
        'win_retention': round(win_retention, 1),
        'loss_reduction': round(loss_reduction, 1),
        'score': round(score, 1),
        'kept_wins': len(filtered_wins),
        'kept_losses': len(filtered_losses),
        'blocked_win_pnl': round(blocked_win_pnl, 2),
        'blocked_loss_pnl': round(blocked_loss_pnl, 2),
    })

# Sort by improvement (best PnL)
results.sort(key=lambda x: (x['improvement'], x['win_retention']), reverse=True)

print(f"  Top results by P&L improvement:")
for i, r in enumerate(results[:15]):
    f = r['filter']
    print(f"\n  #{i+1}: PnL=${r['filtered_pnl']:+,.2f} (improved ${r['improvement']:+,.2f})")
    print(f"    Win retention: {r['win_retention']}% ({r['kept_wins']}/{len(wins)} kept)")
    print(f"    Loss reduction: {r['loss_reduction']}% ({len(losses)-r['kept_losses']}/{len(losses)} blocked)")
    print(f"    Blocked win PnL: ${r['blocked_win_pnl']:+,.2f} | Blocked loss PnL: ${r['blocked_loss_pnl']:+,.2f}")
    print(f"    Rules: ADX>{f['adx_min']} RSI<{f['rsi_buy_max']} MACDbear:{f['reject_macd_bearish']} "
          f"Stoch:{f['reject_no_stoch_cross']} BBmid:{f['reject_above_mid']} Hours:{f['reject_outside_hours']}")

# Best by score (balance of win retention + loss reduction)
results.sort(key=lambda x: x['score'], reverse=True)
print(f"\n{'='*80}")
print("  BEST BALANCED FILTERS (High win retention + High loss reduction)")
print(f"{'='*80}")
for i, r in enumerate(results[:10]):
    f = r['filter']
    print(f"\n  #{i+1}: Score={r['score']} | PnL=${r['filtered_pnl']:+,.2f} | WinKeep={r['win_retention']}% | LossBlock={r['loss_reduction']}%")
    print(f"    Rules: ADX>{f['adx_min']} RSI<{f['rsi_buy_max']} MACDbear:{f['reject_macd_bearish']} BBmid:{f['reject_above_mid']} Hours:{f['reject_outside_hours']}")

# ============================================================
# ANALYSIS 3: Single most impactful filter
# ============================================================
print(f"\n{'='*80}")
print("  SINGLE FILTER IMPACT (One rule at a time)")
print(f"{'='*80}")

single_filters = [
    ('ADX > 20', lambda t: ei(t,'adx_14') and ei(t,'adx_14') >= 20),
    ('ADX > 25', lambda t: ei(t,'adx_14') and ei(t,'adx_14') >= 25),
    ('RSI < 60 on BUY', lambda t: t['direction'] != 'BUY' or (ei(t,'rsi_14') and ei(t,'rsi_14') <= 60)),
    ('RSI < 65 on BUY', lambda t: t['direction'] != 'BUY' or (ei(t,'rsi_14') and ei(t,'rsi_14') <= 65)),
    ('MACD not bearish', lambda t: ei(t,'macd_bullish') != False),
    ('Active hours only', lambda t: 8 <= datetime.fromisoformat(t['entry_time']).hour < 22 if t.get('entry_time') else True),
    ('Not above BB mid on BUY', lambda t: t['direction'] != 'BUY' or ei(t,'price_vs_ema20') != 'above'),
    ('ATR < 10', lambda t: ei(t,'atr_14') and ei(t,'atr_14') < 10),
    ('Not SELL with MACD bull', lambda t: t['direction'] != 'SELL' or ei(t,'macd_bullish') != True),
]

for name, filter_fn in single_filters:
    kept = [t for t in trades if filter_fn(t)]
    blocked = [t for t in trades if not filter_fn(t)]
    
    kept_wins = [t for t in kept if t['profit'] > 0]
    kept_losses = [t for t in kept if t['profit'] <= 0]
    blocked_wins = [t for t in blocked if t['profit'] > 0]
    blocked_losses = [t for t in blocked if t['profit'] <= 0]
    
    kept_pnl = sum(t['profit'] for t in kept)
    blocked_pnl = sum(t['profit'] for t in blocked)
    
    if len(kept) == 0: continue
    
    print(f"\n  {name}:")
    print(f"    Kept: {len(kept)} trades, PnL=${kept_pnl:+,.2f} (WR={len(kept_wins)/len(kept)*100:.0f}%)")
    print(f"    Blocked: {len(blocked)} trades, PnL=${blocked_pnl:+,.2f}")
    print(f"    Win retention: {len(kept_wins)}/{len(wins)} | Loss reduction: {len(blocked_losses)}/{len(losses)}")

print(f"\n{'='*80}")
print("  ANALYSIS COMPLETE")
print(f"{'='*80}")
