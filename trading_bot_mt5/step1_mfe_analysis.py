"""
STEP 1: MFE Analysis — Diagnose Profit Giveback
================================================
Analyzes all closed trades to find where profits were given back.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUT_DIR, "mfe_analysis.txt"), 'w', encoding='utf-8')

from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

print("=" * 70)
print("  STEP 1: MFE ANALYSIS — Profit Giveback Diagnosis")
print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 70)

# Pull 3 months M15 data
print("\nLoading M15 data...")
if not mt5.initialize(): print("FATAL"); sys.exit(1)
to_dt = datetime.now(timezone.utc); from_dt = to_dt - timedelta(days=90)
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_dt, to_dt)
mt5.shutdown()
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df.columns = [c.lower() for c in df.columns]
close = df['close'].values.astype(float); high = df['high'].values.astype(float)
low = df['low'].values.astype(float)
print(f"Loaded {len(df)} candles: {df.index[0]} to {df.index[-1]}")

# Load extracted trades
TRADES_FILE = r'C:\Users\ASUS\extract_mt5_trades_output.json'
with open(TRADES_FILE) as f:
    trade_data = json.load(f)
trades = [t for t in trade_data['trades'] if t['volume'] > 0 and t['entry_price'] > 0 and t['exit_price'] is not None]
print(f"Loaded {len(trades)} closed trades\n")

# For each trade, trace MFE through price data
def ema(a,s): return pd.Series(a).ewm(span=s,adjust=False).mean().values
atr_vals = ema(np.maximum(np.maximum(high-low,np.abs(high-np.roll(close,1))),np.abs(low-np.roll(close,1))),14)

mfe_results = []
for trade in trades:
    try:
        et = datetime.fromisoformat(trade['entry_time']).replace(tzinfo=None)
        xt = datetime.fromisoformat(trade['exit_time']).replace(tzinfo=None) if trade['exit_time'] else None
        
        if xt is None: continue
        
        # Find data range
        mask = (df.index >= et) & (df.index <= xt)
        trade_bars = df[mask]
        if len(trade_bars) < 1: continue
        
        entry_px = trade['entry_price']
        exit_px = trade['exit_price']
        direction = trade['direction']
        atr_at_entry = float(atr_vals[df.index.get_indexer([et], method='nearest')[0]]) if len(df.index) > 0 else 5
        
        # Calculate MFE (max favorable excursion)
        if direction == 'BUY':
            peak_px = trade_bars['high'].max()
            mfe_points = peak_px - entry_px
            final_points = exit_px - entry_px
        else:
            peak_px = trade_bars['low'].min()
            mfe_points = entry_px - peak_px
            final_points = entry_px - exit_px
        
        lot_size = trade['volume']
        mfe_dollars = mfe_points * 100 * lot_size
        final_dollars = trade['profit']
        
        # How far did it go in ATR terms?
        mfe_in_atr = mfe_points / atr_at_entry if atr_at_entry > 0 else 0
        final_in_atr = final_points / atr_at_entry if atr_at_entry > 0 else 0
        
        # Did it reach 1x ATR, 1.5x ATR, 2x ATR?
        reached_1x = mfe_in_atr >= 1.0
        reached_1_5x = mfe_in_atr >= 1.5
        reached_2x = mfe_in_atr >= 2.0
        
        # Find time to peak
        if direction == 'BUY':
            peak_idx = trade_bars['high'].idxmax()
        else:
            peak_idx = trade_bars['low'].idxmin()
        time_to_peak = (peak_idx - et).total_seconds() / 60 if peak_idx and et else 0
        
        mfe_results.append({
            'ticket': trade['ticket'],
            'direction': direction,
            'entry': entry_px,
            'exit': exit_px,
            'peak': peak_px,
            'pnl': final_dollars,
            'mfe_points': round(mfe_points, 2),
            'mfe_dollars': round(mfe_dollars, 2),
            'mfe_atr': round(mfe_in_atr, 1),
            'final_atr': round(final_in_atr, 1),
            'reached_1x_atr': reached_1x,
            'reached_1_5x_atr': reached_1_5x,
            'reached_2x_atr': reached_2x,
            'time_to_peak_min': round(time_to_peak, 0),
            'exit_reason': trade['exit_reason'],
            'duration_min': trade['duration_minutes'],
        })
    except Exception as e:
        print(f"  Error on trade {trade.get('ticket','?')}: {e}")

# Analysis
print("=" * 70)
print("  MFE ANALYSIS RESULTS")
print("=" * 70)

total = len(mfe_results)
wins = [t for t in mfe_results if t['pnl'] > 0]
losses = [t for t in mfe_results if t['pnl'] <= 0]
sl_losses = [t for t in losses if t['exit_reason'] == 'sl']

print(f"\n  Total trades analyzed: {total}")
print(f"  Wins: {len(wins)}  |  Losses: {len(losses)}")

# Key finding: how many SL losses were profitable at peak?
sl_had_1x = [t for t in sl_losses if t['reached_1x_atr']]
sl_had_1_5x = [t for t in sl_losses if t['reached_1_5x_atr']]
sl_had_2x = [t for t in sl_losses if t['reached_2x_atr']]

print(f"\n  >>> STOP LOSS TRADES THAT WERE PROFITABLE AT PEAK:")
print(f"  SL trades: {len(sl_losses)}")
print(f"  Reached +1.0x ATR before SL: {len(sl_had_1x)}/{len(sl_losses)} ({len(sl_had_1x)/len(sl_losses)*100:.0f}%)" if sl_losses else "")
print(f"  Reached +1.5x ATR before SL: {len(sl_had_1_5x)}/{len(sl_losses)} ({len(sl_had_1_5x)/len(sl_losses)*100:.0f}%)" if sl_losses else "")
print(f"  Reached +2.0x ATR before SL: {len(sl_had_2x)}/{len(sl_losses)} ({len(sl_had_2x)/len(sl_losses)*100:.0f}%)" if sl_losses else "")

# Money left on table
sl_giveback = sum(t['mfe_dollars'] - t['pnl'] for t in sl_losses)
sl_mfe_total = sum(t['mfe_dollars'] for t in sl_losses)
print(f"\n  >>> MONEY LEFT ON TABLE (SL losses):")
print(f"  Total MFE reached: ${sl_mfe_total:,.2f}")
print(f"  Total actually lost: ${sum(t['pnl'] for t in sl_losses):,.2f}")
print(f"  Giveback: ${sl_giveback:,.2f}")

# All trades MFE vs final
all_mfe = sum(t['mfe_dollars'] for t in mfe_results)
all_final = sum(t['pnl'] for t in mfe_results)
print(f"\n  >>> ALL TRADES MFE vs FINAL:")
print(f"  Total MFE reached: ${all_mfe:,.2f}")
print(f"  Total PnL: ${all_final:,.2f}")
print(f"  Efficiency: {all_final/all_mfe*100:.1f}%" if all_mfe != 0 else "")

# Average MFE by outcome
print(f"\n  >>> AVERAGE MFE (in ATR):")
print(f"  Wins:  {sum(t['mfe_atr'] for t in wins)/len(wins):.1f}x ATR" if wins else "")
print(f"  Losses: {sum(t['mfe_atr'] for t in losses)/len(losses):.1f}x ATR" if losses else "")
print(f"  SL Losses: {sum(t['mfe_atr'] for t in sl_losses)/len(sl_losses):.1f}x ATR" if sl_losses else "")

# Individual SL givebacks
print(f"\n  >>> DETAIL: SL TRADES THAT REACHED +1.0x ATR:")
for t in sorted(sl_had_1x, key=lambda x: x['mfe_dollars'], reverse=True):
    print(f"  {t['direction']} Entry={t['entry']:.2f} Peak={t['peak']:.2f} Exit={t['exit']:.2f}")
    print(f"    MFE: ${t['mfe_dollars']:+,.2f} ({t['mfe_atr']}x ATR) -> Final: ${t['pnl']:+,.2f}")
    print(f"    Time to peak: {t['time_to_peak_min']:.0f}min | Duration: {t['duration_min']}min")

# Recommendation
print(f"\n  >>> RECOMMENDATION:")
if sl_losses and len(sl_had_1_5x)/len(sl_losses) > 0.3:
    print(f"  CRITICAL: {len(sl_had_1_5x)/len(sl_losses)*100:.0f}% of SL losses were >1.5x ATR profitable.")
    print(f"  ADD: Partial TP at 1.0x-1.5x ATR + move SL to breakeven")
if sl_losses and len(sl_had_1x)/len(sl_losses) > 0.5:
    print(f"  STRONG: {len(sl_had_1x)/len(sl_losses)*100:.0f}% of SL losses were >1.0x ATR profitable at peak.")
    print(f"  ADD: Trailing stop or structural momentum exit to capture gains")
    
print("\n" + "=" * 70)
print("  MFE ANALYSIS COMPLETE")
print("=" * 70)
