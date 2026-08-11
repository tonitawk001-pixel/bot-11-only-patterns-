"""Analyze if BE-at-+$30 would have improved the 10 live trades"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

mt5.initialize()

# The 10 live trades from extract_mt5_data.py
trades = [
    {"entry": "2026-07-28 23:52", "dir": "BUY", "lot": 0.04, "price": 4024.01, "exit_price": 4024.21, "pnl": 0.80, "type": "BE/trail"},
    {"entry": "2026-07-29 01:32", "dir": "BUY", "lot": 0.03, "price": 4021.89, "exit_price": 4022.09, "pnl": 0.60, "type": "BE/trail"},
    {"entry": "2026-07-29 06:24", "dir": "BUY", "lot": 0.03, "price": 4041.79, "exit_price": 4027.32, "pnl": -43.41, "type": "SL"},
    {"entry": "2026-07-29 16:50", "dir": "BUY", "lot": 0.03, "price": 4019.38, "exit_price": 4019.58, "pnl": 0.60, "type": "BE/trail"},
    {"entry": "2026-07-29 17:13", "dir": "BUY", "lot": 0.02, "price": 4046.46, "exit_price": 4027.49, "pnl": -37.94, "type": "SL"},
    {"entry": "2026-07-29 18:00", "dir": "BUY", "lot": 0.01, "price": 4069.67, "exit_price": 4045.29, "pnl": -24.38, "type": "SL"},
    {"entry": "2026-07-29 19:33", "dir": "SELL", "lot": 0.01, "price": 4072.59, "exit_price": 4072.39, "pnl": 0.20, "type": "BE/trail"},
    {"entry": "2026-07-31 07:30", "dir": "SELL", "lot": 0.01, "price": 4071.71, "exit_price": 4071.51, "pnl": 0.20, "type": "BE/trail"},
    {"entry": "2026-07-31 13:00", "dir": "BUY", "lot": 0.02, "price": 4038.35, "exit_price": 4072.05, "pnl": 67.40, "type": "TP"},
    {"entry": "2026-07-31 14:35", "dir": "BUY", "lot": 0.03, "price": 4041.13, "exit_price": 4041.33, "pnl": 0.60, "type": "BE/trail"},
]

print("=" * 70)
print("BE-AT-$30 SIMULATION ON 10 LIVE TRADES")
print("=" * 70)

sl_mult = 2.5
be_target_profit_usd = 30.0  # Move SL to entry after $30 profit
be_buffer_points = 50  # SL at entry + 50 points

results = []
for i, t in enumerate(trades):
    entry_dt = datetime.strptime(t["entry"], "%Y-%m-%d %H:%M")  # tz-naive, matches MT5 data
    entry_price = t["price"]
    direction = t["dir"]
    lots = t["lot"]
    exit_price = t["exit_price"]
    exit_time_str = t.get("exit_time", "unknown")
    
    # Get M5 data starting from entry time (2 hours of data)
    start = entry_dt
    end = entry_dt + timedelta(hours=4)
    
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, start, end)
    if rates is None or len(rates) == 0:
        print(f"  #{i+1}: {direction} @ {entry_price:.2f} — NO M5 DATA")
        continue
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[df.index >= pd.Timestamp(entry_dt)]
    
    if len(df) == 0:
        print(f"  #{i+1}: {direction} @ {entry_price:.2f} — NO POST-ENTRY DATA")
        continue
    
    # Simulate ATR at entry (approximate from available data before entry)
    pre_rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, entry_dt - timedelta(hours=2), entry_dt)
    if pre_rates is not None and len(pre_rates) >= 10:
        pre_df = pd.DataFrame(pre_rates)
        atr_approx = (pre_df['high'].astype(float) - pre_df['low'].astype(float)).tail(20).mean()
    else:
        atr_approx = 3.0  # fallback
    
    # SL distance
    sl_distance = atr_approx * sl_mult
    if direction == "BUY":
        sl_price = entry_price - sl_distance
        be_price = entry_price + be_buffer_points * 0.01
        # +$30 profit in points for BUY
        target_profit_points = be_target_profit_usd / (lots * 100)
        be_trigger_price = entry_price + target_profit_points
    else:
        sl_price = entry_price + sl_distance
        be_price = entry_price - be_buffer_points * 0.01
        target_profit_points = be_target_profit_usd / (lots * 100)
        be_trigger_price = entry_price - target_profit_points
    
    # Simulate: did price reach +$30 before hitting SL?
    reached_profit = False
    profit_bar = None
    hit_sl = False
    sl_bar = None
    final_sl = sl_price  # SL after BE
    be_activated = False
    
    for idx, bar in df.iterrows():
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        if direction == "BUY":
            # Check SL first
            if low <= final_sl:
                hit_sl = True
                sl_bar = idx
                break
            # Check BE trigger (price reaches +$30 profit)
            if not be_activated and high >= be_trigger_price:
                reached_profit = True
                profit_bar = idx
                final_sl = be_price  # Move SL to entry
                be_activated = True
        else:
            # SELL
            if high >= final_sl:
                hit_sl = True
                sl_bar = idx
                break
            if not be_activated and low <= be_trigger_price:
                reached_profit = True
                profit_bar = idx
                final_sl = be_price
                be_activated = True
    
    # Calculate result with BE-at-$30
    if hit_sl:
        new_pnl = (final_sl - entry_price) * lots * 100 if direction == "BUY" else (entry_price - final_sl) * lots * 100
        new_exit = "SL"
    else:
        new_pnl = (close - entry_price) * lots * 100 if direction == "BUY" else (entry_price - close) * lots * 100
        new_exit = "EOD"
    
    # The BE loss trades — check if they would convert to profit
    for t2 in trades[i+1:i+2]: pass  # placeholder
    
    improvement = new_pnl - t["pnl"]
    status = "UP" if improvement > 0 else "DN" if improvement < 0 else "--"
    
    print(f"  #{i+1:2d}: {direction:4s} @ ${entry_price:.2f} (lot {lots:.2f}) [{t['type']}]")
    print(f"       Old SL: ${sl_price:.2f} | BE trigger: +${be_target_profit_usd} at ${be_trigger_price:.2f}")
    print(f"       Reached +$30? {'YES' if reached_profit else 'NO '} | BE activated: {'YES' if be_activated else 'NO'}")
    if hit_sl:
        print(f"       Hit SL at ${final_sl:.2f} | PnL: ${new_pnl:+.2f} (was ${t['pnl']:+.2f}) {status} Improv: ${improvement:+.2f}")
    else:
        print(f"       Still open at close ${close:.2f} | PnL: ${new_pnl:+.2f} (was ${t['pnl']:+.2f}) {status}")
    
    results.append({
        "trade": i+1,
        "old_pnl": t["pnl"],
        "new_pnl": new_pnl,
        "improvement": improvement,
        "reached_profit": reached_profit,
        "be_activated": be_activated,
        "old_type": t["type"],
    })

# Summary
old_total = sum(r["old_pnl"] for r in results)
new_total = sum(r["new_pnl"] for r in results)
be_trades_improved = sum(1 for r in results if r["improvement"] > 0)
be_trades_saved = sum(1 for r in results if r["old_pnl"] < 0 and r["new_pnl"] > 0)

print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"  Old total P&L: ${old_total:+,.2f}")
print(f"  New total P&L: ${new_total:+,.2f}")
print(f"  Net improvement: ${new_total - old_total:+,.2f}")
print(f"  Trades improved: {be_trades_improved}/{len(results)}")
print(f"  Losses turned to wins: {be_trades_saved}")

mt5.shutdown()
print(f"\n✅ Analysis complete.")