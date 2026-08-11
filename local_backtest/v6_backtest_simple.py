"""
FINAL 1-YEAR BACKTEST — SUPER BOT v6.0 (AI + SAFETY)
"""

import os, sys, warnings
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

STARTING_BALANCE = 300.00
SYMBOL = "XAUUSD"
CONTRACT_SIZE = 100
MAX_POSITIONS = 3
DAILY_LOSS_PCT = 0.03
TOTAL_RISK_LIMIT = 0.05
ATR_VOL_THRESHOLD = 2.2
SL_ATR_MULT = 3.0

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    # Load and align
    df15 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M15.csv"), index_col='time', parse_dates=True).sort_index()
    df5 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M5.csv"), index_col='time', parse_dates=True).sort_index()
    return df15, df5

def main():
    df15, df5 = load_data()
    strategy = GoldScalpingStrategy()
    
    # Pre-calculate
    print("Pre-calculating...")
    i15 = compute_all_indicators(df15.rename(columns=lambda x: x.lower()))
    i5 = compute_all_indicators(df5.rename(columns=lambda x: x.lower()))
    
    balance = STARTING_BALANCE
    positions = []
    closed_trades = []
    daily_pnl = 0.0
    last_date = None
    
    print("Running simulation...")
    # Use M15 frequency for entries as per bot
    for ct, row in df15.iterrows():
        if last_date != ct.date(): daily_pnl = 0.0; last_date = ct.date()
        
        # Positions
        floating = sum([(row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots'] for p in positions])
        if (daily_pnl + floating) <= -(balance * DAILY_LOSS_PCT) and positions:
            for p in positions: balance += (row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots']
            positions = []; daily_pnl = -(balance * DAILY_LOSS_PCT); continue

        # Entry logic
        if len(positions) < MAX_POSITIONS:
            # Simple simulation of signal
            res = strategy.analyze({"rsi":pd.Series([50])}, i5, i15, df5.tail(5), df5, df15)
            direction = res.get("direction", "NONE")
            
            # Constraints
            if positions and positions[0]['type'] != direction: direction = "NONE"
            if direction != "NONE":
                # Apply risk
                atr = i15['atr'].iloc[-1]
                sl_dist = atr * SL_ATR_MULT
                lot = max(0.01, round((balance * 0.02) / (sl_dist * 100), 2))
                positions.append({'type':direction, 'entry':row['close'], 'sl':row['close']-sl_dist if direction=='BUY' else row['close']+sl_dist, 'tp':row['close']+sl_dist*1.5 if direction=='BUY' else row['close']-sl_dist*1.5, 'lots':lot})
        
        # Manage
        active = []
        for p in positions:
            if (p['type'] == 'BUY' and (row['high'] >= p['tp'] or row['low'] <= p['sl'])) or \
               (p['type'] == 'SELL' and (row['low'] <= p['tp'] or row['high'] >= p['sl'])):
                pnl = (row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots']
                balance += pnl; daily_pnl += pnl; closed_trades.append(p)
            else: active.append(p)
        positions = active

    print(f"Final Balance: ${balance:.2f}")

if __name__ == "__main__": main()
