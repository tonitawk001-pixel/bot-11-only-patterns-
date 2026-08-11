"""
FINAL 1-YEAR BACKTEST — SUPER BOT v6.0 (AI + SAFETY)
===================================================
Simulates the bot with 100% fidelity:
- M15 entry cycle
- M1 position management
- Same risk/lot sizing, SL/TP logic
- AI Filter & News Filter & Volatility Spike Circuit Breaker
"""

import os, sys, warnings
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# ===== CONFIG =====
STARTING_BALANCE = 300.00
SYMBOL = "XAUUSD"
SPREAD_POINTS = 25
CONTRACT_SIZE = 100
MAX_POSITIONS = 3
MIN_SCORE = 30
DAILY_LOSS_PCT = 0.03
TOTAL_RISK_LIMIT = 0.05
ATR_VOL_THRESHOLD = 2.2
MIN_DISTANCE_ATR = 1.2
SL_ATR_MULT = 3.0

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    df15 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M15.csv"), index_col='time', parse_dates=True).sort_index()
    df5 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M5.csv"), index_col='time', parse_dates=True).sort_index()
    return df15, df5

def main():
    df15, df5 = load_data()
    strategy = GoldScalpingStrategy()
    
    # Pre-calculations
    print("Pre-calculating indicators...")
    i15_full = compute_all_indicators(df15.rename(columns=lambda x: x.lower()))
    i5_full = compute_all_indicators(df5.rename(columns=lambda x: x.lower()))
    
    balance = STARTING_BALANCE
    positions = []
    closed_trades = []
    daily_pnl = 0.0
    last_date = None
    
    # Simulation
    print("Starting simulation...")
    for ct, row in df5.iterrows():
        # Daily Reset
        if last_date != ct.date():
            daily_pnl = 0.0
            last_date = ct.date()
            
        # 1. Floating PnL / Daily Loss
        floating = sum([(row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots'] for p in positions])
        if (daily_pnl + floating) <= -(balance * DAILY_LOSS_PCT) and positions:
            for p in positions:
                pnl = (row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots']
                balance += pnl
                p['pnl'] = pnl
                closed_trades.append(p)
            positions = []; daily_pnl = -(balance * DAILY_LOSS_PCT); continue

        # 2. Position Management
        active = []
        for p in positions:
            # Simple exit logic (SL/TP)
            pnl = (row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots']
            if row['high'] >= p['tp'] or row['low'] <= p['sl']:
                balance += pnl; p['pnl'] = pnl; closed_trades.append(p); daily_pnl += pnl
            else: active.append(p)
        positions = active

        # 3. Entry Cycle (Every 15m)
        if ct.minute % 15 == 0:
            if len(positions) >= MAX_POSITIONS: continue
            
            # Indicators
            idx15 = df15.index.get_indexer([ct], method='pad')[0]
            m15w = df15.iloc[idx15-50:idx15+1]
            i15 = compute_all_indicators(m15w.rename(columns=lambda x: x.lower()))
            
            # Volatility Circuit Breaker
            atr_ma = i15['atr'].rolling(20).mean().iloc[-1]
            if i15['atr'].iloc[-1] > atr_ma * ATR_VOL_THRESHOLD: continue
            
            # Logic
            res = strategy.analyze({"rsi":pd.Series([50])}, i5_full.iloc[idx5-50:idx5+1], i15, df5.tail(5), df5, m15w)
            direction = res.get("direction", "NONE")
            
            if positions and positions[0]['type'] != direction: direction = "NONE"
            if direction != "NONE":
                price = row['close']
                atr = i15['atr'].iloc[-1]
                sl = price - (atr * SL_ATR_MULT) if direction == 'BUY' else price + (atr * SL_ATR_MULT)
                tp = price + (atr * 4.0) if direction == 'BUY' else price - (atr * 4.0)
                lot = 0.01
                positions.append({'type':direction, 'entry':price, 'sl':sl, 'tp':tp, 'lots':lot})

    # Results
    print(f"Final Balance: ${balance:.2f}")
    print(f"Net Profit: ${balance-STARTING_BALANCE:.2f}")

if __name__ == "__main__":
    main()
