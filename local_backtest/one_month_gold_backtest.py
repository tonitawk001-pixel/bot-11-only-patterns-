"""
ONE MONTH GOLD BACKTEST (JUNE-JULY 2026)
========================================
Uses the EXACT logic from main_super.py v5.0
Replicates live behavior: M15 analysis, M5 confluence, Risk tiers.

Period: 2026-06-24 to 2026-07-24
"""

import os, sys, warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Project Setup
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# CONFIG FROM main_super.py
SYMBOL = "XAUUSD"
MIN_SCORE = 30
MAX_POSITIONS = 3
MIN_ATR = 0.5
TRADE_HOURS_START = 0
TRADE_HOURS_END = 24
TP_ATR_MULT = 4.0
TP_ATR_MULT_TREND = 6.0
SL_ATR_MULT = 3.0 # v5.0 uses 3.0x ATR for SL
BE_ATR_MULT = 2.0
TRAIL_ATR_MULT = 0.3
BE_BUFFER_POINTS = 12
RISK_TIERS = [
    (0, 500, 2.0),
    (500, 2000, 2.5),
    (2000, 10000, 3.0),
    (10000, float('inf'), 4.0),
]
ADX_TREND_THRESHOLD = 20
DAILY_LOSS_PCT = 0.03
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
SPREAD_COST_PIP = 0.50 # Realistic XAUUSD spread

def compute_adx(high, low, close, period=14):
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move = high - high.shift(); down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()

def get_risk_pct(balance):
    for lo, hi, pct in RISK_TIERS:
        if lo < balance <= hi:
            return pct / 100.0
    return 0.02

def run_backtest():
    # Load Data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    m15_file = os.path.join(data_dir, "XAUUSD_60d_M15.csv")
    m5_file = os.path.join(data_dir, "XAUUSD_60d_M5.csv")

    df15 = pd.read_csv(m15_file)
    df15['Datetime'] = pd.to_datetime(df15['Datetime'], utc=True)
    df15.set_index('Datetime', inplace=True)
    df15.columns = [c.lower() for c in df15.columns]
    
    df5 = pd.read_csv(m5_file)
    df5['Datetime'] = pd.to_datetime(df5['Datetime'], utc=True)
    df5.set_index('Datetime', inplace=True)
    df5.columns = [c.lower() for c in df5.columns]

    # Filter for last month + warmup
    end_date = df15.index[-1]
    start_date = end_date - timedelta(days=30)
    warmup_start = start_date - timedelta(days=5) # 5 days for indicators to stabilize

    m15_bt = df15[(df15.index >= warmup_start) & (df15.index <= end_date)].sort_index()
    m5_bt = df5[(df5.index >= warmup_start) & (df5.index <= end_date)].sort_index()

    print(f"Backtest Period: {start_date} to {end_date}")
    print(f"Total M15 bars: {len(m15_bt)}")

    strategy = GoldScalpingStrategy()
    balance = 500.0
    positions = []
    closed_trades = []
    daily_pnl = 0.0
    consecutive_losses = 0
    halt_until = None
    last_date = None

    for idx, (ct, row) in enumerate(m15_bt.iterrows()):
        if ct < start_date:
            continue # Warmup period

        price = float(row['close'])
        
        # Friday Close logic
        if ct.weekday() == 4 and ct.hour >= 21:
            for p in positions:
                pnl = (price - p['entry']) * p['lot'] * 100 if p['dir'] == "BUY" else (p['entry'] - price) * p['lot'] * 100
                pnl -= SPREAD_COST_PIP * p['lot'] * 100
                balance += pnl
                p['pnl'] = pnl; p['reason'] = "FRIDAY_CLOSE"; p['close_price'] = price; p['close_time'] = ct
                closed_trades.append(p)
            positions = []
            continue

        # Daily Reset
        if last_date != ct.date():
            daily_pnl = 0.0
            last_date = ct.date()

        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * DAILY_LOSS_PCT:
            continue
        if not (TRADE_HOURS_START <= ct.hour < TRADE_HOURS_END):
            continue

        # Position Management
        surviving = []
        m5_win = m5_bt[m5_bt.index <= ct].tail(100)
        if not m5_win.empty:
            atr_series = compute_all_indicators(m5_win).get('atr')
            atr_val = float(atr_series.iloc[-1]) if atr_series is not None else MIN_ATR
        else:
            atr_val = MIN_ATR

        for p in positions:
            entry, direction, sl, tp, lot = p['entry'], p['dir'], p['sl'], p['tp'], p['lot']
            pv = lot * 100
            hit = False; pnl = 0.0; reason = ""

            # Breakeven/Trailing
            if not p.get('be', False) and p.get('be_target'):
                if (direction == "BUY" and price >= p['be_target']) or (direction == "SELL" and price <= p['be_target']):
                    p['be'] = True; p['sl'] = entry + (0.12 if direction == "BUY" else -0.12)

            if p.get('be'):
                ns = price - atr_val * TRAIL_ATR_MULT if direction == "BUY" else price + atr_val * TRAIL_ATR_MULT
                if direction == "BUY" and ns > p['sl'] + 0.5: p['sl'] = round(ns, 2)
                elif direction == "SELL" and ns < p['sl'] - 0.5: p['sl'] = round(ns, 2)

            # Exit check
            if direction == "BUY":
                if price >= tp: pnl = (tp - entry) * pv; reason = "TP"; hit = True
                elif price <= p['sl']: pnl = (p['sl'] - entry) * pv; reason = "SL/TRAIL"; hit = True
            else:
                if price <= tp: pnl = (entry - tp) * pv; reason = "TP"; hit = True
                elif price >= p['sl']: pnl = (entry - p['sl']) * pv; reason = "SL/TRAIL"; hit = True

            if hit:
                pnl -= SPREAD_COST_PIP * lot * 100
                balance += pnl
                daily_pnl += pnl
                p['pnl'] = pnl; p['reason'] = reason; p['close_price'] = price; p['close_time'] = ct
                closed_trades.append(p)
                if pnl < 0: consecutive_losses += 1
                else: consecutive_losses = 0
                
                if consecutive_losses >= HALT_AFTER_LOSSES:
                    halt_until = ct + timedelta(hours=HALT_HOURS)
                    consecutive_losses = 0
            else:
                surviving.append(p)
        positions = surviving

        # New Entry
        if len(positions) < MAX_POSITIONS:
            m15_window = m15_bt[m15_bt.index <= ct].tail(500)
            m5_window = m5_bt[m5_bt.index <= ct].tail(500)
            
            if len(m15_window) < 50 or len(m5_window) < 50: continue
            
            ind5 = compute_all_indicators(m5_window)
            ind15 = compute_all_indicators(m15_window)
            
            if ind5 is None or ind15 is None: continue
            
            atr_val = float(ind5['atr'].iloc[-1])
            if atr_val < MIN_ATR: continue

            # Strategy Analysis
            try:
                empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
                result = strategy.analyze(
                    m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                    m1_ohlcv=m5_window.tail(20), m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None
                )
            except: continue

            direction = result.get("direction", "NONE")
            score = result.get("setup_score", 0)

            if direction != "NONE" and score >= MIN_SCORE:
                # RSI confluence
                rsi5 = ind5['rsi'].iloc[-1]
                rsi15 = ind15['rsi'].iloc[-1]
                if direction == "BUY" and (rsi5 <= 40 or rsi15 <= 40): continue
                if direction == "SELL" and (rsi5 >= 60 or rsi15 >= 60): continue

                # TP/SL Calculation
                adx_series = compute_adx(m5_window['high'], m5_window['low'], m5_window['close'])
                adx_val = adx_series.iloc[-1]
                tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT
                
                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = atr_val * tp_mult
                
                if direction == "BUY":
                    sl = round(price - sl_dist, 2); tp = round(price + tp_dist, 2)
                else:
                    sl = round(price + sl_dist, 2); tp = round(price - tp_dist, 2)

                risk_pct = get_risk_pct(balance)
                lot = max(0.01, round((balance * risk_pct) / (sl_dist * 100), 2))
                
                positions.append({
                    "entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
                    "open_time": ct, "score": score, "be_target": price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT),
                    "be": False
                })

    # Results
    if not closed_trades:
        print("No trades executed.")
        return

    df_res = pd.DataFrame(closed_trades)
    net_pnl = df_res['pnl'].sum()
    win_rate = (df_res['pnl'] > 0).mean() * 100
    total_trades = len(df_res)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (ONE MONTH)")
    print("="*40)
    print(f"Starting Balance: $500.00")
    print(f"Final Balance:    ${500 + net_pnl:.2f}")
    print(f"Net Profit:       ${net_pnl:.2f} ({(net_pnl/500)*100:+.1f}%)")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total Trades:     {total_trades}")
    print(f"Avg Trade:        ${df_res['pnl'].mean():.2f}")
    print(f"Max Win:          ${df_res['pnl'].max():.2f}")
    print(f"Max Loss:         ${df_res['pnl'].min():.2f}")
    
    # Drawdown
    equity = 500 + df_res['pnl'].cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak * 100
    print(f"Max DD %:         {dd.max():.1f}%")
    
    with open("local_backtest/one_month_results.txt", "w") as f:
        f.write(f"GOLD BACKTEST REPORT: {start_date.date()} to {end_date.date()}\n")
        f.write("="*50 + "\n")
        f.write(f"Net Profit: ${net_pnl:.2f}\n")
        f.write(f"Win Rate: {win_rate:.1f}%\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Max DD: {dd.max():.1f}%\n")

if __name__ == "__main__":
    run_backtest()
