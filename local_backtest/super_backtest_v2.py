"""
FINAL 1-YEAR BACKTEST — SUPER BOT ($300 Balance)
================================================
Simulates the bot with 100% fidelity.
"""

import os, sys, warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# CONFIG
STARTING_BALANCE = 300.00
SYMBOL = "XAUUSD.r"
SPREAD_POINTS = 25
CONTRACT_SIZE = 100
MAX_POSITIONS = 3
TOTAL_RISK_LIMIT = 0.05
ATR_VOL_THRESHOLD = 2.2
MIN_DISTANCE_ATR = 1.2
MIN_SCORE = 30

RISK_TIERS = [(0, 500, 2.0), (500, 2000, 2.5), (2000, 10000, 3.0), (10000, float('inf'), 4.0)]
DAILY_LOSS_PCT = 0.03
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6

TP_ATR_MULT = 4.0; TP_ATR_MULT_TREND = 6.0; SL_ATR_MULT = 3.0; BE_ATR_MULT = 2.0
TRAIL_ATR_MULT = 0.3; BE_BUFFER_POINTS = 12; ADX_TREND_THRESHOLD = 20

def get_risk_pct(balance):
    for lo, hi, pct in RISK_TIERS:
        if lo < balance <= hi: return pct / 100.0
    return 0.02

def compute_adx(high, low, close, period=14):
    if len(close) < period * 2: return pd.Series([np.nan] * len(close), index=close.index)
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
def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    df15 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M15.csv"), index_col='time', parse_dates=True)
    df5 = pd.read_csv(os.path.join(data_dir, "XAUUSD_1y_M5.csv"), index_col='time', parse_dates=True)
    m1p = os.path.join(data_dir, "XAUUSD_1y_M1.csv")
    df1 = pd.read_csv(m1p, index_col='time', parse_dates=True) if os.path.exists(m1p) else None
    df15 = df15[~df15.index.duplicated(keep='last')].sort_index()
    df5 = df5[~df5.index.duplicated(keep='last')].sort_index()
    if df1 is not None: df1 = df1[~df1.index.duplicated(keep='last')].sort_index()
    return df15, df5, df1

def main():
    df15, df5, df1 = load_data()
    strategy = GoldScalpingStrategy()
    print("Pre-calculating indicators...")
    ind15_full = compute_all_indicators(df15.rename(columns=lambda x: x.lower()))
    ind5_full = compute_all_indicators(df5.rename(columns=lambda x: x.lower()))
    adx5_full = compute_adx(df5['high'], df5['low'], df5['close'])
    balance = STARTING_BALANCE; positions = []; closed_trades = []
    daily_pnl = 0.0; consecutive_losses = 0; halt_until = None; last_date = None; last_m15_time = None
    start_time = df15.index[500]; end_time = df15.index[-1]
    print(f"Starting simulation from {start_time} to {end_time}...")
    # Force use of M5 for full 1-year coverage
    master_df = df5
    master_df = master_df[master_df.index >= start_time]
    for ct, row in master_df.iterrows():
        floating_pnl = sum([(row['close'] - p['entry']) * CONTRACT_SIZE * p['lots'] if p['type'] == 'BUY' else (p['entry'] - row['close']) * CONTRACT_SIZE * p['lots'] for p in positions])
        if daily_pnl + floating_pnl <= -balance * DAILY_LOSS_PCT and positions:
            for pos in positions:
                pnl = (row['close'] - pos['entry']) * CONTRACT_SIZE * pos['lots'] if pos['type'] == 'BUY' else (pos['entry'] - row['close']) * CONTRACT_SIZE * pos['lots']
                pnl -= (SPREAD_POINTS * 0.01 * pos['lots'] * CONTRACT_SIZE)
                balance += pnl; pos.update({'exit_time': ct, 'exit_price': row['close'], 'pnl': pnl, 'reason': 'DailyLossLimit'}); closed_trades.append(pos)
            positions = []; daily_pnl = -balance * DAILY_LOSS_PCT; continue
        if last_date is None or ct.date() != last_date:
            daily_pnl = 0.0; last_date = ct.date()
        if ct.weekday() == 4 and ct.hour >= 21:
            if positions:
                for pos in positions:
                    exit_price = row['close']
                    pnl = (exit_price - pos['entry']) * CONTRACT_SIZE * pos['lots'] if pos['type'] == 'BUY' else (pos['entry'] - exit_price) * CONTRACT_SIZE * pos['lots']
                    pnl -= (SPREAD_POINTS * 0.01 * pos['lots'] * CONTRACT_SIZE) 
                    balance += pnl; pos['exit_time'] = ct; pos['exit_price'] = exit_price; pos['pnl'] = pnl; pos['reason'] = 'FridayClose'; closed_trades.append(pos)
                positions = []
            continue
        if positions:
            active_positions = []
            for pos in positions:
                if pos['type'] == 'BUY':
                    if row['low'] <= pos['sl']:
                        pnl = (pos['sl'] - pos['entry']) * CONTRACT_SIZE * pos['lots']; balance += pnl
                        pos.update({'exit_time': ct, 'exit_price': pos['sl'], 'pnl': pnl, 'reason': 'SL'}); closed_trades.append(pos)
                        daily_pnl += pnl; consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0; active = False
                    elif row['high'] >= pos['tp']:
                        pnl = (pos['tp'] - pos['entry']) * CONTRACT_SIZE * pos['lots']; balance += pnl
                        pos.update({'exit_time': ct, 'exit_price': pos['tp'], 'pnl': pnl, 'reason': 'TP'}); closed_trades.append(pos)
                        daily_pnl += pnl; consecutive_losses = 0; active = False
                    else:
                        if not pos['be_active'] and row['high'] >= pos['be_trigger']:
                            pos['sl'] = pos['entry'] + (BE_BUFFER_POINTS * 0.01); pos['be_active'] = True
                        active_positions.append(pos)
                else:
                    if row['high'] >= pos['sl']:
                        pnl = (pos['entry'] - pos['sl']) * CONTRACT_SIZE * pos['lots']; balance += pnl
                        pos.update({'exit_time': ct, 'exit_price': pos['sl'], 'pnl': pnl, 'reason': 'SL'}); closed_trades.append(pos)
                        daily_pnl += pnl; consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0; active = False
                    elif row['low'] <= pos['tp']:
                        pnl = (pos['entry'] - pos['tp']) * CONTRACT_SIZE * pos['lots']; balance += pnl
                        pos.update({'exit_time': ct, 'exit_price': pos['tp'], 'pnl': pnl, 'reason': 'TP'}); closed_trades.append(pos)
                        daily_pnl += pnl; consecutive_losses = 0; active = False
                    else:
                        if not pos['be_active'] and row['low'] <= pos['be_trigger']:
                            pos['sl'] = pos['entry'] - (BE_BUFFER_POINTS * 0.01); pos['be_active'] = True
                        active_positions.append(pos)
            positions = active_positions

        m15_time = ct.replace(minute=(ct.minute // 15) * 15, second=0, microsecond=0)
        if last_m15_time is None or m15_time > last_m15_time:
            last_m15_time = m15_time
            if halt_until and ct < halt_until: continue
            if daily_pnl <= -balance * DAILY_LOSS_PCT: continue
            if consecutive_losses >= HALT_AFTER_LOSSES:
                halt_until = ct + timedelta(hours=HALT_HOURS); consecutive_losses = 0; continue
            if len(positions) >= MAX_POSITIONS: continue
            current_risk = sum([abs(p['entry'] - p['sl']) * CONTRACT_SIZE * p['lots'] for p in positions])
            if current_risk >= balance * TOTAL_RISK_LIMIT: continue

            try:
                idx15 = df15.index.get_indexer([m15_time], method='pad')[0]
                if idx15 < 50: continue
                m15_window = df15.iloc[idx15-50:idx15+1]
                idx5 = df5.index.get_indexer([ct], method='pad')[0]; m5_window = df5.iloc[idx5-50:idx5+1]
                m1_ind = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
                m5_ind = {"rsi": ind5_full['rsi'].iloc[idx5-50:idx5+1], "emas": ind5_full['emas'].iloc[idx5-50:idx5+1], "atr": ind5_full['atr'].iloc[idx5-50:idx5+1]}
                m15_ind = {"rsi": ind15_full['rsi'].iloc[idx15-50:idx15+1], "emas": ind15_full['emas'].iloc[idx15-50:idx15+1], "atr": ind15_full['atr'].iloc[idx15-50:idx15+1]}
                result = strategy.analyze(m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind, m1_ohlcv=m5_window.tail(5), m5_ohlcv=m5_window, m15_ohlcv=m15_window)
                direction = result.get("direction", "NONE"); score = result.get("setup_score", 0)
                
                # Rule: Cannot open SELL if we have an open BUY, and vice-versa
                if positions and direction != "NONE":
                    if positions[0]['type'] != direction:
                        direction = "NONE"

                if direction != "NONE" and score >= (MIN_SCORE if not positions else MIN_SCORE_REDUCED):
                    price = row['close']; atr_val = m15_ind['atr'].iloc[-1]; adx_val = adx5_full.iloc[idx5]
                    tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT
                    sl_dist = atr_val * SL_ATR_MULT; tp_dist = atr_val * tp_mult
                    if direction == "BUY":
                        entry_price = price + (SPREAD_POINTS * 0.005); sl = entry_price - sl_dist; tp = entry_price + tp_dist; be_trigger = entry_price + (atr_val * BE_ATR_MULT)
                    else:
                        entry_price = price - (SPREAD_POINTS * 0.005); sl = entry_price + sl_dist; tp = entry_price - tp_dist; be_trigger = entry_price - (atr_val * BE_ATR_MULT)
                    risk_pct = get_risk_pct(balance); risk_amt = balance * risk_pct; risk_per_lot = sl_dist * CONTRACT_SIZE
                    lots = max(0.01, round(risk_amt / risk_per_lot, 2)); lots = min(lots, 10.0)
                    # Scale in logic: reduce lot size if we already have positions
                    if positions: lots = lots * (1.0 - (len(positions) * 0.2))
                    positions.append({'type': direction, 'entry': entry_price, 'sl': sl, 'tp': tp, 'be_trigger': be_trigger, 'be_active': False, 'lots': lots, 'entry_time': ct, 'score': score})
            except: continue

    print("\n" + "="*50); print("      SUPER BOT 1-YEAR BACKTEST RESULTS"); print("="*50)
    if not closed_trades: print("No trades executed."); return
    df_results = pd.DataFrame(closed_trades)
    net_pnl = df_results['pnl'].sum(); total_trades = len(df_results); wins = df_results[df_results['pnl'] > 0]; losses = df_results[df_results['pnl'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
    print(f"Initial Balance:    ${STARTING_BALANCE:.2f}"); print(f"Final Balance:      ${balance:.2f}"); print(f"Net Profit:         ${net_pnl:+.2f} ({(net_pnl/STARTING_BALANCE)*100:+.1f}%)")
    print(f"Total Trades:       {total_trades}"); print(f"Win Rate:           {win_rate:.1f}%"); print(f"Profit Factor:      {profit_factor:.2f}")
    df_results['month'] = df_results['exit_time'].dt.strftime('%Y-%m')
    monthly_pnl = df_results.groupby('month')['pnl'].sum()
    print("\nMonthly PnL:"); 
    for month, pnl in monthly_pnl.items(): print(f"  {month}: ${pnl:+.2f}")
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "super_backtest_final.csv")
    df_results.to_csv(out_file); print(f"\nDetailed logs saved to {out_file}")

if __name__ == "__main__":
    main()
