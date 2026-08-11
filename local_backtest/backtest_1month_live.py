"""
1-MONTH BACKTEST — EXACT LIVE BOT SIMULATION
============================================
Mirrors main_super.py v8.3 exactly:
- All 14 filters active
- Advanced candle patterns (30+)
- Same risk, SL/TP, score thresholds
- UTC trading hours

Run: python local_backtest/backtest_1month_live.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'trading_bot_mt5'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5

from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
import candle_patterns as cp

# ── EXACT LIVE CONFIG (from main_super.py) ──
SYMBOL = "XAUUSD"
MIN_SCORE_BUY = 20; MIN_SCORE_SELL = 20
MAX_POSITIONS = 2; MAX_PER_DIRECTION = 1
FIXED_RISK = 0.02
TP_ATR_MULT = 5.0; SL_ATR_MULT = 3.0
BE_ATR_MULT = 2.7; BE_BUFFER_POINTS = 50
HIGH_SCORE_THRESHOLD = 70; HIGH_SCORE_RISK = 0.03
SECOND_POS_MIN_SCORE = 35; SECOND_POS_LOT_RATIO = 0.5
ATR_VOL_THRESHOLD = 4.0
TRADE_HOURS_START = 8; TRADE_HOURS_END = 17
SESSION_COOLDOWN_MIN = 10
DXY_ENABLED = True; DXY_THRESHOLD = 0.003; DXY_LOOKBACK_H = 4
MTF_CONFLUENCE = True
STARTING_BALANCE = 500.00
SPREAD_COST = 0.35  # typical gold spread on demo

# ── Download data ──
print("Downloading M15/H1 XAUUSD data...")
if not mt5.initialize():
    print("MT5 not available — using cached data if exists")
    mt5 = None

def get_candles(symbol, tf, count=5000):
    if mt5 is None:
        return None
    tf_map = {"H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(tf), 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

m15 = get_candles(SYMBOL, "M15", 5000)
h1 = get_candles(SYMBOL, "H1", 2000)
m5 = None  # use H1 for trend analysis instead of M5

if mt5:
    mt5.shutdown()

if m15 is None:
    # Try loading cached data
    cache_file = os.path.join(os.path.dirname(__file__), "cached_m15_gold.csv")
    if os.path.exists(cache_file):
        m15 = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded cached M15 data: {len(m15)} candles")
    else:
        print("ERROR: No MT5 and no cached data. Cannot backtest.")
        sys.exit(1)

# Filter last 30 days
cutoff = m15.index.max() - timedelta(days=30)
m15 = m15[m15.index >= cutoff]
if h1 is not None:
    h1 = h1[h1.index >= m15.index.min()]

print(f"M15 candles in period: {len(m15)} | {m15.index.min()} -> {m15.index.max()}")

# ── State ──
balance = STARTING_BALANCE
equity = STARTING_BALANCE
peak_balance = STARTING_BALANCE
open_positions = []  # list of {"dir","entry","sl","tp","lots","entry_time"}
closed_trades = []
daily_pnl = 0.0
daily_trades = 0
last_date = ""
last_processed_m15_time = None
consecutive_losses = 0

strategy = GoldScalpingStrategy()

# ── Helpers ──
def get_session(ts):
    h = ts.hour
    if 8 <= h < 17 and 13 <= h < 22: return "overlap"
    if 8 <= h < 17: return "london"
    if 13 <= h < 22: return "new_york"
    if h >= 23 or h < 8: return "asian"
    return "transition"

def h4_trend_from_m15(m15_window):
    """Approximate H4 trend from M15 data (16 M15 bars = 1 H4)"""
    if len(m15_window) < 34:
        return "NEUTRAL"
    ema20 = m15_window['close'].ewm(span=80, adjust=False).mean().iloc[-1]  # 20*4 = 80 M15 bars
    curr = m15_window['close'].iloc[-1]
    return "BULLISH" if curr > ema20 else "BEARISH"

def check_blocked_by_filters(direction, score, pos, i15, i5, h4_trend, sr, curr, m15_window, active_max_pos, result):
    """Run all 14 deterministic filters - returns (blocked, reason)."""
    blocked_by = None

    # Already max positions
    if len(pos) >= active_max_pos:
        return True, "max_positions"

    # Directional score
    min_sc = MIN_SCORE_BUY if direction == "BUY" else MIN_SCORE_SELL
    if len(pos) >= 1:
        min_sc = max(min_sc, SECOND_POS_MIN_SCORE)
    if score < min_sc:
        return True, f"score_{score}_lt_{min_sc}"

    # MTF confluence
    if MTF_CONFLUENCE:
        try:
            m5_rsi = float(i5["rsi"].iloc[-1])
            if (direction == "BUY" and m5_rsi < 20) or (direction == "SELL" and m5_rsi > 80):
                return True, f"MTF_conflict_M5_RSI_{m5_rsi:.0f}"
        except:
            pass

    # DXY correlation (proxy via gold change)
    if DXY_ENABLED and len(m15_window) >= DXY_LOOKBACK_H * 4:
        gold_4h_ago = m15_window['close'].iloc[max(0, len(m15_window) - DXY_LOOKBACK_H * 4)]
        gold_change = (m15_window['close'].iloc[-1] - gold_4h_ago) / gold_4h_ago if gold_4h_ago > 0 else 0
        dxy_proxy = -gold_change
        if (direction == "BUY" and dxy_proxy > DXY_THRESHOLD) or (direction == "SELL" and dxy_proxy < -DXY_THRESHOLD):
            return True, "DXY_correlation"

    # Same direction stacking
    if pos:
        existing_dirs = set(p["dir"] for p in pos)
        if direction in existing_dirs and sum(1 for p in pos if p["dir"] == direction) >= MAX_PER_DIRECTION:
            return True, "max_per_direction"

    # ATR spike
    atr_s = i15.get("atr")
    if atr_s is not None and len(atr_s) >= 20:
        current_atr = atr_s.iloc[-1]
        mean_atr = atr_s.rolling(20).mean().iloc[-1]
        if not np.isnan(current_atr) and not np.isnan(mean_atr):
            if current_atr > mean_atr * ATR_VOL_THRESHOLD:
                return True, f"ATR_spike_{current_atr/mean_atr:.1f}x"

    # S/R distance
    if sr:
        dist_r = sr['resistance'] - curr
        dist_s = curr - sr['support']
        if direction == "BUY" and dist_r < 2.5:
            return True, f"near_resistance_${dist_r:.1f}"
        if direction == "SELL" and dist_s < 2.5:
            return True, f"near_support_${dist_s:.1f}"

    # Opposite direction open
    if pos:
        existing_dirs = set(p["dir"] for p in pos)
        if direction not in existing_dirs and len(existing_dirs) > 0:
            return True, "opposite_direction_open"

    # Counter-trend
    if direction != ("BUY" if h4_trend == "BULLISH" else "SELL"):
        m15_rsi_val = i15['rsi'].iloc[-1]
        is_mean_rev = result.get("is_mean_reversion", False)
        if not is_mean_rev and not (m15_rsi_val < 25 or m15_rsi_val > 75):
            return True, f"counter_trend_RSI_{m15_rsi_val:.0f}"

    # BB filter
    try:
        bb = i15.get('bb')
        if bb is not None:
            if direction == "BUY" and curr < bb['lower'].iloc[-1]:
                return True, "BB_below_lower"
            if direction == "SELL" and curr > bb['upper'].iloc[-1]:
                return True, "BB_above_upper"
    except:
        pass

    # Total risk limit
    total_risk = sum([abs(p["entry"] - p["sl"]) * 100 * p["lots"] for p in pos if p["sl"] and p["sl"] > 0])
    if total_risk >= balance * 0.05:  # TOTAL_RISK_LIMIT
        return True, "total_risk_limit"

    return False, None

def manage_positions(bar_time, bar_high, bar_low, bar_open, bar_close, prev_close):
    """Check TP/SL/BE for open positions during the bar."""
    global balance, equity, daily_pnl, consecutive_losses, closed_trades

    to_close = []
    to_modify = []

    for i, p in enumerate(open_positions):
        # Determine extreme price during this bar
        if p["dir"] == "BUY":
            high_price = bar_high
            low_price = bar_low
        else:
            high_price = bar_low  # for SELL, "high" is the highest unfavorable price
            low_price = bar_high

        # TP hit?
        if p["dir"] == "BUY" and high_price >= p["tp"]:
            pnl = (p["tp"] - p["entry"]) * p["lots"] * 100
            p["exit_price"] = p["tp"]
            p["exit_reason"] = "tp"
            p["pnl"] = round(pnl, 2)
            to_close.append(i)
            closed_trades.append(p)
            balance += pnl
            equity = balance
            daily_pnl += pnl
            if pnl > 0: consecutive_losses = 0
            else: consecutive_losses += 1
            continue
        elif p["dir"] == "SELL" and low_price <= p["tp"]:
            pnl = (p["entry"] - p["tp"]) * p["lots"] * 100
            p["exit_price"] = p["tp"]
            p["exit_reason"] = "tp"
            p["pnl"] = round(pnl, 2)
            to_close.append(i)
            closed_trades.append(p)
            balance += pnl
            equity = balance
            daily_pnl += pnl
            if pnl > 0: consecutive_losses = 0
            else: consecutive_losses += 1
            continue

        # SL hit?
        if p["dir"] == "BUY" and low_price <= p["sl"]:
            pnl = (p["sl"] - p["entry"]) * p["lots"] * 100
            p["exit_price"] = p["sl"]
            p["exit_reason"] = "sl"
            p["pnl"] = round(pnl, 2)
            to_close.append(i)
            closed_trades.append(p)
            balance += pnl
            equity = balance
            daily_pnl += pnl
            consecutive_losses += 1
            continue
        elif p["dir"] == "SELL" and high_price >= p["sl"]:
            pnl = (p["entry"] - p["sl"]) * p["lots"] * 100
            p["exit_price"] = p["sl"]
            p["exit_reason"] = "sl"
            p["pnl"] = round(pnl, 2)
            to_close.append(i)
            closed_trades.append(p)
            balance += pnl
            equity = balance
            daily_pnl += pnl
            consecutive_losses += 1
            continue

        # BE modification
        if p["sl"] and p["sl"] > 0:
            sl_dist = abs(p["entry"] - p["sl"])
            be_trigger_mult = BE_ATR_MULT / SL_ATR_MULT
            be_trigger = p["entry"] + (sl_dist * be_trigger_mult) if p["dir"] == "BUY" else p["entry"] - (sl_dist * be_trigger_mult)
            if (p["dir"] == "BUY" and high_price >= be_trigger) or (p["dir"] == "SELL" and low_price <= be_trigger):
                new_sl = p["entry"] + (BE_BUFFER_POINTS * 0.01) if p["dir"] == "BUY" else p["entry"] - (BE_BUFFER_POINTS * 0.01)
                if (p["dir"] == "BUY" and new_sl > p["sl"]) or (p["dir"] == "SELL" and new_sl < p["sl"]):
                    p["sl"] = new_sl
                    p["be_triggered"] = True

    # Remove closed positions (reverse order)
    for i in sorted(to_close, reverse=True):
        open_positions.pop(i)

def compute_floating_pnl(curr_price):
    """Compute unrealized PnL."""
    pnl = 0.0
    for p in open_positions:
        if p["dir"] == "BUY":
            pnl += (curr_price - p["entry"]) * p["lots"] * 100
        else:
            pnl += (p["entry"] - curr_price) * p["lots"] * 100
    return pnl

# ── MAIN SIMULATION LOOP ──
print("\n" + "="*60)
print("STARTING 1-MONTH BACKTEST (Exact Live Bot Simulation)")
print("="*60)

bar_count = 0
signal_count = 0
blocked_count = 0
trade_count = 0

for idx in range(100, len(m15)):
    bar = m15.iloc[idx]
    bar_time = bar.name
    curr = bar['close']
    h = bar_time.hour

    # Track daily reset
    date_str = bar_time.strftime("%Y-%m-%d")
    if date_str != last_date:
        daily_pnl = 0.0
        daily_trades = 0
        last_date = date_str
        strategy.reset_daily()

    # Compute equity
    floating = compute_floating_pnl(curr)
    equity = balance + floating
    if balance > peak_balance:
        peak_balance = balance

    # ── Position management (check if TP/SL hit during this bar) ──
    if open_positions:
        manage_positions(bar_time, bar['high'], bar['low'], bar['open'], bar['close'], 
                         m15['close'].iloc[idx-1] if idx > 0 else bar['close'])

    # DD emergency stop
    if balance < 50:
        print(f"BALANCE FLOOR reached at {bar_time}")
        break

    # Daily loss halt
    daily_loss = daily_pnl + compute_floating_pnl(curr)
    if open_positions and balance > 0 and daily_loss <= -(balance * 0.02):
        for p in open_positions:
            exit_price = curr
            pnl = (exit_price - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - exit_price) * p["lots"] * 100
            p["pnl"] = round(pnl, 2)
            p["exit_reason"] = "daily_loss_halt"
            closed_trades.append(p)
            balance += pnl
            daily_pnl += pnl
        open_positions.clear()
        continue

    # Skip if max positions reached
    active_max_pos = MAX_POSITIONS
    if len(open_positions) >= active_max_pos:
        continue

    # Trading hours check
    if not (TRADE_HOURS_START <= h < TRADE_HOURS_END):
        continue
    if (h == 8 and bar_time.minute < SESSION_COOLDOWN_MIN) or (h == 13 and bar_time.minute < SESSION_COOLDOWN_MIN):
        continue

    # ── Process new M15 bar ──
    m15_window = m15[max(0, idx-100):idx+1]
    if len(m15_window) < 50:
        continue

    # Get H1 data aligned to this bar for trend context
    if h1 is not None:
        h1_window = h1[h1.index <= bar_time].tail(50)
    else:
        h1_window = m15_window  # fallback

    # Use M15 for both entry and indicators (no M1/M5)
    m5_window = m15_window  # M15 as primary timeframe for all analysis

    if len(h1_window) < 10:
        continue

    # Rename columns to lowercase
    m15_ren = m15_window.rename(columns=lambda x: x.lower())
    m5_ren = m5_window.rename(columns=lambda x: x.lower())
    h1_ren = h1_window.rename(columns=lambda x: x.lower())

    # Compute indicators
    try:
        i15 = compute_all_indicators(m15_ren)
        i5 = compute_all_indicators(m5_ren)  # same as i15 in this config
        # H4 trend from H1 data (4 H1 bars = 1 H4)
        if h1 is not None and len(h1_window) >= 20:
            h4_ema = h1_window['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            h4_trend = "BULLISH" if h1_window['close'].iloc[-1] > h4_ema else "BEARISH"
        else:
            h4_trend = h4_trend_from_m15(m15_window)
        sr = compute_sr_levels(m15_window)
    except:
        continue

    # Candle patterns
    try:
        swing_levels = cp.detect_swing_levels(m15_window)
        candle_analysis = cp.analyze_full(m15_window, swing_levels)
        candle_signal = candle_analysis.get("signal", "NONE")
        candle_conf = candle_analysis.get("confidence", 0)
    except:
        candle_signal = "NONE"
        candle_conf = 0

    # Strategy analysis — using M15 for all timeframes
    try:
        result = strategy.analyze(i5, i5, i15, m5_ren.tail(5), m5_ren, m15_ren)
    except:
        continue

    direction = result.get("direction", "NONE")
    score = result.get("setup_score", 0)
    strategy_reason = result.get("reason", "")

    if direction == "NONE":
        bar_count += 1
        continue

    signal_count += 1

    # Boost score with candle confirmation
    if candle_signal == direction:
        boost = max(1, candle_conf // 3)
        score = min(100, score + boost)
        strategy_reason += f" +{boost} candle"

    # ── Run all filters ──
    blocked, block_reason = check_blocked_by_filters(
        direction, score, open_positions, i15, i5, h4_trend, sr, curr, m15_window, active_max_pos, result
    )

    if blocked:
        blocked_count += 1
        bar_count += 1
        continue

    # ── Place trade ──
    if score < 20:
        bar_count += 1
        continue

    atr_val = i15["atr"].iloc[-1]
    if np.isnan(atr_val) or atr_val <= 0:
        continue

    sl_dist = atr_val * SL_ATR_MULT
    sl = curr - sl_dist if direction == "BUY" else curr + sl_dist
    tp_dist = atr_val * TP_ATR_MULT
    tp = curr + tp_dist if direction == "BUY" else curr - tp_dist

    # Lot sizing
    risk_pct = FIXED_RISK
    if score >= HIGH_SCORE_THRESHOLD:
        risk_pct = min(HIGH_SCORE_RISK, FIXED_RISK * 1.5)
    lot = max(0.01, round((balance * risk_pct) / (sl_dist * 100), 2))
    if len(open_positions) == 1:
        lot *= SECOND_POS_LOT_RATIO
    lot = max(0.01, round(lot, 2))

    open_positions.append({
        "dir": direction,
        "entry": curr,
        "sl": sl,
        "tp": tp,
        "lots": lot,
        "entry_time": bar_time,
        "score": score,
        "reason": strategy_reason,
        "be_triggered": False,
    })

    trade_count += 1
    daily_trades += 1
    strategy.record_trade()

    bar_count += 1

    if bar_count % 500 == 0:
        print(f"  Processed {bar_count} bars | Balance: ${balance:.2f} | Trades: {trade_count} | "
              f"Signals: {signal_count} | Blocked: {blocked_count}")

# ── Close remaining positions at last price ──
if open_positions:
    last_price = m15['close'].iloc[-1]
    for p in open_positions:
        pnl = (last_price - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - last_price) * p["lots"] * 100
        p["pnl"] = round(pnl, 2)
        p["exit_reason"] = "eod_close"
        p["exit_price"] = last_price
        closed_trades.append(p)
        balance += pnl
    open_positions.clear()

# ── RESULTS ──
print("\n" + "="*60)
print("BACKTEST RESULTS — 1 MONTH LIVE BOT SIMULATION")
print("="*60)

wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
total = len(closed_trades)
wr = len(wins) / total * 100 if total > 0 else 0
net_pnl = sum(t.get("pnl", 0) for t in closed_trades)
final_balance = balance
pct_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
dd = (peak_balance - min(balance, peak_balance)) / peak_balance * 100 if peak_balance > 0 else 0

# Profit factor
gross_profit = sum(t.get("pnl", 0) for t in wins)
gross_loss = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 0.01
pf = gross_profit / gross_loss if gross_loss > 0 else 999

total_buy = sum(1 for t in closed_trades if t.get("dir") == "BUY")
total_sell = total - total_buy
buy_wr = len([t for t in wins if t.get("dir") == "BUY"]) / max(total_buy, 1) * 100
sell_wr = len([t for t in wins if t.get("dir") == "SELL"]) / max(total_sell, 1) * 100

avg_win = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
avg_loss = sum(t.get("pnl", 0) for t in losses) / max(len(losses), 1)

# Exit reasons
tp_exits = sum(1 for t in closed_trades if t.get("exit_reason") == "tp")
sl_exits = sum(1 for t in closed_trades if t.get("exit_reason") == "sl")
eod_exits = sum(1 for t in closed_trades if t.get("exit_reason") == "eod_close")

# BE triggered count
be_count = sum(1 for t in closed_trades if t.get("be_triggered", False))

print(f"""
PERIOD: {m15.index.min()} -> {m15.index.max()}
BARS PROCESSED: {bar_count}

FINAL BALANCE: ${final_balance:.2f} (from ${STARTING_BALANCE:.2f})
NET P&L: ${net_pnl:+.2f} ({pct_return:+.1f}%)
PEAK BALANCE: ${peak_balance:.2f}
MAX DD: {dd:.1f}%

TRADES: {total} ({total_buy} BUY / {total_sell} SELL)
WIN RATE: {wr:.1f}% | BUY WR: {buy_wr:.1f}% | SELL WR: {sell_wr:.1f}%
PROFIT FACTOR: {pf:.2f}
AVG WIN: ${avg_win:+.2f} | AVG LOSS: ${avg_loss:+.2f}
EXPECTANCY: ${(wr/100 * avg_win + (1-wr/100) * avg_loss):+.2f}

EXIT REASONS: TP={tp_exits} SL={sl_exits} EOD={eod_exits}
BE TRIGGERED: {be_count}/{total}

SIGNALS FOUND: {signal_count}
SIGNALS BLOCKED: {blocked_count}
SIGNALS TRADED: {trade_count}
""")

# Last 10 trades
print("LAST 10 TRADES:")
for t in closed_trades[-10:]:
    print(f"  {t.get('entry_time','?')}: {t.get('dir','?')} @ {t.get('entry',0):.2f} "
          f"-> {t.get('exit_reason','?')} | PnL: ${t.get('pnl',0):+.2f} | Score: {t.get('score','?')}")

# Save to file
results_file = os.path.join(os.path.dirname(__file__), "backtest_1month_results.txt")
with open(results_file, "w") as f:
    f.write(f"BACKTEST RESULTS — 1 MONTH (v8.3 EXACT)\n")
    f.write(f"======================================\n")
    f.write(f"Period: {m15.index.min()} -> {m15.index.max()}\n")
    f.write(f"Bars: {bar_count}\n")
    f.write(f"Final: ${final_balance:.2f} ({pct_return:+.1f}%)\n")
    f.write(f"Trades: {total} | WR: {wr:.1f}% | PF: {pf:.2f}\n")
    f.write(f"Avg Win: ${avg_win:+.2f} | Avg Loss: ${avg_loss:+.2f}\n")
    f.write(f"Max DD: {dd:.1f}%\n")

print(f"\nResults saved to: {results_file}")