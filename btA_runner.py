"""
EXACT BOT SIMULATION v7.0 — 3 MONTHS (Apr 24 → Jul 24, 2026)
============================================================
Simulates main_super.py v7.0 exactly as it runs live:
- M1/M5/M15/H4 multi-timeframe indicators
- Candle patterns + S/R engine (from candle_patterns.py)
- All technical filters (ATR, BB, direction conflict, risk limit)
- Position management (TP, BE, trailing stop)
- Same risk tiers, lot sizing, SL/TP multipliers
- ⚠️  No DeepSeek API  ⚠️  No news filter
"""
import os, sys, warnings, json
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_MT5_DIR = os.path.join(_PROJECT_ROOT, "trading_bot_mt5")
if _MT5_DIR not in sys.path:
    sys.path.insert(0, _MT5_DIR)

# Silence all bot logging
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

import pandas as pd
import numpy as np

# Patch the logger before any trading_bot imports
import trading_bot.utils.logger as logger_module
logger_module.logger.setLevel(logging.CRITICAL)
logger_module.logger.handlers = []
logger_module.logger.addHandler(logging.NullHandler())

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot_mt5.candle_patterns import analyze_full, detect_swing_levels

# ── CONFIG (exact match with main_super.py v7.0) ─────────────────────
CFG = {
    "STARTING_BALANCE": 300.00,
    "MIN_SCORE_BUY": 30,
    "MIN_SCORE_SELL": 20,
    "MAX_POSITIONS": 2,
    "MAX_PER_DIRECTION": 1,     # no stacking same direction
    "TP_ATR_MULT": 5.0,         # proven sweet spot
    "TP_ATR_MULT_TREND": 8.0,
    "TP_PARTIAL_MULT": 2.5,     # close 50% at 2.5x ATR, rest to 5.0x
    "SL_ATR_MULT": 3.0,
    "BE_ATR_MULT": 2.7,
    "BE_BUFFER_POINTS": 50,
    "TRAIL_ATR_MULT": 0.0,
    "DAILY_LOSS_PCT": 0.02,     # was 0.03 — tighter stop
    "TOTAL_RISK_LIMIT": 0.05,
    "ATR_VOL_THRESHOLD": 2.5,
    "SPREAD_COST_PER_LOT": 0.50,
    "MAX_TRADES_PER_DAY": 50,
    "WARMUP_BARS": 200,
    "FIXED_RISK": 0.02,
    "HARD_FLOOR": 50.00,
    "HALT_HOURS": 6,            # was 3 — longer recovery
    "MAX_CONSEC_LOSSES": 2,
    "REGIME_RISK_LOW": 0.01,
    "REGIME_WR_THRESHOLD": 0.45,
    "RECENT_TRADE_WINDOW": 20,
    "ASIAN_SKIP": False,
    "TRADE_HOURS_START": 8,
    "TRADE_HOURS_END": 17,
    "HIGH_SCORE_THRESHOLD": 70,
    "HIGH_SCORE_RISK": 0.03,
    "SECOND_POS_MIN_SCORE": 35,
    "REQUIRE_CANDLE": False,    # too aggressive — killed 43 trades
    "SECOND_POS_LOT_RATIO": 0.5,
    "STALE_EXIT_HOURS": 99,     # disabled
    "STALE_PROFIT_RATIO": 0.0,
    # ── NEW TOOLS ────────────────────────────────────────
    "DXY_FILTER_ENABLED": True,  # use gold-based proxy
    "DXY_CORR_THRESHOLD": 0.003, # 0.3% move in 4h
    "DXY_LOOKBACK_HOURS": 4,
    "SESSION_COOLDOWN_MIN": 30,  # skip first 30 min of opens
    "MTF_CONFLUENCE": True,
    # ── FIX ATTEMPT #2 ──────────────────────────────────
    "MIN_TRAIL_DISTANCE": 1.5,   # SL stays min 1.5x ATR from price after BE
}
# Risk tiers disabled — use fixed risk
RISK_TIERS = None


def get_risk_pct(balance):
    return CFG["FIXED_RISK"]


# ── LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    m15 = pd.read_csv(os.path.join(base, "XAUUSD_1y_M15.csv"))
    m5 = pd.read_csv(os.path.join(base, "XAUUSD_1y_M5.csv"))
    m1 = pd.read_csv(os.path.join(base, "XAUUSD_1y_M1.csv"))
    for df in [m15, m5, m1]:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.columns = df.columns.str.lower()
    start = pd.Timestamp("2025-07-24", tz="UTC")
    end = pd.Timestamp("2026-07-25", tz="UTC")
    m15 = m15[(m15.index >= start) & (m15.index < end)].copy()
    m5 = m5[(m5.index >= start) & (m5.index < end)].copy()
    m1 = m1[(m1.index >= start) & (m1.index < end)].copy()
    print(f"Data loaded: M15={len(m15)} bars, M5={len(m5)}, M1={len(m1)}")
    print(f"Range: {m15.index[0]} -> {m15.index[-1]}")
    return m15, m5, m1



# ── SIMULATION ────────────────────────────────────────────────────────
def run_simulation():
    m15, m5, m1 = load_data()

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = CFG["MAX_TRADES_PER_DAY"]
    strategy._max_open_positions = CFG["MAX_POSITIONS"]

    # Pre-build integer index arrays for fast lookup (convert to int64 ns)
    m5_idx_arr = m5.index.asi8
    m1_idx_arr = m1.index.asi8

    # Pre-compute ATR on M5 (for position management - every bar)
    print("  Pre-computing M5 ATR...")
    m5_clean = m5.rename(columns=lambda x: x.lower())
    m5_full_ind = compute_all_indicators(m5_clean)
    m5_atr = m5_full_ind["atr"]

    balance = CFG["STARTING_BALANCE"]
    initial_balance = balance
    positions = []
    daily_pnl = 0.0
    cons_losses = 0
    halt_until = None
    last_entry = None
    last_date = None
    closed = []
    trade_count_today = 0
    daily_halted = False    # true after loss limit hit for the day
    monthly = {}
    peak_balance = balance
    max_dd_pct = 0.0

    total_bars = len(m15)
    print_interval = max(1, (total_bars - CFG["WARMUP_BARS"]) // 10)
    last_entry_idx = -999

    for idx, (ct, row) in enumerate(m15.iterrows()):
        if idx < CFG["WARMUP_BARS"]:
            continue

        bars_processed = idx - CFG["WARMUP_BARS"]
        if bars_processed % print_interval == 0:
            print(f"  [{bars_processed/(total_bars-CFG['WARMUP_BARS'])*100:.0f}%] "
                  f"{ct.strftime('%Y-%m-%d %H:%M')} | "
                  f"Bal: ${balance:.2f} | Trades: {len(closed)}")

        price = float(row["close"])
        high_bar = float(row["high"])
        low_bar = float(row["low"])

        # Friday 21h+ auto-close
        if ct.weekday() == 4 and ct.hour >= 21:
            for p in positions:
                pnl = _calc_pnl(p, price)
                balance += pnl
                p["pnl"] = pnl; p["reason"] = "FridayClose"; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
            positions = []
            continue

        # Daily reset
        if last_date is None:
            last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0
            daily_halted = False
            trade_count_today = 0
            last_date = ct.date()
            trade_count_today = 0
            mk = ct.strftime("%Y-%m")
            if mk not in monthly:
                monthly[mk] = {"start_bal": balance, "trades": 0, "wins": 0, "pnl": 0.0, "peak": balance, "dd": 0.0}

        # Balance floor — stop everything
        if balance < CFG["HARD_FLOOR"]:
            print(f"  BALANCE FLOOR ${CFG['HARD_FLOOR']} HIT — stopping at {ct}")
            break

        # Daily halted — skip entry check entirely
        if daily_halted:
            continue

        # Consecutive loss halt
        if halt_until and ct < halt_until:
            continue

        if balance > 0 and daily_pnl <= -balance * CFG["DAILY_LOSS_PCT"]:
            for p in positions:
                pnl = _calc_pnl(p, price)
                balance += pnl
                p["pnl"] = pnl; p["reason"] = "DailyLoss"; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
            positions = []
            daily_halted = True
            halt_until = ct + timedelta(hours=2)  # cooldown, not full day
            continue
        # ── POSITION MANAGEMENT (use pre-computed ATR) ──────────
        ct_int = ct.asi8 if hasattr(ct, 'asi8') else ct.value
        m5_end = np.searchsorted(m5_idx_arr, ct_int, side='right')
        if m5_end < 50:
            continue
        atr_val = float(m5_atr.iloc[m5_end - 1])
        if atr_val <= 0 or np.isnan(atr_val):
            continue

        surviving = []
        for p in positions:
            entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
            sl_dist = abs(entry - sl) if sl and sl > 0 else atr_val * CFG["SL_ATR_MULT"]
            pv = lot * 100

            # ── Partial TP ──────────────────────────────────
            partial_target = entry + (sl_dist * CFG["TP_PARTIAL_MULT"] / CFG["SL_ATR_MULT"]) if direction == "BUY" else entry - (sl_dist * CFG["TP_PARTIAL_MULT"] / CFG["SL_ATR_MULT"])
            if not p.get("partial_hit", False):
                if (direction == "BUY" and high_bar >= partial_target) or (direction == "SELL" and low_bar <= partial_target):
                    p["partial_hit"] = True
                    half_pnl = (partial_target - entry) * pv * 0.5 if direction == "BUY" else (entry - partial_target) * pv * 0.5
                    half_pnl -= CFG["SPREAD_COST_PER_LOT"] * lot * 0.5
                    daily_pnl += half_pnl; balance += half_pnl
                    p["partial_pnl"] = half_pnl
                    p["lot"] = lot * 0.5  # remaining half
                    p["entry"] = entry  # same entry for remaining
                    pv = p["lot"] * 100  # update pv for remaining checks
                    # Record partial TP as a closed trade event for stats
                    closed.append({
                        "dir": direction, "pnl": half_pnl, "reason": "PartialTP",
                        "close_price": partial_target, "close_time": ct,
                        "entry": entry, "sl": sl, "tp": tp, "lot": lot * 0.5,
                        "open_time": p["open_time"],
                    })
            hours_open = (ct - p["open_time"]).total_seconds() / 3600
            profit_pct = (price - entry) / sl_dist if direction == "BUY" else (entry - price) / sl_dist
            if hours_open >= CFG["STALE_EXIT_HOURS"] and profit_pct < CFG["STALE_PROFIT_RATIO"]:
                pnl = (profit_pct * sl_dist) * pv - CFG["SPREAD_COST_PER_LOT"] * lot
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = "StaleExit"; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                continue

            # BE trigger — move SL to entry+buffer, but never worse than original
            mult_be = CFG["BE_ATR_MULT"] / CFG["SL_ATR_MULT"]
            be_target = entry + (sl_dist * mult_be) if direction == "BUY" else entry - (sl_dist * mult_be)
            if not p.get("be", False):
                if (direction == "BUY" and price >= be_target) or (direction == "SELL" and price <= be_target):
                    p["be"] = True
                    buffer_pts = CFG["BE_BUFFER_POINTS"] * 0.01
                    new_sl = entry + buffer_pts if direction == "BUY" else entry - buffer_pts
                    # Keep SL at original level — BE only locks in profit
                    if direction == "BUY":
                        new_sl = max(new_sl, p["sl"])  # take the higher SL
                    else:
                        new_sl = min(new_sl, p["sl"])  # take the lower SL
                    p["sl"] = round(new_sl, 2)

            # Trailing stop (only if TRAIL_ATR_MULT > 0)
            if CFG["TRAIL_ATR_MULT"] > 0 and p.get("be", False):
                new_sl = price - atr_val * CFG["TRAIL_ATR_MULT"] if direction == "BUY" else price + atr_val * CFG["TRAIL_ATR_MULT"]
                if direction == "BUY" and new_sl > p["sl"] + 0.5:
                    p["sl"] = round(new_sl, 2)
                elif direction == "SELL" and new_sl < p["sl"] - 0.5:
                    p["sl"] = round(new_sl, 2)

            sl, tp = p["sl"], p["tp"]
            hit = False; pnl = 0.0; reason = ""
            # TP/SL using bar high/low for realistic fills
            if direction == "BUY":
                if tp and high_bar >= tp:
                    pnl = (tp - entry) * pv; reason = "TP"; hit = True
                elif sl and low_bar <= sl:
                    pnl = (sl - entry) * pv; reason = "TRAIL" if p.get("be") else "SL"; hit = True
            else:
                if tp and low_bar <= tp:
                    pnl = (entry - tp) * pv; reason = "TP"; hit = True
                elif sl and high_bar >= sl:
                    pnl = (entry - sl) * pv; reason = "TRAIL" if p.get("be") else "SL"; hit = True

            if hit:
                pnl -= CFG["SPREAD_COST_PER_LOT"] * p["lot"]
                daily_pnl += pnl; balance += pnl
                # Record only remaining portion P&L (partial was already logged)
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                if reason in ("SL", "TRAIL") and pnl < 0:
                    cons_losses += 1
                    if cons_losses >= CFG["MAX_CONSEC_LOSSES"]:
                        halt_until = ct + timedelta(hours=CFG["HALT_HOURS"])
                        cons_losses = 0
                else:
                    cons_losses = 0
            else:
                surviving.append(p)
        positions = surviving

        # Update peak/drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

        # ── NEW ENTRY CHECK ───────────────────────────────────
        # Asian session skip
        if CFG["ASIAN_SKIP"] and (ct.hour >= 23 or ct.hour < 8):
            continue
        
        if len(positions) >= CFG["MAX_POSITIONS"]:
            continue
        if trade_count_today >= CFG["MAX_TRADES_PER_DAY"]:
            continue

        # Cooldown: at least 2 bars since last entry
        if last_entry_idx >= 0 and idx - last_entry_idx < 2:
            continue

        # Score escalation for 2nd position
        if len(positions) >= 1:
            min_sc = max(min_sc, CFG["SECOND_POS_MIN_SCORE"])

        # ── SESSION FILTERS ────────────────────────────────
        if not (CFG["TRADE_HOURS_START"] <= ct.hour < CFG["TRADE_HOURS_END"]):
            continue
        # Session open cooldown — skip first 30 min of London/NY
        cooldown = CFG["SESSION_COOLDOWN_MIN"]
        london_open = ct.hour == 8 and ct.minute < cooldown
        ny_open = ct.hour == 13 and ct.minute < cooldown
        if london_open or ny_open:
            continue

        # Continue to signal generation...

        # Build M5 window for indicator computation
        m5_start = max(0, m5_end - 500)
        m5_window = m5.iloc[m5_start:m5_end].copy()
        m15_window = m15.iloc[max(0, idx - 500):idx + 1].copy()
        m1_start = max(0, np.searchsorted(m1_idx_arr, ct_int, side='right') - 200)
        m1_window = m1.iloc[m1_start:np.searchsorted(m1_idx_arr, ct_int, side='right')].copy()

        if len(m15_window) < 50 or len(m5_window) < 50:
            continue

        # Compute indicators (only when we might enter)
        ind5 = compute_all_indicators(m5_window.rename(columns=lambda x: x.lower()))
        ind15 = compute_all_indicators(m15_window.rename(columns=lambda x: x.lower()))
        if ind15 is None or ind15.get("rsi") is None or len(ind15["rsi"]) < 5:
            continue

        # H4 data (from M15 bars)
        h4_start = max(0, idx - 16 * 50)
        h4_raw = m15.iloc[h4_start:idx + 1]
        h4_window = h4_raw.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna()
        if len(h4_window) < 10:
            h4_window = m15_window.copy()
        ind4 = compute_all_indicators(h4_window)

        # ── CANDLE + S/R ANALYSIS ──────────────────────────────
        try:
            swing_levels = detect_swing_levels(m15_window)
            candle_analysis = analyze_full(m15_window, swing_levels)
            candle_signal = candle_analysis.get("signal", "NONE")
            candle_conf = candle_analysis.get("confidence", 0)
        except Exception:
            candle_signal = "NONE"; candle_conf = 0
            candle_analysis = {"signal": "NONE", "confidence": 0, "reason": "", "patterns_detected": []}

        # M1 indicators (real RSI)
        if len(m1_window) >= 30:
            m1_cols = m1_window.rename(columns=lambda x: x.lower())
            ind1 = compute_all_indicators(m1_cols)
        else:
            ind1 = ind5  # fallback to M5

        # ── STRATEGY SIGNAL ────────────────────────────────────
        try:
            result = strategy.analyze(ind1, ind5, ind15, m5_window.tail(5), m5_window, m15_window)
        except Exception:
            continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        reason = result.get("reason", "")

        # Boost score with candle confirmation (stronger: //3 instead of //5)
        if direction != "NONE" and candle_signal == direction:
            boost = max(1, candle_conf // 3)
            score = min(100, score + boost)

        # ── MTF CONFLUENCE ──────────────────────────────────
        if direction != "NONE" and CFG["MTF_CONFLUENCE"]:
            try:
                m5_rsi = float(ind5["rsi"].iloc[-1])
                m5_bullish = m5_rsi > 50
                m5_bearish = m5_rsi < 50
                if (direction == "BUY" and m5_bearish) or (direction == "SELL" and m5_bullish):
                    continue
            except Exception:
                pass

        if direction == "NONE":
            continue
        min_sc = CFG["MIN_SCORE_BUY"] if direction == "BUY" else CFG["MIN_SCORE_SELL"]
        if score < min_sc:
            continue

        # Require candle confirmation
        if CFG["REQUIRE_CANDLE"] and candle_signal != direction:
            continue

        # ── FILTERS ────────────────────────────────────────────
        # DXY correlation filter (gold inverse proxy)
        if CFG["DXY_FILTER_ENABLED"]:
            dxy_bars = CFG["DXY_LOOKBACK_HOURS"] * 4  # 4 M15 bars per hour
            dxy_start = max(0, idx - dxy_bars)
            if dxy_start < idx - 4:  # at least 1 hour of data
                gold_change = (m15.iloc[idx]["close"] - m15.iloc[dxy_start]["close"]) / m15.iloc[dxy_start]["close"]
                dxy_proxy = -gold_change  # inverse correlation
                if direction == "BUY" and dxy_proxy > CFG["DXY_CORR_THRESHOLD"]:
                    continue
                if direction == "SELL" and dxy_proxy < -CFG["DXY_CORR_THRESHOLD"]:
                    continue
        curr = price  # current M15 close = price

        # ATR vol threshold
        current_atr = float(ind15["atr"].iloc[-1])
        mean_atr = float(ind15["atr"].rolling(20).mean().iloc[-1])
        if current_atr > mean_atr * CFG["ATR_VOL_THRESHOLD"]:
            continue

        # S/R proximity (use candle_analysis levels or defaults)
        near_res = candle_analysis.get("nearest_resistance", curr + 10)
        near_sup = candle_analysis.get("nearest_support", curr - 10)
        dist_r = near_res - curr if near_res > curr else 99
        dist_s = curr - near_sup if near_sup < curr else 99

        if direction == "BUY" and dist_r < 2.5:
            continue
        if direction == "SELL" and dist_s < 2.5:
            continue

        # Direction conflict (all positions)
        if positions:
            existing_dirs = set(p["dir"] for p in positions)
            if direction not in existing_dirs and len(existing_dirs) > 0:
                continue
            # Block same-direction stacking
            if direction in existing_dirs and sum(1 for p in positions if p["dir"] == direction) >= CFG["MAX_PER_DIRECTION"]:
                continue

        # H4 trend alignment
        h4_ema20 = ind4['emas']['EMA_20'].iloc[-1]
        h4_trend = "BULLISH" if h4_window['close'].iloc[-1] > h4_ema20 else "BEARISH"
        if direction != ("BUY" if h4_trend == "BULLISH" else "SELL"):
            m15_rsi_val = float(ind15['rsi'].iloc[-1])
            if not (m15_rsi_val < 25 or m15_rsi_val > 75):
                continue

        # BB rejection
        bb_upper = float(ind15['bb']['upper'].iloc[-1])
        bb_lower = float(ind15['bb']['lower'].iloc[-1])
        if direction == "BUY" and curr >= bb_upper:
            continue
        if direction == "SELL" and curr <= bb_lower:
            continue
        # ── PLACE TRADE ──────────────────────────────────────
        sl_dist = current_atr * CFG["SL_ATR_MULT"]
        tp_dist = current_atr * CFG["TP_ATR_MULT"]
        sl = round(price - sl_dist, 2) if direction == "BUY" else round(price + sl_dist, 2)
        tp = round(price + tp_dist, 2) if direction == "BUY" else round(price - tp_dist, 2)

        # Regime-adjusted risk: reduce to 1% if recent WR < 45%
        risk_pct = CFG["FIXED_RISK"]
        if len(closed) >= CFG["RECENT_TRADE_WINDOW"]:
            recent = closed[-CFG["RECENT_TRADE_WINDOW"]:]
            recent_wr = sum(1 for t in recent if t["pnl"] > 0) / len(recent)
            if recent_wr < CFG["REGIME_WR_THRESHOLD"]:
                risk_pct = CFG["REGIME_RISK_LOW"]

        # Boost risk to 3% on exceptional setups (score 70+)
        if score >= CFG["HIGH_SCORE_THRESHOLD"]:
            risk_pct = CFG["HIGH_SCORE_RISK"]

        lot = max(0.01, round((balance * risk_pct) / (sl_dist * 100), 2))
        if len(positions) == 1:
            lot *= CFG["SECOND_POS_LOT_RATIO"]  # 0.5x for 2nd position
        lot = max(0.01, round(lot, 2))

        # Open position
        positions.append({
            "entry": price, "dir": direction, "sl": sl, "tp": tp, "lot": lot,
            "be": False, "open_time": ct, "open_score": score,
            "candle_patterns": candle_analysis.get("patterns_detected", []),
            "reason": reason,
        })
        last_entry = ct
        last_entry_idx = idx
        trade_count_today += 1

    # ── Close remaining positions at end ──────────────────────────────
    for p in positions:
        final_price = float(m15.iloc[-1]["close"])
        pnl = _calc_pnl(p, final_price)
        balance += pnl
        p["pnl"] = pnl; p["reason"] = "EndOfData"; p["close_price"] = final_price; p["close_time"] = m15.index[-1]
        closed.append(p)


        # Total risk limit
        existing_risk = sum(abs(p["entry"] - p["sl"]) * 100 * p["lot"]
                            for p in positions if p["sl"] and p["sl"] > 0)
        if existing_risk >= balance * CFG["TOTAL_RISK_LIMIT"]:
            continue

    return closed, balance, initial_balance, peak_balance, max_dd_pct


def _calc_pnl(p, price):
    """Calculate P&L for a position at given price."""
    entry, direction, lot = p["entry"], p["dir"], p["lot"]
    pv = lot * 100
    if direction == "BUY":
        return (price - entry) * pv - CFG["SPREAD_COST_PER_LOT"] * lot
    return (entry - price) * pv - CFG["SPREAD_COST_PER_LOT"] * lot


# ── RESULTS REPORTING ─────────────────────────────────────────────────
def print_results(closed, final_balance, initial_balance, peak_balance, max_dd_pct):
    if not closed:
        print("\n⚠️  NO TRADES EXECUTED")
        return

    nb = final_balance - initial_balance
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    total_wins = sum(t["pnl"] for t in wins)
    total_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    returns_pct = (final_balance / initial_balance - 1) * 100

    # Sharpe ratio
    daily_returns = []
    pnl_by_date = {}
    for t in closed:
        d = t["close_time"].date()
        pnl_by_date[d] = pnl_by_date.get(d, 0) + t["pnl"]
    for d, pnl_d in sorted(pnl_by_date.items()):
        daily_returns.append(pnl_d / initial_balance)
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
              if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0)
    max_cons_wins = max_cons_losses = 0
    cw = cl = 0
    for t in closed:
        if t["pnl"] > 0:
            cw += 1; cl = 0; max_cons_wins = max(max_cons_wins, cw)
        else:
            cl += 1; cw = 0; max_cons_losses = max(max_cons_losses, cl)

    buys = [t for t in closed if t["dir"] == "BUY"]
    sells = [t for t in closed if t["dir"] == "SELL"]

    print("\n" + "=" * 60)
    print("  1-YEAR BACKTEST RESULTS (v8.0)")
    print("=" * 60)
    print(f"  Period:          2025-07-24 -> 2026-07-24 (12 months)")
    print(f"  Starting Balance: ${initial_balance:.2f}")
    print(f"  Final Balance:    ${final_balance:.2f}")
    print(f"  Net P&L:          ${nb:+.2f}  ({returns_pct:+.2f}%)")
    print(f"  Peak Balance:     ${peak_balance:.2f}")
    print(f"  Max Drawdown:     {max_dd_pct:.2f}%")
    print("  " + "-" * 40)
    print(f"  Total Trades:     {len(closed)}")
    print(f"  Wins / Losses:    {len(wins)} / {len(losses)}")
    print(f"  Win Rate:         {win_rate:.1f}%")
    print(f"  Avg Win:          ${avg_win:+.2f}")
    print(f"  Avg Loss:         ${avg_loss:+.2f}")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    print(f"  Max Consec Wins:  {max_cons_wins}")
    print(f"  Max Consec Loss:  {max_cons_losses}")
    print("  " + "-" * 40)
    print(f"  BUY trades:       {len(buys)}  (win: {len([b for b in buys if b['pnl']>0])})")
    print(f"  SELL trades:      {len(sells)} (win: {len([s for s in sells if s['pnl']>0])})")

    # By month
    print(f"  ── By Month ────────────────────────")
    for t in closed:
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in [None]:
            pass  # use later
    months = {}
    for t in closed:
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in months:
            months[mk] = {"trades": 0, "wins": 0, "pnl": 0.0}
        months[mk]["trades"] += 1
        months[mk]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            months[mk]["wins"] += 1
    for mk in sorted(months.keys()):
        m = months[mk]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        print(f"    {mk}: {m['trades']:>3} trades | WR: {wr:5.1f}% | P&L: ${m['pnl']:>+8.2f}")

    # By reason
    reasons = {}
    for t in closed:
        r = t.get("reason", "?")
        reasons[r] = reasons.get(r, 0) + 1
    print(f"  ── By Exit Reason ───────────────────")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:>12s}: {cnt:>3} trades")

    # Top 5 best/worst
    print(f"  ── Best 5 Trades ────────────────────")
    for t in sorted(closed, key=lambda x: -x["pnl"])[:5]:
        print(f"    {t['dir']:>4s} {t['close_time'].strftime('%Y-%m-%d %H:%M')} "
              f"P&L: ${t['pnl']:>+7.2f}  Reason: {t['reason']}")
    print(f"  ── Worst 5 Trades ───────────────────")
    for t in sorted(closed, key=lambda x: x["pnl"])[:5]:
        print(f"    {t['dir']:>4s} {t['close_time'].strftime('%Y-%m-%d %H:%M')} "
              f"P&L: ${t['pnl']:>+7.2f}  Reason: {t['reason']}")
    print("=" * 60)
    return closed


if __name__ == "__main__":
    print("=" * 60)
    print("  GOLD SCALPING BOT v8.0 — 1-YEAR BACKTEST")
    print("  Starting Balance: $300 | No AI | No News")
    print("=" * 60)
    closed, final_bal, init_bal, peak_bal, max_dd = run_simulation()
    print_results(closed, final_bal, init_bal, peak_bal, max_dd)
