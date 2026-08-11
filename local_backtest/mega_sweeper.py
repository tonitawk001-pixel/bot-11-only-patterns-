"""
MEGA SWEEPER — Ultimate Parameter Optimization
===============================================
Tests ALL combinations:
  - 3 analysis timeframes: M1 / M5 / M15
  - Real M1 data (fixing the empty_m1 bug)
  - All trading parameters
  - Continuous position management (every candle)

Goal: Find absolute best performer with ≤30% drawdown

Data available:
  - M15: 60 days (May-Jul 2026) for M15/M5 modes
  - M1:  7 days (Jul 14-21 2026) for M1 mode only

Run: python local_backtest/mega_sweeper.py
"""

import os, sys, warnings, time
warnings.filterwarnings("ignore")

os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

# Silence logger
import trading_bot.utils.logger as logger_module
logger_module.logger.setLevel(logging.CRITICAL)
logger_module.logger.handlers = []
logger_module.logger.addHandler(logging.NullHandler())

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STARTING_BALANCE = 304.99
TRADE_HOURS_START = 8
TRADE_HOURS_END = 22
BE_ATR_MULT = 2.0
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
ENTRY_COOLDOWN_MINUTES = 0
DAILY_LOSS_PCT = 0.03
SPREAD_COST_PIP = 0.50
MAX_TRADES_PER_DAY = 50
MAX_DRAWDOWN_LIMIT = 30.0

# ===== PARAMETER GRID =====
MIN_SCORES = [35, 40, 45, 50]
RISK_PCTS = [1.0, 1.5, 2.0, 2.5]
TP_MULTS = [3.5, 5.0, 6.0]
SL_MULTS = [1.0, 1.5, 2.0]
TRAIL_MULTS = [0.3, 0.4, 0.5, 0.7]
MAX_POSS = [1, 2]

# Analysis modes to test
ANALYSIS_MODES = [
    {"name": "M15", "desc": "Analyze on M15 candles (every 15 min)", "base_tf": "M15"},
    {"name": "M5", "desc": "Analyze on M5 candles (every 5 min)", "base_tf": "M5"},
    {"name": "M1", "desc": "Analyze on M1 candles (every 1 min)", "base_tf": "M1"},
]


def in_session(ct):
    return TRADE_HOURS_START <= ct.hour < TRADE_HOURS_END

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


def load_data(mode_name):
    """Load data for given analysis mode."""
    if mode_name == "M1":
        fp = os.path.join(DATA_DIR, "XAUUSD_7d_M1.csv")
        if not os.path.exists(fp):
            print(f"  [!] M1 data not found")
            return None, None, None
        m1 = pd.read_csv(fp)
        m1['Datetime'] = pd.to_datetime(m1['Datetime'], utc=True)
        m1.set_index('Datetime', inplace=True)
        m1.columns = [c.lower() for c in m1.columns]
        m1 = m1[~m1.index.duplicated(keep='last')].dropna()
        m1.sort_index(inplace=True)

        # For M1 mode, we need M5 and M15 data too (for the multi-TF strategy)
        # Resample M1 to M5 and M15
        m5 = m1.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        m15 = m1.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        print(f"    M1: {len(m1)} candles, M5: {len(m5)}, M15: {len(m15)}")
        return m1, m5, m15

    # M5 or M15 mode — use the 60-day data
    m15_fp = os.path.join(DATA_DIR, "XAUUSD_60d_M15.csv")
    m5_fp = os.path.join(DATA_DIR, "XAUUSD_60d_M5.csv")

    if not os.path.exists(m15_fp):
        print(f"  [!] M15 data not found")
        return None, None, None

    m15 = pd.read_csv(m15_fp)
    m15['Datetime'] = pd.to_datetime(m15['Datetime'], utc=True)
    m15.set_index('Datetime', inplace=True)
    m15.columns = [c.lower() for c in m15.columns]
    m15 = m15[~m15.index.duplicated(keep='last')].dropna()
    m15.sort_index(inplace=True)

    m5 = None
    if os.path.exists(m5_fp):
        m5 = pd.read_csv(m5_fp)
        m5['Datetime'] = pd.to_datetime(m5['Datetime'], utc=True)
        m5.set_index('Datetime', inplace=True)
        m5.columns = [c.lower() for c in m5.columns]
        m5 = m5[~m5.index.duplicated(keep='last')].dropna()
        m5.sort_index(inplace=True)

    print(f"    M15: {len(m15)} candles, M5: {len(m5) if m5 is not None else 0}")

    if mode_name == "M5" and m5 is None:
        print(f"  [!] No M5 data for M5 mode, using M15 as fallback")
        return m15, m15, m15

    return None, m5, m15  # No M1 data for M5/M15 modes


def run_backtest(params, m1_df, m5_df, m15_df, analysis_tf):
    """
    Run backtest with given parameters and analysis timeframe.
    
    analysis_tf: 'M1', 'M5', or 'M15' — determines which candles trigger analysis
    """
    min_score = params["min_score"]
    risk_pct = params["risk_pct"] / 100.0
    tp_mult = params["tp_mult"]
    sl_mult = params["sl_mult"]
    trail_mult = params["trail_mult"]
    max_pos = params["max_pos"]
    ADX_TREND_THRESHOLD = 20
    TP_ATR_MULT_TREND = max(tp_mult * 1.4, 5.0)

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = MAX_TRADES_PER_DAY
    strategy._max_open_positions = max_pos

    balance = STARTING_BALANCE
    positions = []
    daily_pnl = 0.0
    cons_losses = 0
    halt_until = None
    last_entry = None
    last_date = None
    closed = []
    trade_count_today = 0
    current_trade_day = None

    # Determine which dataframe drives analysis
    if analysis_tf == "M1":
        df_analysis = m1_df  # Check every 1-minute candle
        df_m5 = m5_df
        df_m15 = m15_df
    elif analysis_tf == "M5":
        df_analysis = m5_df  # Check every 5-minute candle
        df_m5 = m5_df
        df_m15 = m15_df
    else:
        df_analysis = m15_df  # Check every 15-minute candle
        df_m5 = m5_df
        df_m15 = m15_df

    if df_analysis is None or len(df_analysis) < 200:
        return None

    last_processed_time = None

    for i in range(200, len(df_analysis)):
        ct = df_analysis.index[i]
        price = float(df_analysis["close"].iloc[i])

        # CONTINUOUS POSITION MANAGEMENT: Check positions on EVERY cycle
        # regardless of whether we're doing analysis
        if not in_session(ct):
            # Still need to check positions even outside session
            if positions:
                # Check if any positions got hit
                atr_val_local = 0
                try:
                    # Find closest M5 candle for ATR
                    if df_m5 is not None:
                        m5_before = df_m5[df_m5.index <= ct]
                        if len(m5_before) >= 20:
                            m5w = m5_before.tail(100).copy()
                            ind5 = compute_all_indicators(m5w)
                            if ind5 and ind5.get("atr") is not None and len(ind5["atr"]) > 0:
                                atr_val_local = float(ind5["atr"].iloc[-1])
                except:
                    pass
                positions[:] = update_positions(positions, price, atr_val_local, balance, closed, trail_mult)
            continue

        # Friday checks
        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue
        if ct.weekday() == 4 and ct.hour >= 18: continue

        # Daily tracking
        if last_date is None: last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0; current_trade_day = ct.date()
        if current_trade_day is None: current_trade_day = ct.date()
        if ct.date() == current_trade_day: pass
        else: trade_count_today = 0; current_trade_day = ct.date()

        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS_PCT: continue
        if trade_count_today >= MAX_TRADES_PER_DAY: continue

        # Build indicator windows — use M5 for short-term, M15 for long-term
        if df_m5 is not None:
            m5_before = df_m5[df_m5.index <= ct]
            m5_window = m5_before.tail(500).copy()
        else:
            m5_window = df_analysis.iloc[max(0, i-200):i+1].copy()

        if df_m15 is not None:
            m15_before = df_m15[df_m15.index <= ct]
            m15_window = m15_before.tail(500).copy()
        else:
            m15_window = df_analysis.iloc[max(0, i-500):i+1].copy()

        if len(m5_window) < 50 or len(m15_window) < 50: continue

        # Update positions with current ATR
        ind5_temp = compute_all_indicators(m5_window)
        atr_val = 0
        if ind5_temp and ind5_temp.get("atr") is not None and len(ind5_temp["atr"]) > 0:
            atr_val = float(ind5_temp["atr"].iloc[-1])

        positions[:] = update_positions(positions, price, atr_val, balance, closed, trail_mult)

        if cons_losses >= HALT_AFTER_LOSSES and halt_until is None:
            halt_until = ct + timedelta(hours=HALT_HOURS); cons_losses = 0; continue
        if len(positions) >= max_pos: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES: continue

        # ===== SKIP ANALYSIS if not on analysis candle boundary =====
        # For M1: analyze every candle (always on boundary)
        # For M5: analyze on every 5-min boundary (all candles are 5-min)
        # For M15: analyze on every 15-min boundary (all candles are 15-min)
        # All candle-based analysis means we analyze on every candle in df_analysis
        # This is already handled since we iterate df_analysis

        if atr_val < 0.5: continue  # Skip if ATR too low

        # Compute indicators for this candle
        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5: continue

        # ADX
        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except: adx_val = 0

        current_tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else tp_mult

        # Run strategy with REAL M1 data if available
        if m1_df is not None:
            m1_before = m1_df[m1_df.index <= ct]
            m1_window = m1_before.tail(50).copy()
            if len(m1_window) >= 20:
                m1_ind = compute_all_indicators(m1_window)
            else:
                m1_ind = None
        else:
            m1_ind = None

        try:
            if m1_ind is not None and analysis_tf == "M1":
                # REAL M1 data — use actual indicators
                result = strategy.analyze(
                    m1_indicators=m1_ind, m5_indicators=ind5, m15_indicators=ind15,
                    m1_ohlcv=m1_window, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None,
                )
            else:
                # Standard: fake M1 data (current bot behavior)
                empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
                eo = m5_window.tail(20)
                result = strategy.analyze(
                    m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                    m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None,
                )
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < min_score: continue

        # RSI confluence
        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40): continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60): continue
        except: pass

        # EMA200
        closes = m15_window["close"].values
        if len(closes) >= 200:
            ema200 = pd.Series(closes).ewm(200, adjust=False).mean().values
            if len(ema200) >= 10:
                rising = ema200[-1] > ema200[-10]
                if direction == "BUY" and not rising: continue
                if direction == "SELL" and rising: continue

        # SL/TP
        sd = atr_val * sl_mult; td = atr_val * current_tp_mult
        if direction == "BUY": sl = round(price - sd, 2); tp = round(price + td, 2)
        else: sl = round(price + sd, 2); tp = round(price - td, 2)

        # Lot size
        risk_amt = balance * risk_pct
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)
        trade_count_today += 1

        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score, "be_target": be_target, "be": False}
        positions.append(pos); last_entry = ct

    if not closed:
        return {"net_pnl": 0, "max_dd_pct": 0, "trades": 0, "win_rate": 0, "profit_factor": 0, "final_balance": STARTING_BALANCE}

    total_pnl = sum(t["pnl"] for t in closed)
    wins = sum(1 for t in closed if t["pnl"] > 0)
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0]
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0]

    peak = STARTING_BALANCE; maxdd = 0
    eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)

    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0

    return {
        "net_pnl": total_pnl, "max_dd_pct": maxdd, "trades": len(closed),
        "win_rate": win_rate, "profit_factor": pf,
        "final_balance": STARTING_BALANCE + total_pnl,
    }


def update_positions(positions, price, atr_val, balance, closed, trail_mult):
    """Universal position updater — identical to live bot logic."""
    surviving = []
    for p in positions:
        entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
        pv = lot * 100

        if not p.get("be", False) and p.get("be_target"):
            if direction == "BUY" and price >= p["be_target"]: p["be"] = True; p["sl"] = entry
            elif direction == "SELL" and price <= p["be_target"]: p["be"] = True; p["sl"] = entry
        if p.get("be"):
            ns = price - atr_val * trail_mult if direction == "BUY" else price + atr_val * trail_mult
            if direction == "BUY" and ns > sl + 0.5: p["sl"] = round(ns, 2)
            elif direction == "SELL" and ns < sl - 0.5: p["sl"] = round(ns, 2)

        sl, tp = p["sl"], p["tp"]
        hit = False; pnl = 0.0; reason = ""
        if direction == "BUY":
            if tp and price >= tp: pnl = (tp - entry) * pv; reason = "TP"; hit = True
            elif sl and price <= sl: pnl = (sl - entry) * pv; reason = "TRAIL" if sl > entry else "SL"; hit = True
        else:
            if tp and price <= tp: pnl = (entry - tp) * pv; reason = "TP"; hit = True
            elif sl and price >= sl: pnl = (entry - sl) * pv; reason = "TRAIL" if sl < entry else "SL"; hit = True
        if hit:
            pnl -= SPREAD_COST_PIP * lot * 100
            p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price
            closed.append(p)
        else:
            surviving.append(p)
    return surviving


def run_mode(mode_config):
    """Run all parameter combinations for a given analysis mode."""
    print(f"\n{'='*80}")
    print(f"  MODE: {mode_config['name']} — {mode_config['desc']}")
    print(f"{'='*80}")

    m1_df, m5_df, m15_df = load_data(mode_config["name"])
    if mode_config["name"] == "M1" and m1_df is None:
        print("  SKIPPING M1 mode — no data")
        return []
    if mode_config["name"] in ("M5", "M15") and m15_df is None:
        print("  SKIPPING — no M15 data")
        return []

    # Calculate total combos
    total = len(MIN_SCORES) * len(RISK_PCTS) * len(TP_MULTS) * len(SL_MULTS) * len(TRAIL_MULTS) * len(MAX_POSS)
    print(f"  Testing {total} parameter combinations...")

    results = []
    count = 0
    start_time = time.time()

    for ms in MIN_SCORES:
        for rp in RISK_PCTS:
            for tp in TP_MULTS:
                for sl in SL_MULTS:
                    for tr in TRAIL_MULTS:
                        for mp in MAX_POSS:
                            params = {
                                "min_score": ms, "risk_pct": rp, "tp_mult": tp,
                                "sl_mult": sl, "trail_mult": tr, "max_pos": mp,
                            }

                            result = run_backtest(params, m1_df, m5_df, m15_df, mode_config["name"])
                            count += 1

                            if result is not None:
                                results.append({
                                    "mode": mode_config["name"],
                                    "min_score": ms, "risk_pct": rp, "tp_mult": tp,
                                    "sl_mult": sl, "trail_mult": tr, "max_pos": mp,
                                    "net_pnl": result["net_pnl"],
                                    "max_dd": result["max_dd_pct"],
                                    "trades": result["trades"],
                                    "win_rate": result["win_rate"],
                                    "profit_factor": result["profit_factor"],
                                    "final_balance": result["final_balance"],
                                })

                            if count % max(1, total // 10) == 0:
                                pct = count / total * 100
                                elapsed = time.time() - start_time
                                print(f"    {count}/{total} ({pct:.0f}%) | {elapsed:.0f}s", end="\r")

    elapsed = time.time() - start_time
    print(f"\n  Completed {count} tests in {elapsed:.1f}s")

    return results


def print_comparison(all_results):
    """Print best results for each mode and overall winner."""
    print("\n" + "=" * 100)
    print("  🏆 MEGA SWEEPER RESULTS — ALL MODES")
    print("=" * 100)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    if df.empty:
        print("  No results!")
        return None, None

    # Filter ≤30% DD
    qualified = df[df["max_dd"] <= MAX_DRAWDOWN_LIMIT].copy()

    if qualified.empty:
        print("  NO combos within 30% DD limit. Showing top by profit:")
        qualified = df.nlargest(10, "net_pnl").copy()

    # Save all
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mega_sweeper_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  All {len(df)} results saved to: {output_path}")

    # Best per mode
    print(f"\n{'='*100}")
    print(f"  BEST PER MODE (≤30% DD)")
    print(f"{'='*100}")
    print(f"  {'Mode':<6} {'Score':<6} {'Risk%':<6} {'TP':<6} {'SL':<6} {'Trail':<6} {'Pos':<4} {'Net P&L':<10} {'Return%':<10} {'DD%':<7} {'Trades':<7} {'PF':<7}")
    print(f"  {'-'*80}")

    best_overall = None
    best_overall_profit = -999999

    for mode in ["M1", "M5", "M15"]:
        mode_results = qualified[qualified["mode"] == mode]
        if mode_results.empty:
            print(f"  {mode:<6} NO QUALIFIED RESULTS")
            continue
        best = mode_results.sort_values("net_pnl", ascending=False).iloc[0]
        ret = (best["net_pnl"] / STARTING_BALANCE) * 100
        print(f"  {mode:<6} {best['min_score']:<6.0f} {best['risk_pct']:<6.1f} {best['tp_mult']:<6.1f} {best['sl_mult']:<6.1f} {best['trail_mult']:<6.1f} {best['max_pos']:<4.0f} ${best['net_pnl']:<+7.2f} {ret:<+9.1f}% {best['max_dd']:<6.1f}% {best['trades']:<7.0f} {best['profit_factor']:<7.2f}")

        if best["net_pnl"] > best_overall_profit:
            best_overall = best
            best_overall_profit = best["net_pnl"]

    # Overall winner
    if best_overall is not None:
        print(f"\n{'='*80}")
        print(f"  🏆 OVERALL WINNER")
        print(f"{'='*80}")
        ret = (best_overall["net_pnl"] / STARTING_BALANCE) * 100
        print(f"  Mode:             {best_overall['mode']}")
        print(f"  MIN_SCORE:        {best_overall['min_score']:.0f}")
        print(f"  RISK %:           {best_overall['risk_pct']:.1f}%")
        print(f"  TP MULTIPLIER:    {best_overall['tp_mult']:.1f}x")
        print(f"  SL MULTIPLIER:    {best_overall['sl_mult']:.1f}x")
        print(f"  TRAIL MULTIPLIER: {best_overall['trail_mult']:.1f}x")
        print(f"  MAX POSITIONS:    {best_overall['max_pos']:.0f}")
        print(f"")
        print(f"  Net Profit:       ${best_overall['net_pnl']:+.2f}")
        print(f"  Return:           {ret:+.1f}%")
        print(f"  Max Drawdown:     {best_overall['max_dd']:.1f}%")
        print(f"  Trades:           {best_overall['trades']:.0f}")
        print(f"  Profit Factor:    {best_overall['profit_factor']:.2f}")
        print(f"  Final Balance:    ${best_overall['final_balance']:.2f}")
        print(f"  Win Rate:         {best_overall['win_rate']:.1f}%")

    return best_overall, qualified


def main():
    print("=" * 80)
    print("  MEGA SWEEPER v2.0")
    print("  Testing M1, M5, M15 analysis frequencies × all parameters")
    print("  Continuous position management on every cycle")
    print("  Real M1 data (fixing the empty_m1 bug)")
    print("=" * 80)

    all_results = []

    for mode in ANALYSIS_MODES:
        results = run_mode(mode)
        all_results.extend(results)

    best_overall, qualified = print_comparison(all_results)

    # Update live bot if we have results
    if best_overall is not None and qualified is not None:
        best_m15 = qualified[qualified["mode"] == "M15"]
        if not best_m15.empty:
            best15 = best_m15.sort_values("net_pnl", ascending=False).iloc[0]
            print(f"\n{'='*80}")
            print(f"  BEST M15 MODE (for live bot — 60 days data)")
            print(f"{'='*80}")
            ret = (best15["net_pnl"] / STARTING_BALANCE) * 100
            print(f"  MIN_SCORE: {best15['min_score']:.0f}, Risk: {best15['risk_pct']:.1f}%, TP: {best15['tp_mult']:.1f}x, SL: {best15['sl_mult']:.1f}x, Trail: {best15['trail_mult']:.1f}x, Pos: {best15['max_pos']:.0f}")
            print(f"  Net: ${best15['net_pnl']:+.2f} ({ret:+.1f}%), DD: {best15['max_dd']:.1f}%, PF: {best15['profit_factor']:.2f}")

            print(f"\n  ✅ RECOMMENDED LIVE BOT PARAMETERS:")
            print(f"  MIN_SCORE = {best15['min_score']:.0f}")
            print(f"  RISK_PCT = {best15['risk_pct']:.1f}%")
            print(f"  TP_MULT = {best15['tp_mult']:.1f}")
            print(f"  SL_MULT = {best15['sl_mult']:.1f}")
            print(f"  TRAIL_MULT = {best15['trail_mult']:.1f}")
            print(f"  MAX_POSITIONS = {best15['max_pos']:.0f}")

    print(f"\n{'='*80}")
    print(f"  SWEEP COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()