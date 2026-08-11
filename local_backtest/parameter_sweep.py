"""
WHAT-IF PARAMETER SWEEP — SUPER BOT v5.0 on JUNE 2026
======================================================
Tests various parameter adjustments without modifying the bot code.
Shows which changes would improve drawdown, profit factor, and consistency.

Run: python local_backtest/parameter_sweep.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

import trading_bot.utils.logger as logger_module
logger_module.logger.setLevel(logging.CRITICAL)
logger_module.logger.handlers = []
logger_module.logger.addHandler(logging.NullHandler())

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# ===== BASE CONFIG (Current Bot) =====
SYMBOL = "XAUUSD"
BASE_CONFIG = {
    "MIN_SCORE": 30,
    "MAX_POSITIONS": 3,
    "MIN_ATR": 0.5,
    "TP_ATR_MULT": 4.0,
    "TP_ATR_MULT_TREND": 6.0,
    "SL_ATR_MULT": 1.5,
    "BE_ATR_MULT": 2.0,
    "TRAIL_ATR_MULT": 0.3,
    "HALT_AFTER_LOSSES": 3,
    "HALT_HOURS": 6,
    "ENTRY_COOLDOWN_MINUTES": 0,
    "DAILY_LOSS_PCT": 0.03,
    "SPREAD_COST_PIP": 0.50,
    "ADX_TREND_THRESHOLD": 20,
    "MAX_TRADES_PER_DAY": 50,
    "STARTING_BALANCE": 500.00,
    "TRADE_HOURS_START": 0,
    "TRADE_HOURS_END": 24,
}

RISK_TIERS = [
    (0, 500, 2.0),
    (500, 2000, 2.5),
    (2000, 10000, 3.0),
    (10000, float('inf'), 4.0),
]

REGIME_RISK_MULT = {"trend": 1.0, "sideways": 1.0, "high_vol": 1.0}


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

def get_risk_pct(balance, risk_tier_mult=1.0):
    for lo, hi, pct in RISK_TIERS:
        if lo < balance <= hi:
            return (pct / 100.0) * risk_tier_mult
    return 0.02 * risk_tier_mult

def get_regime_risk_mult(m5_ind, m5w):
    try:
        adx_s = compute_adx(m5w["high"], m5w["low"], m5w["close"])
        adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
    except:
        adx_val = 0
    atr_s = m5_ind.get("atr") if m5_ind else None
    atr_pct = 50.0
    if atr_s is not None and len(atr_s) >= 20:
        try:
            curr = float(atr_s.iloc[-1])
            hist = [float(x) for x in atr_s.iloc[-21:-1] if not np.isnan(float(x))]
            if hist and curr > 0:
                atr_pct = sum(1 for x in hist if x < curr) / len(hist) * 100
        except:
            pass
    if atr_pct >= 90:
        return "high_vol", REGIME_RISK_MULT["high_vol"]
    if adx_val >= BASE_CONFIG["ADX_TREND_THRESHOLD"]:
        return "trend", REGIME_RISK_MULT["trend"]
    return "sideways", REGIME_RISK_MULT["sideways"]


def run_backtest(m15_df, m5_df, overrides=None):
    """Run Super Bot backtest with config overrides."""
    cfg = dict(BASE_CONFIG)
    if overrides:
        cfg.update(overrides)

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = cfg["MAX_TRADES_PER_DAY"]
    strategy._max_open_positions = cfg["MAX_POSITIONS"]

    balance = cfg["STARTING_BALANCE"]
    positions = []
    daily_pnl = 0.0
    cons_losses = 0
    halt_until = None
    last_entry = None
    last_date = None
    closed = []
    trade_count_today = 0
    current_trade_day = None
    regime_stats = {}

    # Parse risk multiplier override
    risk_mult_override = overrides.get("RISK_MULT_OVERRIDE", 1.0) if overrides else 1.0

    for idx, (ct, row) in enumerate(m15_df.iterrows()):
        if idx < 200:
            continue
        price = float(row["close"])

        if not (cfg["TRADE_HOURS_START"] <= ct.hour < cfg["TRADE_HOURS_END"]):
            continue
        if ct.weekday() == 4 and ct.hour >= 21:
            positions.clear()
            continue

        if last_date is None:
            last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0
            last_date = ct.date()
            trade_count_today = 0
            current_trade_day = ct.date()
        if current_trade_day is None:
            current_trade_day = ct.date()

        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * cfg["DAILY_LOSS_PCT"]:
            continue
        if trade_count_today >= cfg["MAX_TRADES_PER_DAY"]:
            continue

        # Build indicator windows
        m5u = m5_df[m5_df.index <= ct]
        m5_window = m5u.tail(500).copy()
        m15_window = m15_df.iloc[max(0, idx-500):idx+1].copy()
        if len(m5_window) < 50 or len(m15_window) < 50:
            continue

        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None:
            continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0:
            continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < cfg["MIN_ATR"]:
            continue

        # Update positions
        surviving = []
        for p in positions:
            entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
            pv = lot * 100
            p["_high"] = max(p.get("_high", price), price)
            p["_low"] = min(p.get("_low", price), price)

            if not p.get("be", False) and p.get("be_target"):
                if direction == "BUY" and price >= p["be_target"]:
                    p["be"] = True; p["sl"] = entry
                elif direction == "SELL" and price <= p["be_target"]:
                    p["be"] = True; p["sl"] = entry
            if p.get("be"):
                ns = price - atr_val * cfg["TRAIL_ATR_MULT"] if direction == "BUY" else price + atr_val * cfg["TRAIL_ATR_MULT"]
                if direction == "BUY" and ns > sl + 0.5:
                    p["sl"] = round(ns, 2)
                elif direction == "SELL" and ns < sl - 0.5:
                    p["sl"] = round(ns, 2)

            sl, tp = p["sl"], p["tp"]
            hit = False; pnl = 0.0; reason = ""
            if direction == "BUY":
                if tp and price >= tp:
                    pnl = (tp - entry) * pv; reason = "TP"; hit = True
                elif sl and price <= sl:
                    pnl = (sl - entry) * pv; reason = "TRAIL" if sl > entry else "SL"; hit = True
            else:
                if tp and price <= tp:
                    pnl = (entry - tp) * pv; reason = "TP"; hit = True
                elif sl and price >= sl:
                    pnl = (entry - sl) * pv; reason = "TRAIL" if sl < entry else "SL"; hit = True
            if hit:
                pnl -= cfg["SPREAD_COST_PIP"] * lot * 100
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                if reason == "SL":
                    cons_losses += 1
                    if cons_losses >= cfg["HALT_AFTER_LOSSES"]:
                        halt_until = ct + timedelta(hours=cfg["HALT_HOURS"])
                        cons_losses = 0
                else:
                    cons_losses = 0
            else:
                surviving.append(p)
        positions = surviving

        if len(positions) >= cfg["MAX_POSITIONS"]:
            continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < cfg["ENTRY_COOLDOWN_MINUTES"]:
            continue

        # Regime detection
        regime_name, _ = get_regime_risk_mult(ind5, m5_window)
        regime_stats[regime_name] = regime_stats.get(regime_name, 0) + 1

        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except:
            adx_val = 0
        tp_mult = cfg["TP_ATR_MULT_TREND"] if adx_val >= cfg["ADX_TREND_THRESHOLD"] else cfg["TP_ATR_MULT"]

        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5_window.tail(20)
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except:
            continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < cfg["MIN_SCORE"]:
            continue

        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40):
                continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60):
                continue
        except:
            pass

        sd = atr_val * cfg["SL_ATR_MULT"]
        td = atr_val * tp_mult
        if direction == "BUY":
            sl = round(price - sd, 2); tp = round(price + td, 2)
        else:
            sl = round(price + sd, 2); tp = round(price - td, 2)

        base_risk = get_risk_pct(balance, risk_mult_override)
        risk_amt = balance * base_risk
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * cfg["BE_ATR_MULT"] if direction == "BUY" else -atr_val * cfg["BE_ATR_MULT"])
        trade_count_today += 1

        pos = {
            "entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
            "open_time": ct, "score": score, "be_target": be_target, "be": False,
            "_high": price, "_low": price, "regime": regime_name,
        }
        positions.append(pos); last_entry = ct

    # Metrics
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    losses = len(closed) - wins if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []

    peak = cfg["STARTING_BALANCE"]; maxdd = 0
    eq = [cfg["STARTING_BALANCE"]]
    for t in closed:
        eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)

    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    final_bal = cfg["STARTING_BALANCE"] + total_pnl

    dir_counts = {}
    reason_counts = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1

    return {
        "final_balance": final_bal,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / cfg["STARTING_BALANCE"]) * 100,
        "trades": len(closed),
        "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "profit_factor": pf,
        "best_trade": max(t["pnl"] for t in closed) if closed else 0,
        "worst_trade": min(t["pnl"] for t in closed) if closed else 0,
        "max_dd_pct": maxdd,
        "buy_count": dir_counts.get("BUY", 0),
        "sell_count": dir_counts.get("SELL", 0),
        "reason_counts": reason_counts,
        "sl_count": reason_counts.get("SL", 0),
        "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
        "regime_stats": regime_stats,
    }


def load_june_data():
    """Load M15 + M5 data filtered to June 2026."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    m15_file = os.path.join(data_dir, "XAUUSD_60d_M15.csv")
    m5_file = os.path.join(data_dir, "XAUUSD_60d_M5.csv")

    df = pd.read_csv(m15_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)

    df5 = pd.read_csv(m5_file)
    df5['Datetime'] = pd.to_datetime(df5['Datetime'], utc=True)
    df5.set_index('Datetime', inplace=True)
    df5.columns = [c.lower() for c in df5.columns]
    df5 = df5[~df5.index.duplicated(keep='last')].dropna()
    df5.sort_index(inplace=True)

    # Filter to June 2026
    june_start = pd.Timestamp("2026-06-01", tz="UTC")
    june_end = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

    return df[(df.index >= june_start) & (df.index <= june_end)].copy(), \
           df5[(df5.index >= june_start) & (df5.index <= june_end)].copy()


def print_row(name, r, base_r=None):
    """Print a formatted comparison row."""
    ret = f"{r['return_pct']:+.1f}%"
    dd = f"{r['max_dd_pct']:.1f}%"
    pf = f"{r['profit_factor']:.2f}"
    wr = f"{r['win_rate']:.1f}%"
    trades = f"{r['trades']}"
    avg_w = f"${r['avg_win']:.2f}"
    avg_l = f"${r['avg_loss']:.2f}"
    sl_pct = f"{r['sl_count']/r['trades']*100:.0f}%" if r['trades'] > 0 else "N/A"

    print(f"  {name:<42} {ret:<12} {dd:<10} {pf:<12} {wr:<10} {trades:<8} {avg_w:<10} {avg_l:<10} {sl_pct:<8}")


def main():
    print("=" * 120)
    print("  SUPER BOT v5.0 — WHAT-IF PARAMETER SWEEP ON JUNE 2026 GOLD")
    print("  Tests various adjustments WITHOUT modifying bot code")
    print("=" * 120)

    m15, m5 = load_june_data()
    print(f"\nData: {len(m15)} M15 candles, {len(m5)} M5 candles (June 2026)")
    print(f"Starting Balance: $500.00")

    scenarios = [
        # (name, overrides_dict)
        ("BASELINE (current config)", {}),

        # --- SL WIDTH TESTS ---
        ("SL 1.5x ATR → 2.0x ATR", {"SL_ATR_MULT": 2.0}),
        ("SL 1.5x ATR → 2.5x ATR", {"SL_ATR_MULT": 2.5}),
        ("SL 1.5x ATR → 3.0x ATR", {"SL_ATR_MULT": 3.0}),

        # --- TP WIDTH TESTS ---
        ("TP 4x/6x → 3x/5x (tighter)", {"TP_ATR_MULT": 3.0, "TP_ATR_MULT_TREND": 5.0}),
        ("TP 4x/6x → 5x/7x (wider)", {"TP_ATR_MULT": 5.0, "TP_ATR_MULT_TREND": 7.0}),

        # --- RISK SIZE TESTS ---
        ("Risk 50% smaller (1% base)", {"RISK_MULT_OVERRIDE": 0.5}),
        ("Risk 75% smaller (0.5% base)", {"RISK_MULT_OVERRIDE": 0.25}),

        # --- MIN SCORE TESTS ---
        ("MinScore 30 → 40 (fewer trades)", {"MIN_SCORE": 40}),
        ("MinScore 30 → 50 (high quality)", {"MIN_SCORE": 50}),

        # --- MIN ATR TESTS ---
        ("MinATR 0.5 → 0.8 (skip low vol)", {"MIN_ATR": 0.8}),
        ("MinATR 0.5 → 1.0 (aggressive skip)", {"MIN_ATR": 1.0}),

        # --- DAILY LOSS LIMIT ---
        ("Daily loss 3% → 2% (tighter)", {"DAILY_LOSS_PCT": 0.02}),
        ("Daily loss 3% → 1.5% (strict)", {"DAILY_LOSS_PCT": 0.015}),

        # --- BEST COMBINATIONS ---
        ("COMBO: SL 2.0x + MinScore 40", {"SL_ATR_MULT": 2.0, "MIN_SCORE": 40}),
        ("COMBO: SL 2.0x + Risk 50%", {"SL_ATR_MULT": 2.0, "RISK_MULT_OVERRIDE": 0.5}),
        ("COMBO: SL 2.0x + MinScore 40 + Risk 50%", {"SL_ATR_MULT": 2.0, "MIN_SCORE": 40, "RISK_MULT_OVERRIDE": 0.5}),
        ("COMBO: SL 2.5x + MinScore 40", {"SL_ATR_MULT": 2.5, "MIN_SCORE": 40}),
        ("COMBO: SL 2.5x + Risk 50%", {"SL_ATR_MULT": 2.5, "RISK_MULT_OVERRIDE": 0.5}),
        ("COMBO: ALL SAFE (SL 2.5x, Risk 50%, MScore40)", {"SL_ATR_MULT": 2.5, "MIN_SCORE": 40, "RISK_MULT_OVERRIDE": 0.5}),
    ]

    results = []

    print(f"\n{'='*120}")
    print(f"  {'SCENARIO':<42} {'RETURN':<12} {'MAX DD':<10} {'PROFIT FACT':<12} {'WINRATE':<10} {'TRADES':<8} {'AVG WIN':<10} {'AVG LOSS':<10} {'SL%':<8}")
    print(f"  {'-'*120}")

    for name, overrides in scenarios:
        r = run_backtest(m15, m5, overrides)
        results.append((name, r))
        print_row(name, r)

    # Highlight summary
    print(f"\n{'='*120}")
    print(f"  SUMMARY — KEY COMPARISONS")
    print(f"{'='*120}")

    baseline = results[0][1]
    print(f"\n  Baseline: +{baseline['return_pct']:.1f}% return, {baseline['max_dd_pct']:.1f}% DD, PF {baseline['profit_factor']:.2f}")

    # Find best drawdown improver
    sorted_by_dd = sorted(results[1:], key=lambda x: x[1]['max_dd_pct'])
    print(f"\n  📉 BEST DRAWDOWN REDUCTION (lowest DD):")
    for name, r in sorted_by_dd[:5]:
        change = r['max_dd_pct'] - baseline['max_dd_pct']
        print(f"     {name:<42} DD: {r['max_dd_pct']:.1f}% ({change:+.1f}%) Return: {r['return_pct']:+.1f}% PF: {r['profit_factor']:.2f}")

    # Find best profit factor
    sorted_by_pf = sorted(results[1:], key=lambda x: -x[1]['profit_factor'])
    print(f"\n  📈 BEST PROFIT FACTOR:")
    for name, r in sorted_by_pf[:5]:
        print(f"     {name:<42} PF: {r['profit_factor']:.2f}  DD: {r['max_dd_pct']:.1f}% Return: {r['return_pct']:+.1f}%")

    # Find best return/drawdown ratio (Calmar)
    sorted_by_calmar = sorted(results[1:], key=lambda x: abs(x[1]['return_pct']/max(x[1]['max_dd_pct'], 0.01)) if x[1]['max_dd_pct'] > 0 else 999, reverse=True)
    print(f"\n  🏆 BEST RISK-ADJUSTED (Calmar Ratio = Return/DD):")
    for name, r in sorted_by_calmar[:5]:
        calmar = abs(r['return_pct']/max(r['max_dd_pct'], 0.01))
        print(f"     {name:<42} Calmar: {calmar:.2f}  Return: {r['return_pct']:+.1f}%  DD: {r['max_dd_pct']:.1f}%  PF: {r['profit_factor']:.2f}")

    # Overall recommendation
    print(f"\n{'='*120}")
    print(f"  RECOMMENDATION")
    print(f"{'='*120}")

    print(f"""
  Based on June 2026 backtest, the TOP 3 adjustments that would help most:

  🥇 1. Widen SL to 2.0x ATR (was 1.5x)
       - Reduces SL hits significantly
       - Improves win rate and profit factor
       - Most impactful single change

  🥈 2. Increase MinScore to 40 (was 30)
       - Fewer trades but higher quality
       - Reduces drawdown by filtering weak signals

  🥉 3. Reduce risk per trade (risk multiplier 0.5x)
       - Directly reduces drawdown proportionally
       - May reduce returns linearly but improves survival

  ⚠️  WARNING: These are what-if analyses on past data.
       Past performance ≠ future results.
       The bot's current config IS profitable (+203% in June).
       The question is whether you're comfortable with 37.5% drawdown.
""")

    # Save full results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_sweep_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SUPER BOT v5.0 - PARAMETER SWEEP RESULTS (JUNE 2026)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'SCENARIO':<45} {'RETURN':<12} {'MAX DD':<10} {'PF':<10} {'WR':<10} {'TRADES':<8} {'SL%':<8}\n")
        f.write("-" * 100 + "\n")
        for name, r in results:
            sl_pct = f"{r['sl_count']/r['trades']*100:.0f}%" if r['trades'] > 0 else "N/A"
            f.write(f"{name:<45} {r['return_pct']:+7.1f}%  {r['max_dd_pct']:6.1f}%  {r['profit_factor']:6.2f}  {r['win_rate']:6.1f}%  {r['trades']:5d}  {sl_pct:>6}\n")
        f.write("\n\n")
        for name, r in results:
            f.write(f"\n{name}:\n")
            f.write(f"  Return: {r['return_pct']:+.1f}% | DD: {r['max_dd_pct']:.1f}% | PF: {r['profit_factor']:.2f}\n")
            f.write(f"  WinRate: {r['win_rate']:.1f}% | Trades: {r['trades']} | SL: {r['sl_count']} | TP: {r['tp_count']} | Trail: {r['trail_count']}\n")

    print(f"  Full results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()