"""
EXACT BOT SIMULATION — 3 MONTHS ON M15+M5 DATA
==============================================
Simulates the bot EXACTLY as it runs in production:
- M15 candle analysis (every 15 min)
- M5 indicators for decisions
- Continuous position management
- Same risk/lot sizing, SL/TP logic

Tests: Baseline vs SL 3.0x vs SL 2.0x+R50%

Run: python local_backtest/exact_bot_simulation.py
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

BASE_CONFIG = {
    "MIN_SCORE": 30, "MAX_POSITIONS": 3, "MIN_ATR": 0.5,
    "TP_ATR_MULT": 4.0, "TP_ATR_MULT_TREND": 6.0, "SL_ATR_MULT": 1.5,
    "BE_ATR_MULT": 2.0, "TRAIL_ATR_MULT": 0.3,
    "HALT_AFTER_LOSSES": 3, "HALT_HOURS": 6, "ENTRY_COOLDOWN_MINUTES": 0,
    "DAILY_LOSS_PCT": 0.03, "SPREAD_COST_PIP": 0.50, "ADX_TREND_THRESHOLD": 20,
    "MAX_TRADES_PER_DAY": 50, "STARTING_BALANCE": 500.00,
    "TRADE_HOURS_START": 0, "TRADE_HOURS_END": 24,
}

RISK_TIERS = [(0, 500, 2.0), (500, 2000, 2.5), (2000, 10000, 3.0), (10000, float('inf'), 4.0)]
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

def get_risk_pct(balance, risk_mult=1.0):
    for lo, hi, pct in RISK_TIERS:
        if lo < balance <= hi:
            return (pct / 100.0) * risk_mult
    return 0.02 * risk_mult

def get_regime_risk_mult(m5_ind, m5w, adx_thresh=20):
    try:
        adx_s = compute_adx(m5w["high"], m5w["low"], m5w["close"])
        adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
    except:
        adx_val = 0
    atr_s = m5_ind.get("atr")
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
    if adx_val >= adx_thresh:
        return "trend", REGIME_RISK_MULT["trend"]
    return "sideways", REGIME_RISK_MULT["sideways"]


def run_simulation(m15_df, m5_df, label, overrides=None):
    """Run exact bot simulation on M15+M5 data. Ticks through every M15 candle."""
    cfg = dict(BASE_CONFIG)
    if overrides:
        cfg.update(overrides)

    risk_mult = overrides.get("RISK_MULT_OVERRIDE", 1.0) if overrides else 1.0

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
    regime_stats = {}

    # Monthly tracking
    monthly = {}

    for idx, (ct, row) in enumerate(m15_df.iterrows()):
        if idx < cfg.get("WARMUP", 200):
            continue

        price = float(row["close"])

        # Session check
        if not (cfg["TRADE_HOURS_START"] <= ct.hour < cfg["TRADE_HOURS_END"]):
            continue
        if ct.weekday() == 4 and ct.hour >= 21:
            positions.clear()
            continue

        # Daily reset
        if last_date is None:
            last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0
            last_date = ct.date()
            trade_count_today = 0
            # Record daily balance
            mk = ct.strftime("%Y-%m")
            if mk not in monthly:
                monthly[mk] = {"start_bal": balance, "end_bal": balance,
                               "trades": 0, "wins": 0, "pnl": 0.0,
                               "peak": balance, "dd": 0.0}

        # Halt/daily loss check
        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * cfg["DAILY_LOSS_PCT"]:
            continue

        # Build M5 window (like bot does: m5_df[m5_df.index <= ct])
        m5u = m5_df[m5_df.index <= ct]
        m5_window = m5u.tail(500).copy()
        m15_window = m15_df.iloc[max(0, idx-500):idx+1].copy()

        if len(m5_window) < 50 or len(m15_window) < 50:
            continue

        # Compute indicators
        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None:
            continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0:
            continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < cfg["MIN_ATR"]:
            continue

        # ===== POSITION MANAGEMENT (runs every M15 candle) =====
        surviving = []
        for p in positions:
            entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
            pv = lot * 100
            p["_high"] = max(p.get("_high", price), price)
            p["_low"] = min(p.get("_low", price), price)

            # Breakeven check
            if not p.get("be", False) and p.get("be_target"):
                if direction == "BUY" and price >= p["be_target"]:
                    p["be"] = True; p["sl"] = entry
                elif direction == "SELL" and price <= p["be_target"]:
                    p["be"] = True; p["sl"] = entry

            # Trailing
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
                mk = ct.strftime("%Y-%m")
                if mk not in monthly:
                    monthly[mk] = {"start_bal": 0, "end_bal": 0, "trades": 0, "wins": 0, "pnl": 0.0, "peak": 0, "dd": 0.0}
                monthly[mk]["trades"] += 1
                monthly[mk]["pnl"] += pnl
                if pnl > 0:
                    monthly[mk]["wins"] += 1
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

        # ===== NEW ENTRY CHECK (every M15 candle) =====
        if len(positions) >= cfg["MAX_POSITIONS"]:
            continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < cfg["ENTRY_COOLDOWN_MINUTES"]:
            continue
        if trade_count_today >= cfg["MAX_TRADES_PER_DAY"]:
            continue

        # Regime
        regime_name, _ = get_regime_risk_mult(ind5, m5_window, cfg["ADX_TREND_THRESHOLD"])
        regime_stats[regime_name] = regime_stats.get(regime_name, 0) + 1

        # ADX for TP
        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except:
            adx_val = 0
        tp_mult = cfg["TP_ATR_MULT_TREND"] if adx_val >= cfg["ADX_TREND_THRESHOLD"] else cfg["TP_ATR_MULT"]

        # Strategy analysis
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

        # RSI confluence
        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40):
                continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60):
                continue
        except:
            pass

        # SL/TP
        sd = atr_val * cfg["SL_ATR_MULT"]
        td = atr_val * tp_mult
        if direction == "BUY":
            sl = round(price - sd, 2); tp = round(price + td, 2)
        else:
            sl = round(price + sd, 2); tp = round(price - td, 2)

        # Lot sizing
        base_risk = get_risk_pct(balance, risk_mult)
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
        positions.append(pos)
        last_entry = ct

    # ===== CALCULATE METRICS =====
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    losses = len(closed) - wins if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []

    # Max drawdown from equity curve
    peak_eq = cfg["STARTING_BALANCE"]
    max_dd = 0.0
    eq_curve = [cfg["STARTING_BALANCE"]]
    for t in closed:
        eq_curve.append(eq_curve[-1] + t["pnl"])
    for eq_val in eq_curve:
        peak_eq = max(peak_eq, eq_val)
        dd = (peak_eq - eq_val) / peak_eq * 100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    final_bal = cfg["STARTING_BALANCE"] + total_pnl

    dir_counts = {}
    reason_counts = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1

    # Monthly summary
    monthly_summary = {}
    for mk, mdata in sorted(monthly.items()):
        monthly_summary[mk] = mdata

    # Finalize monthly data with end balances
    # Recalculate monthly with proper peak tracking
    running_bal = cfg["STARTING_BALANCE"]
    per_month_closed = {mk: [] for mk in monthly.keys()}
    for t in sorted(closed, key=lambda x: x["close_time"]):
        mk = t["close_time"].strftime("%Y-%m")
        if mk in per_month_closed:
            per_month_closed[mk].append(t)

    monthly_final = {}
    for mk in sorted(per_month_closed.keys()):
        trades = per_month_closed[mk]
        m_pnl = sum(t["pnl"] for t in trades)
        m_wins = sum(1 for t in trades if t["pnl"] > 0)
        monthly_final[mk] = {
            "pnl": m_pnl, "trades": len(trades), "wins": m_wins,
            "return_pct": (m_pnl / cfg["STARTING_BALANCE"]) * 100,
        }

    return {
        "label": label,
        "final_balance": final_bal, "net_pnl": total_pnl,
        "return_pct": (total_pnl / cfg["STARTING_BALANCE"]) * 100,
        "trades": len(closed), "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "profit_factor": pf, "max_dd_pct": max_dd,
        "buy_count": dir_counts.get("BUY", 0), "sell_count": dir_counts.get("SELL", 0),
        "sl_count": reason_counts.get("SL", 0), "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
        "monthly": monthly_final,
    }


def load_data():
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

    return df, df5


def print_results(results):
    r = results
    print(f"\n  {'='*55}")
    print(f"  {r['label']}")
    print(f"  {'='*55}")
    print(f"  Final Balance:     ${r['final_balance']:.2f}  (start $500)")
    print(f"  Net Profit:        ${r['net_pnl']:+.2f}  ({r['return_pct']:+.1f}%)")
    print(f"  Max Drawdown:      {r['max_dd_pct']:.1f}%")
    print(f"  Profit Factor:     {r['profit_factor']:.2f}")
    print(f"  Win Rate:          {r['win_rate']:.1f}%  ({r['wins']}W/{r['losses']}L)")
    print(f"  Trades:            {r['trades']}")
    print(f"  Avg Win/Loss:      ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
    print(f"  Exit:              SL={r['sl_count']}  TP={r['tp_count']}  Trail={r['trail_count']}")
    print(f"  Direction:         BUY={r['buy_count']}  SELL={r['sell_count']}")

    if r["monthly"]:
        print(f"\n  Monthly:")
        for mk, md in sorted(r["monthly"].items()):
            wr = md['wins']/md['trades']*100 if md['trades'] > 0 else 0
            print(f"    {mk}: ${md['pnl']:+.2f}  ({md['trades']}T, {md['wins']}W, {wr:.0f}% WR)")


def main():
    print("=" * 100)
    print("  EXACT BOT SIMULATION — 3 MONTHS (MAY-JUL 2026)")
    print("  M15 analysis | M5 indicators | Live bot replication")
    print("=" * 100)

    m15, m5 = load_data()
    print(f"\nData: {len(m15)} M15 candles, {len(m5)} M5 candles")
    print(f"Period: {m15.index[0]} to {m15.index[-1]}")

    configs = [
        ("BASELINE (current: SL 1.5x)", {}),
        ("A: SL 3.0x ATR only", {"SL_ATR_MULT": 3.0}),
        ("B: SL 2.0x ATR + Risk 50%", {"SL_ATR_MULT": 2.0, "RISK_MULT_OVERRIDE": 0.5}),
    ]

    results = []
    for label, ov in configs:
        print(f"\n  Simulating: {label}...")
        r = run_simulation(m15, m5, label, ov)
        results.append(r)
        print_results(r)

    # Comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"\n  {'Metric':<28} {'BASELINE':<18} {'A: SL 3.0x':<18} {'B: SL 2.0x+R50%':<18}")
    print(f"  {'-'*82}")
    metric_keys = [
        ("Net Profit ($)", "net_pnl", "${:+.2f}"),
        ("Return (%)", "return_pct", "{:+.1f}%"),
        ("Max Drawdown (%)", "max_dd_pct", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Win Rate (%)", "win_rate", "{:.1f}%"),
        ("Total Trades", "trades", "{:d}"),
        ("Avg Win ($)", "avg_win", "${:.2f}"),
        ("Avg Loss ($)", "avg_loss", "${:.2f}"),
        ("SL Hits", "sl_count", "{:d}"),
        ("TP Hits", "tp_count", "{:d}"),
    ]

    for name, key, fmt in metric_keys:
        vals = []
        for r in results:
            v = r[key]
            if isinstance(v, float):
                vals.append(fmt.format(v))
            else:
                vals.append(str(v))
        print(f"  {name:<28} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Monthly comparison
    print(f"\n  {'-'*82}")
    print(f"  MONTHLY COMPARISON")
    print(f"  {'-'*82}")
    all_months = set()
    for r in results:
        all_months.update(r["monthly"].keys())
    for mk in sorted(all_months):
        vals = []
        for r in results:
            md = r["monthly"].get(mk, {})
            pnl = md.get("pnl", 0)
            vals.append(f"${pnl:+.2f}")
        print(f"  {mk:<28} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Recommendation
    print(f"\n{'='*100}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'='*100}")

    baseline = results[0]
    a = results[1]
    b = results[2]

    # Simple scoring
    a_score = 0
    b_score = 0

    # Drawdown (lower is better)
    if a["max_dd_pct"] < baseline["max_dd_pct"]: a_score += 1
    if b["max_dd_pct"] < baseline["max_dd_pct"]: b_score += 1
    if a["max_dd_pct"] <= 30: a_score += 1
    if b["max_dd_pct"] <= 30: b_score += 1

    # Profit (higher is better)
    if a["net_pnl"] > baseline["net_pnl"]: a_score += 1
    if b["net_pnl"] > baseline["net_pnl"]: b_score += 1

    # PF (higher is better)
    if a["profit_factor"] > baseline["profit_factor"]: a_score += 1
    if b["profit_factor"] > baseline["profit_factor"]: b_score += 1

    # Win rate
    if a["win_rate"] > baseline["win_rate"]: a_score += 1
    if b["win_rate"] > baseline["win_rate"]: b_score += 1

    # SL % reduction
    baseline_sl_pct = baseline["sl_count"] / max(baseline["trades"], 1)
    a_sl_pct = a["sl_count"] / max(a["trades"], 1)
    b_sl_pct = b["sl_count"] / max(b["trades"], 1)
    if a_sl_pct < baseline_sl_pct: a_score += 1
    if b_sl_pct < baseline_sl_pct: b_score += 1

    print(f"\n  Scoring (out of 6 possible pts):")
    print(f"    {'A: SL 3.0x ATR':<25} {a_score} pts")
    print(f"    {'B: SL 2.0x + Risk 50%':<25} {b_score} pts")

    if a_score >= b_score:
        winner = "A: SL 3.0x ATR only"
        w_r = a
    else:
        winner = "B: SL 2.0x ATR + Risk 50%"
        w_r = b

    print(f"\n  ★ RECOMMENDED: {winner}")
    print(f"  ★ {w_r['return_pct']:+.1f}% return | {w_r['max_dd_pct']:.1f}% DD | PF {w_r['profit_factor']:.2f} | WR {w_r['win_rate']:.1f}%")

    # Show month-by-month performance
    print(f"\n  Month-by-month comparison:")
    for mk in sorted(all_months):
        a_pnl = a["monthly"].get(mk, {}).get("pnl", 0)
        b_pnl = b["monthly"].get(mk, {}).get("pnl", 0)
        w_pnl = w_r["monthly"].get(mk, {}).get("pnl", 0)
        print(f"    {mk}: A=${a_pnl:+.2f}  B=${b_pnl:+.2f}  ★Winner=${w_pnl:+.2f}")

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exact_simulation_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("EXACT BOT SIMULATION RESULTS (3 months M15+M5)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recommended: {winner}\n\n")
        for r in results:
            f.write(f"\n{r['label']}:\n")
            f.write(f"  Net: ${r['net_pnl']:.2f} ({r['return_pct']:+.1f}%)\n")
            f.write(f"  DD: {r['max_dd_pct']:.1f}% | PF: {r['profit_factor']:.2f} | WR: {r['win_rate']:.1f}%\n")
            f.write(f"  Trades: {r['trades']} | SL: {r['sl_count']} | TP: {r['tp_count']} | Trail: {r['trail_count']}\n")
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()