"""
6-MONTH COMPARISON: SL 3.0x ATR vs SL 2.0x + Risk 50%
======================================================
Tests both configs on January-June 2026 H1 data to determine
which is more consistent across different market conditions.

Run: python local_backtest/six_month_comparison.py
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

# ===== BASE CONFIG =====
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


def run_backtest(main_df, overrides=None):
    """Run Super Bot backtest on H1 data (uses main_df as both M15 and M5 proxy)."""
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

    risk_mult_override = overrides.get("RISK_MULT_OVERRIDE", 1.0) if overrides else 1.0

    for idx in range(200, len(main_df)):
        ct = main_df.index[idx]
        price = float(main_df["close"].iloc[idx])

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

        # Use same dataframe for both M5 and M15 proxy
        m5_window = main_df.iloc[max(0, idx-200):idx+1].copy()
        m15_window = main_df.iloc[max(0, idx-500):idx+1].copy()

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
    monthly_pnl = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in monthly_pnl:
            monthly_pnl[mk] = {"pnl": 0.0, "trades": 0, "wins": 0}
        monthly_pnl[mk]["pnl"] += t["pnl"]
        monthly_pnl[mk]["trades"] += 1
        if t["pnl"] > 0:
            monthly_pnl[mk]["wins"] += 1

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
        "max_dd_pct": maxdd,
        "buy_count": dir_counts.get("BUY", 0),
        "sell_count": dir_counts.get("SELL", 0),
        "reason_counts": reason_counts,
        "sl_count": reason_counts.get("SL", 0),
        "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
        "monthly_pnl": monthly_pnl,
    }


def load_h1_data():
    """Load the 6-month H1 data."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    # Try 6mo_H1 first, fallback to 1y_H1
    fp = os.path.join(data_dir, "XAUUSD_6mo_H1.csv")
    if not os.path.exists(fp):
        fp = os.path.join(data_dir, "XAUUSD_1y_H1.csv")

    df = pd.read_csv(fp)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)
    print(f"  Loaded {os.path.basename(fp)}: {len(df)} candles")
    print(f"  Period: {df.index[0]} to {df.index[-1]}")
    return df


def print_results(label, results):
    """Print formatted results."""
    r = results["total"]
    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"  Final Balance:     ${r['final_balance']:.2f}")
    print(f"  Net Profit:        ${r['net_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
    print(f"  Total Trades:      {r['trades']}")
    print(f"  Win Rate:          {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)")
    if r['avg_win']:
        print(f"  Avg Win / Loss:    ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
    print(f"  Profit Factor:     {r['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {r['max_dd_pct']:.1f}%")
    print(f"  BUY / SELL:        {r['buy_count']} / {r['sell_count']}")
    print(f"  Exit Reasons:      SL={r['sl_count']} TP={r['tp_count']} Trail={r['trail_count']}")

    print(f"\n  Monthly Breakdown:")
    print(f"  {'Month':<10} {'Trades':<8} {'P&L':<12} {'Return':<10}")
    print(f"  {'-'*40}")
    best_month = None
    worst_month = None
    months_with_data = 0
    profitable_months = 0
    for mk in sorted(results["monthly"].keys()):
        md = results["monthly"][mk]
        print(f"  {mk:<10} {md['trades']:<8} ${md['pnl']:<+9.2f} {md['return_pct']:<+9.1f}%")
        months_with_data += 1
        if md['pnl'] > 0:
            profitable_months += 1
        if best_month is None or md['pnl'] > best_month['pnl']:
            best_month = {"month": mk, "pnl": md['pnl'], "return_pct": md['return_pct']}
        if worst_month is None or md['pnl'] < worst_month['pnl']:
            worst_month = {"month": mk, "pnl": md['pnl'], "return_pct": md['return_pct']}

    if months_with_data > 0:
        print(f"\n  📊 Consistency Metrics:")
        print(f"  Profitable Months:  {profitable_months}/{months_with_data} ({profitable_months/months_with_data*100:.0f}%)")
        print(f"  Best Month:         {best_month['month']} ${best_month['pnl']:+.2f} ({best_month['return_pct']:+.1f}%)")
        print(f"  Worst Month:        {worst_month['month']} ${worst_month['pnl']:+.2f} ({worst_month['return_pct']:+.1f}%)")

    return profitable_months, months_with_data


def main():
    print("=" * 100)
    print("  6-MONTH COMPARISON: SL 3.0x ATR vs SL 2.0x + Risk 50%")
    print("  Testing consistency across different market conditions")
    print("=" * 100)

    df = load_h1_data()

    # Define the two configs
    configs = [
        ("A: SL 3.0x ATR only", {"SL_ATR_MULT": 3.0}),
        ("B: SL 2.0x ATR + Risk 50%", {"SL_ATR_MULT": 2.0, "RISK_MULT_OVERRIDE": 0.5}),
    ]

    results_dict = {}

    for label, overrides in configs:
        print(f"\n  Running: {label}...")
        r = run_backtest(df, overrides)

        # Calculate monthly returns
        monthly = {}
        total_pnl = r["net_pnl"]
        for mk, md in r["monthly_pnl"].items():
            monthly[mk] = {
                "pnl": md["pnl"],
                "trades": md["trades"],
                "return_pct": (md["pnl"] / 500.0) * 100,
            }

        # Sort monthly keys
        sorted_months = sorted(monthly.keys())
        running_balance = 500.0
        cumulative_pnl = 0.0
        for mk in sorted_months:
            cumulative_pnl += monthly[mk]["pnl"]
            monthly[mk]["cumulative_pnl"] = cumulative_pnl
            monthly[mk]["cumulative_return"] = (cumulative_pnl / 500.0) * 100

        results_dict[label] = {
            "total": r,
            "monthly": monthly,
            "sorted_months": sorted_months,
        }

    # Print full results
    for label in results_dict:
        rd = results_dict[label]
        pm, tm = print_results(label, rd)

    # PRINT SIDE-BY-SIDE COMPARISON TABLE
    print(f"\n{'='*100}")
    print(f"  SIDE-BY-SIDE COMPARISON TABLE")
    print(f"{'='*100}")

    config_a = results_dict[configs[0][0]]
    config_b = results_dict[configs[1][0]]

    all_months = sorted(set(config_a["sorted_months"] + config_b["sorted_months"]))

    print(f"\n  {'Month':<10} {'SL3.0x Return':<16} {'SL3.0x Cum':<14} {'SL2.0x+R50% Return':<20} {'SL2.0x+R50% Cum':<16} {'Winner':<10}")
    print(f"  {'-'*86}")

    a_wins = 0
    b_wins = 0
    for mk in all_months:
        a_ret = config_a["monthly"].get(mk, {}).get("return_pct", 0)
        b_ret = config_b["monthly"].get(mk, {}).get("return_pct", 0)
        a_cum = config_a["monthly"].get(mk, {}).get("cumulative_return", 0)
        b_cum = config_b["monthly"].get(mk, {}).get("cumulative_return", 0)
        winner = "A" if a_ret > b_ret else "B" if b_ret > a_ret else "="
        if winner == "A": a_wins += 1
        elif winner == "B": b_wins += 1
        print(f"  {mk:<10} {a_ret:>+8.1f}%      {a_cum:>+8.1f}%    {b_ret:>+8.1f}%             {b_cum:>+8.1f}%        {winner}")

    # FINAL COMPARISON
    a_total = config_a["total"]
    b_total = config_b["total"]

    print(f"\n  {'='*100}")
    print(f"  FINAL VERDICT")
    print(f"  {'='*100}")

    print(f"\n  {'Metric':<35} {'SL 3.0x ATR':<20} {'SL 2.0x+R50%':<20} {'Winner'}")
    print(f"  {'-'*75}")
    metrics = [
        ("Net Profit", f"${a_total['net_pnl']:.2f}", f"${b_total['net_pnl']:.2f}", a_total['net_pnl'] > b_total['net_pnl']),
        ("Total Return", f"{a_total['return_pct']:.1f}%", f"{b_total['return_pct']:.1f}%", a_total['return_pct'] > b_total['return_pct']),
        ("Max Drawdown", f"{a_total['max_dd_pct']:.1f}%", f"{b_total['max_dd_pct']:.1f}%", a_total['max_dd_pct'] < b_total['max_dd_pct']),
        ("Profit Factor", f"{a_total['profit_factor']:.2f}", f"{b_total['profit_factor']:.2f}", a_total['profit_factor'] > b_total['profit_factor']),
        ("Win Rate", f"{a_total['win_rate']:.1f}%", f"{b_total['win_rate']:.1f}%", a_total['win_rate'] > b_total['win_rate']),
        ("SL Hits %", f"{a_total['sl_count']/a_total['trades']*100:.0f}%" if a_total['trades'] else "N/A",
                     f"{b_total['sl_count']/b_total['trades']*100:.0f}%" if b_total['trades'] else "N/A",
                     a_total['sl_count']/a_total['trades'] < b_total['sl_count']/b_total['trades'] if a_total['trades'] and b_total['trades'] else False),
        ("Monthly Wins", f"{a_wins}/{len(all_months)}", f"{b_wins}/{len(all_months)}", a_wins > b_wins),
    ]

    for name, a_val, b_val, a_better in metrics:
        winner_mark = "✅ A" if a_better else "✅ B" if not a_better else "="
        print(f"  {name:<35} {a_val:<20} {b_val:<20} {winner_mark}")

    # CONSISTENCY ANALYSIS
    print(f"\n  {'='*100}")
    print(f"  CONSISTENCY ANALYSIS")
    print(f"  {'='*100}")

    # Calculate month-over-month volatility
    a_returns = [config_a["monthly"][mk]["return_pct"] for mk in all_months if mk in config_a["monthly"]]
    b_returns = [config_b["monthly"][mk]["return_pct"] for mk in all_months if mk in config_b["monthly"]]

    a_volatility = np.std(a_returns) if a_returns else 0
    b_volatility = np.std(b_returns) if b_returns else 0

    a_negative_months = sum(1 for r in a_returns if r < 0)
    b_negative_months = sum(1 for r in b_returns if r < 0)

    print(f"\n  {'Metric':<40} {'SL 3.0x ATR':<20} {'SL 2.0x+R50%':<20}")
    print(f"  {'-'*80}")
    print(f"  Monthly Return Std Dev:    {a_volatility:<+9.1f}%        {b_volatility:<+9.1f}%")
    print(f"  Negative Months:           {a_negative_months}/{len(a_returns)}                    {b_negative_months}/{len(b_returns)}")
    print(f"  Best Single Month:         {max(a_returns):<+9.1f}%        {max(b_returns):<+9.1f}%")
    print(f"  Worst Single Month:        {min(a_returns):<+9.1f}%        {min(b_returns):<+9.1f}%")
    calmar_a = abs(a_total['return_pct'] / a_total['max_dd_pct']) if a_total['max_dd_pct'] > 0 else float('inf')
    calmar_b = abs(b_total['return_pct'] / b_total['max_dd_pct']) if b_total['max_dd_pct'] > 0 else float('inf')
    print(f"  Calmar Ratio:              {calmar_a:<9.2f}              {calmar_b:<9.2f}")

    # FINAL DECISION
    print(f"\n  {'='*100}")
    print(f"  RECOMMENDATION")
    print(f"  {'='*100}")

    # Decision logic
    scores = {"A: SL 3.0x ATR": 0, "B: SL 2.0x + Risk 50%": 0}

    # 1. Drawdown safety
    if a_total['max_dd_pct'] <= 30:
        scores["A: SL 3.0x ATR"] += 1
    else:
        scores["A: SL 3.0x ATR"] -= 1

    if b_total['max_dd_pct'] <= 30:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["B: SL 2.0x + Risk 50%"] -= 1

    # 2. Profitability
    if b_total['net_pnl'] > a_total['net_pnl']:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["A: SL 3.0x ATR"] += 1

    # 3. Consistency (lower volatility)
    if b_volatility < a_volatility:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["A: SL 3.0x ATR"] += 1

    # 4. Fewer negative months
    if b_negative_months < a_negative_months:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["A: SL 3.0x ATR"] += 1

    # 5. Calmar ratio
    if calmar_b > calmar_a:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["A: SL 3.0x ATR"] += 1

    # 6. Profit factor
    if b_total['profit_factor'] >= a_total['profit_factor']:
        scores["B: SL 2.0x + Risk 50%"] += 1
    else:
        scores["A: SL 3.0x ATR"] += 1

    print(f"\n  Scoring (higher = better):")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {k:<25} {v} pts")

    winner = max(scores, key=scores.get)
    print(f"\n  {'★'*60}")
    print(f"  ★ RECOMMENDED: {winner}")
    print(f"  {'★'*60}")

    if "Risk 50%" in winner:
        print(f"")
        print(f"  Rationale:")
        print(f"  - Lower drawdown ({b_total['max_dd_pct']:.1f}% vs {a_total['max_dd_pct']:.1f}%)")
        print(f"  - More consistent month-to-month (volatility {b_volatility:.1f}% vs {a_volatility:.1f}%)")
        print(f"  - Better Calmar ratio ({calmar_b:.2f} vs {calmar_a:.2f})")
        print(f"  - Prevents large losses in bad market conditions")
    else:
        print(f"")
        print(f"  Rationale:")
        print(f"  - Higher absolute return ({a_total['return_pct']:.1f}% vs {b_total['return_pct']:.1f}%)")
        print(f"  - Acceptable drawdown ({a_total['max_dd_pct']:.1f}%)")
        print(f"  - Higher profit factor ({a_total['profit_factor']:.2f})")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "six_month_comparison.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("6-MONTH COMPARISON: SL 3.0x ATR vs SL 2.0x + Risk 50%\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Winner: {winner}\n\n")
        f.write(f"SL 3.0x ATR: Net=${a_total['net_pnl']:.2f} DD={a_total['max_dd_pct']:.1f}% PF={a_total['profit_factor']:.2f}\n")
        f.write(f"SL 2.0x+R50%: Net=${b_total['net_pnl']:.2f} DD={b_total['max_dd_pct']:.1f}% PF={b_total['profit_factor']:.2f}\n")

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()