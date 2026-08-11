"""
SUPER BOT v5.0 — LAST MONTH (JUNE 2026) BACKTEST
=================================================
Runs the exact Super Bot strategy on June 2026 M15+M5 data.
Uses the same logic as backtest_super_1year.py but filters for June only.

Run: python local_backtest/backtest_last_month.py
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

# ===== SUPER BOT CONFIG (exact match) =====
SYMBOL = "XAUUSD"
MIN_SCORE = 30
MAX_POSITIONS = 3
MIN_ATR = 0.5
TRADE_HOURS_START = 0
TRADE_HOURS_END = 24
TP_ATR_MULT = 4.0
TP_ATR_MULT_TREND = 6.0
SL_ATR_MULT = 1.5
BE_ATR_MULT = 2.0
TRAIL_ATR_MULT = 0.3
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
ENTRY_COOLDOWN_MINUTES = 0
DAILY_LOSS_PCT = 0.03
SPREAD_COST_PIP = 0.50
ADX_TREND_THRESHOLD = 20
MAX_TRADES_PER_DAY = 50
STARTING_BALANCE = 500.00
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

RISK_TIERS = [
    (0, 500, 2.0),
    (500, 2000, 2.5),
    (2000, 10000, 3.0),
    (10000, float('inf'), 4.0),
]

REGIME_RISK_MULT = {
    "trend": 1.0,
    "sideways": 1.0,
    "high_vol": 1.0,
}


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

def get_risk_pct(balance):
    for lo, hi, pct in RISK_TIERS:
        if lo < balance <= hi:
            return pct / 100.0
    return 0.02

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
    if adx_val >= ADX_TREND_THRESHOLD:
        return "trend", REGIME_RISK_MULT["trend"]
    return "sideways", REGIME_RISK_MULT["sideways"]


def run_backtest(m15_df, m5_df=None, start_date=None, end_date=None):
    """Run Super Bot backtest with optional date range filtering."""
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = MAX_TRADES_PER_DAY
    strategy._max_open_positions = MAX_POSITIONS

    # Filter by date range
    if start_date:
        m15_df = m15_df[m15_df.index >= start_date].copy()
    if end_date:
        m15_df = m15_df[m15_df.index <= end_date].copy()

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
    regime_stats = {}
    daily_balances = [STARTING_BALANCE]
    daily_balance_dates = []

    skipped_no_m5 = 0
    skipped_atr = 0
    skipped_signal = 0
    skipped_positions_full = 0

    total_candles = len(m15_df)

    for idx, (ct, row) in enumerate(m15_df.iterrows()):
        if idx < 200:
            continue

        price = float(row["close"])

        if not in_session(ct):
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
            daily_balances.append(balance)
            daily_balance_dates.append(ct.date())
        if current_trade_day is None:
            current_trade_day = ct.date()

        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * DAILY_LOSS_PCT:
            continue
        if trade_count_today >= MAX_TRADES_PER_DAY:
            continue

        # Build indicator windows
        if m5_df is not None:
            m5u = m5_df[m5_df.index <= ct]
            m5_window = m5u.tail(500).copy()
        else:
            m5_window = m15_df.iloc[max(0, idx-200):idx+1].copy()
        m15_window = m15_df.iloc[max(0, idx-500):idx+1].copy()

        if len(m5_window) < 50 or len(m15_window) < 50:
            continue

        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None:
            skipped_no_m5 += 1
            continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0:
            skipped_no_m5 += 1
            continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < MIN_ATR:
            skipped_atr += 1
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
                    p["be"] = True
                    p["sl"] = entry
                elif direction == "SELL" and price <= p["be_target"]:
                    p["be"] = True
                    p["sl"] = entry
            if p.get("be"):
                ns = price - atr_val * TRAIL_ATR_MULT if direction == "BUY" else price + atr_val * TRAIL_ATR_MULT
                if direction == "BUY" and ns > sl + 0.5:
                    p["sl"] = round(ns, 2)
                elif direction == "SELL" and ns < sl - 0.5:
                    p["sl"] = round(ns, 2)

            sl, tp = p["sl"], p["tp"]
            hit = False
            pnl = 0.0
            reason = ""
            if direction == "BUY":
                if tp and price >= tp:
                    pnl = (tp - entry) * pv
                    reason = "TP"
                    hit = True
                elif sl and price <= sl:
                    pnl = (sl - entry) * pv
                    reason = "TRAIL" if sl > entry else "SL"
                    hit = True
            else:
                if tp and price <= tp:
                    pnl = (entry - tp) * pv
                    reason = "TP"
                    hit = True
                elif sl and price >= sl:
                    pnl = (entry - sl) * pv
                    reason = "TRAIL" if sl < entry else "SL"
                    hit = True
            if hit:
                pnl -= SPREAD_COST_PIP * lot * 100
                daily_pnl += pnl
                balance += pnl
                p["pnl"] = pnl
                p["reason"] = reason
                p["close_price"] = price
                p["close_time"] = ct
                closed.append(p)
                if reason == "SL":
                    cons_losses += 1
                    if cons_losses >= HALT_AFTER_LOSSES:
                        halt_until = ct + timedelta(hours=HALT_HOURS)
                        cons_losses = 0
                else:
                    cons_losses = 0
            else:
                surviving.append(p)
        positions = surviving

        if len(positions) >= MAX_POSITIONS:
            skipped_positions_full += 1
            continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES:
            continue

        # Regime detection
        regime_name, risk_mult = get_regime_risk_mult(ind5, m5_window)
        regime_stats[regime_name] = regime_stats.get(regime_name, 0) + 1

        # ADX for dynamic TP
        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except:
            adx_val = 0
        tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT

        # Run strategy
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5_window.tail(20)
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None,
            )
        except:
            continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < MIN_SCORE:
            skipped_signal += 1
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
        sd = atr_val * SL_ATR_MULT
        td = atr_val * tp_mult
        if direction == "BUY":
            sl = round(price - sd, 2)
            tp = round(price + td, 2)
        else:
            sl = round(price + sd, 2)
            tp = round(price - td, 2)

        # Adaptive lot sizing
        base_risk = get_risk_pct(balance)
        effective_risk = base_risk * risk_mult
        risk_amt = balance * effective_risk
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)
        trade_count_today += 1

        pos = {
            "entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
            "open_time": ct, "score": score, "be_target": be_target, "be": False,
            "_high": price, "_low": price, "regime": regime_name,
        }
        positions.append(pos)
        last_entry = ct

    # Calculate metrics
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    losses = len(closed) - wins if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []

    peak = STARTING_BALANCE
    maxdd = 0
    eq = [STARTING_BALANCE]
    for t in closed:
        eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)

    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    final_bal = STARTING_BALANCE + total_pnl

    dir_counts = {}
    reason_counts = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1

    # Monthly breakdown
    monthly_data = {}
    for t in closed:
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in monthly_data:
            monthly_data[mk] = {"trades": 0, "pnl": 0.0, "wins": 0}
        monthly_data[mk]["trades"] += 1
        monthly_data[mk]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            monthly_data[mk]["wins"] += 1

    return {
        "final_balance": final_bal,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": (sum(win_pnls) / len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0,
        "profit_factor": pf,
        "best_trade": max(t["pnl"] for t in closed) if closed else 0,
        "worst_trade": min(t["pnl"] for t in closed) if closed else 0,
        "max_dd_pct": maxdd,
        "buy_count": dir_counts.get("BUY", 0),
        "sell_count": dir_counts.get("SELL", 0),
        "reason_counts": reason_counts,
        "regime_stats": regime_stats,
        "daily_balances": daily_balances,
        "daily_balance_dates": daily_balance_dates,
        "monthly_data": monthly_data,
        "skipped_no_m5": skipped_no_m5,
        "skipped_atr": skipped_atr,
        "skipped_signal": skipped_signal,
        "skipped_positions_full": skipped_positions_full,
        "total_candles": total_candles,
    }


def load_data():
    """Load M15 + M5 data for backtest."""
    data_dir = DATA_DIR
    m15_file = os.path.join(data_dir, "XAUUSD_60d_M15.csv")
    m5_file = os.path.join(data_dir, "XAUUSD_60d_M5.csv")

    if not os.path.exists(m15_file):
        print("  ERROR: 60d M15 data not found!")
        return None, None

    df = pd.read_csv(m15_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)

    df5 = None
    if os.path.exists(m5_file):
        df5 = pd.read_csv(m5_file)
        df5['Datetime'] = pd.to_datetime(df5['Datetime'], utc=True)
        df5.set_index('Datetime', inplace=True)
        df5.columns = [c.lower() for c in df5.columns]
        df5 = df5[~df5.index.duplicated(keep='last')].dropna()
        df5.sort_index(inplace=True)

    return df, df5


def print_results(results, label, period_label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  {period_label}")
    print(f"{'='*80}")
    print(f"  Starting Balance:  ${STARTING_BALANCE:.2f}")
    print(f"  Final Balance:     ${results['final_balance']:.2f}")
    print(f"  Net Profit:        ${results['net_pnl']:+.2f} ({results['return_pct']:+.1f}%)")
    print(f"  Total Trades:      {results['trades']}")
    print(f"  Win Rate:          {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
    if results['avg_win']:
        print(f"  Avg Win / Loss:    ${results['avg_win']:.2f} / ${results['avg_loss']:.2f}")
    if results['profit_factor']:
        print(f"  Profit Factor:     {results['profit_factor']:.2f}")
    if results['buy_count']:
        print(f"  Best / Worst:      ${results['best_trade']:+.2f} / ${results['worst_trade']:+.2f}")
        print(f"  BUY / SELL:        {results['buy_count']} / {results['sell_count']}")
    print(f"  Max Drawdown:      {results['max_dd_pct']:.1f}%")
    print(f"  Exit Reasons:      {results['reason_counts']}")
    if results['regime_stats']:
        total_regime = sum(results['regime_stats'].values())
        print(f"\n  Regime Distribution:")
        for r, c in sorted(results['regime_stats'].items(), key=lambda x: -x[1]):
            print(f"    {r:<15} {c:<6} ({c/total_regime*100:.1f}%)")

    # Monthly breakdown
    if results.get('monthly_data'):
        print(f"\n  Monthly Breakdown:")
        print(f"  {'Month':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'P&L':<12} {'WR':<8}")
        print(f"  {'-'*52}")
        for mk in sorted(results['monthly_data'].keys()):
            md = results['monthly_data'][mk]
            wr = md['wins'] / md['trades'] * 100 if md['trades'] > 0 else 0
            print(f"  {mk:<10} {md['trades']:<8} {md['wins']:<6} {md['trades']-md['wins']:<8} ${md['pnl']:<+9.2f} {wr:<7.1f}%")


def main():
    print("=" * 80)
    print("  SUPER BOT v5.0 — LAST MONTH BACKTEST")
    print("  June 2026 | M15+M5 Data | Config Match")
    print("=" * 80)

    df, df5 = load_data()
    if df is None:
        print("Cannot run backtest without data!")
        return

    # Date ranges for June 2026
    june_start = pd.Timestamp("2026-06-01", tz="UTC")
    june_end = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

    print(f"\nData available: {df.index[0]} to {df.index[-1]}")
    print(f"June 2026 range: {june_start} to {june_end}")

    # Filter to June 2026
    june_m15 = df[(df.index >= june_start) & (df.index <= june_end)].copy()
    june_m5 = None
    if df5 is not None:
        june_m5 = df5[(df5.index >= june_start) & (df5.index <= june_end)].copy()

    print(f"June M15 candles: {len(june_m15)}")
    if june_m5 is not None:
        print(f"June M5 candles:  {len(june_m5)}")

    print(f"\nRunning Super Bot backtest on June 2026 M15+M5 data...")
    results = run_backtest(june_m15, june_m5)

    print_results(results, "SUPER BOT v5.0 — JUNE 2026", "M15 analysis | M5 indicators | $500 start")

    # Risk/Return summary
    print(f"\n{'='*80}")
    print(f"  RISK/RETURN SUMMARY")
    print(f"{'='*80}")
    return_pct = results['return_pct']
    max_dd = results['max_dd_pct']
    calmar = abs(return_pct / max_dd) if max_dd and max_dd > 0 else float('inf')

    print(f"  Total Return:       {return_pct:+.1f}%")
    print(f"  Max Drawdown:       {max_dd:.1f}%")
    print(f"  Calmar Ratio:       {calmar:.2f}")
    print(f"  Profit Factor:      {results['profit_factor']:.2f}")
    print(f"  Win Rate:           {results['win_rate']:.1f}%")
    print(f"  Trades:             {results['trades']}")
    print(f"  Avg Daily Trades:   {results['trades'] / 30:.1f}")

    # Evaluation
    print(f"\n{'='*80}")
    print(f"  BOT EVALUATION")
    print(f"{'='*80}")

    passed = True
    checks = []

    if max_dd > 30:
        checks.append(f"❌ Drawdown {max_dd:.1f}% > 30% limit")
        passed = False
    else:
        checks.append(f"✅ Drawdown {max_dd:.1f}% ≤ 30% limit")

    if results['net_pnl'] <= 0:
        checks.append(f"❌ Net profit ${results['net_pnl']:.2f} is negative (losing)")
        passed = False
    else:
        checks.append(f"✅ Net profit ${results['net_pnl']:+.2f} is positive")

    if results['win_rate'] < 30:
        checks.append(f"⚠️  Win rate {results['win_rate']:.1f}% < 30% (low)")
    else:
        checks.append(f"✅ Win rate {results['win_rate']:.1f}% ≥ 30%")

    if results['profit_factor'] < 1.0:
        checks.append(f"❌ Profit factor {results['profit_factor']:.2f} < 1.0")
        passed = False
    else:
        checks.append(f"✅ Profit factor {results['profit_factor']:.2f} ≥ 1.0")

    for c in checks:
        print(f"  {c}")

    print(f"\n  VERDICT: {'✅ GOOD BOT' if passed else '❌ NEEDS IMPROVEMENT'}")
    if passed:
        monthly_return = results['return_pct']
        yearly_projection = monthly_return * 12
        print(f"  Monthly Return: {monthly_return:+.1f}%")
        print(f"  Yearly Projection: {yearly_projection:+.1f}%")
        if monthly_return > 10:
            print(f"  ⭐ Strong performance!")
        elif monthly_return > 5:
            print(f"  👍 Decent performance")
        elif monthly_return > 0:
            print(f"  🤏 Marginal performance")
    else:
        print(f"  ⚠️  Bot has issues that need addressing")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "june_2026_results.txt")
    with open(output_path, "w") as f:
        f.write(f"SUPER BOT v5.0 — JUNE 2026 BACKTEST RESULTS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Period: June 2026\n")
        f.write(f"Starting Balance: ${STARTING_BALANCE:.2f}\n")
        f.write(f"Final Balance: ${results['final_balance']:.2f}\n")
        f.write(f"Net Profit: ${results['net_pnl']:+.2f} ({results['return_pct']:+.1f}%)\n")
        f.write(f"Total Trades: {results['trades']}\n")
        f.write(f"Win Rate: {results['win_rate']:.1f}%\n")
        f.write(f"Avg Win: ${results['avg_win']:.2f}\n")
        f.write(f"Avg Loss: ${results['avg_loss']:.2f}\n")
        f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        f.write(f"Max Drawdown: {results['max_dd_pct']:.1f}%\n")
        f.write(f"Best Trade: ${results['best_trade']:.2f}\n")
        f.write(f"Worst Trade: ${results['worst_trade']:.2f}\n")
        f.write(f"BUY/SELL: {results['buy_count']}/{results['sell_count']}\n")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()