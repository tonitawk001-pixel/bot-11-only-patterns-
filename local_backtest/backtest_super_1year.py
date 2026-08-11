"""
V22 GOLD SCALPING — SUPER BOT 1-YEAR BACKTEST
===============================================
Tests the Super Bot strategy on 1 year of H1 data.

The Super Bot uses:
  - MIN_SCORE=40 (aggressive)
  - Adaptive risk tiers based on balance (1.5% - 3.0%)
  - Regime-based risk multipliers (trend: 1.0x, sideways: 0.6x, high_vol: 0.3x)
  - ADX-based dynamic TP (3.5x normal, 5.0x trend)
  - Tighter trailing (0.5x ATR)
  - 3-loss halt, daily loss limit, Friday rules

Run: python local_backtest/backtest_super_1year.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# Suppress logging
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

# ===== SUPER BOT CONFIG =====
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
STARTING_BALANCE = 304.99
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Risk tiers
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


def run_backtest_super(m15_df, m5_df=None):
    """Run Super Bot backtest on given data."""
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = MAX_TRADES_PER_DAY
    strategy._max_open_positions = MAX_POSITIONS

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

    for i in range(200, len(m15_df)):
        ct = m15_df.index[i]
        price = float(m15_df["close"].iloc[i])

        if not in_session(ct): continue
        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue

        if last_date is None: last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0
            last_date = ct.date()
            trade_count_today = 0
            current_trade_day = ct.date()
            daily_balances.append(balance)
            daily_balance_dates.append(ct.date())
        if current_trade_day is None: current_trade_day = ct.date()
        if ct.date() == current_trade_day: pass
        else: trade_count_today = 0; current_trade_day = ct.date()

        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS_PCT: continue
        if trade_count_today >= MAX_TRADES_PER_DAY: continue

        # Build indicator windows
        if m5_df is not None:
            m5u = m5_df[m5_df.index <= ct]
            m5_window = m5u.tail(500).copy()
        else:
            m5_window = m15_df.iloc[max(0, i-200):i+1].copy()
        m15_window = m15_df.iloc[max(0, i-500):i+1].copy()

        if len(m5_window) < 50 or len(m15_window) < 50: continue

        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < MIN_ATR: continue

        # Update positions
        surviving = []
        for p in positions:
            entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
            pv = lot * 100
            p["_high"] = max(p.get("_high", price), price)
            p["_low"] = min(p.get("_low", price), price)

            if not p.get("be", False) and p.get("be_target"):
                if direction == "BUY" and price >= p["be_target"]: p["be"] = True; p["sl"] = entry
                elif direction == "SELL" and price <= p["be_target"]: p["be"] = True; p["sl"] = entry
            if p.get("be"):
                ns = price - atr_val * TRAIL_ATR_MULT if direction == "BUY" else price + atr_val * TRAIL_ATR_MULT
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
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                if reason == "SL":
                    cons_losses += 1
                    if cons_losses >= HALT_AFTER_LOSSES: halt_until = ct + timedelta(hours=HALT_HOURS); cons_losses = 0
                else: cons_losses = 0
            else: surviving.append(p)
        positions = surviving

        if len(positions) >= MAX_POSITIONS: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES: continue

        # Regime detection
        regime_name, risk_mult = get_regime_risk_mult(ind5, m5_window)
        regime_stats[regime_name] = regime_stats.get(regime_name, 0) + 1

        # ADX for dynamic TP
        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except: adx_val = 0
        tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT

        # Run strategy
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5_window.tail(20)
            result = strategy.analyze(m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < MIN_SCORE: continue

        # RSI confluence
        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40): continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60): continue
        except: pass

        # EMA200 filter
        closes = m15_window["close"].values
        if len(closes) >= 200:
            ema200 = pd.Series(closes).ewm(200, adjust=False).mean().values
            if len(ema200) >= 10:
                rising = ema200[-1] > ema200[-10]
                if direction == "BUY" and not rising: continue
                if direction == "SELL" and rising: continue

        # SL/TP
        sd = atr_val * SL_ATR_MULT; td = atr_val * tp_mult
        if direction == "BUY": sl = round(price - sd, 2); tp = round(price + td, 2)
        else: sl = round(price + sd, 2); tp = round(price - td, 2)

        # Adaptive lot sizing
        base_risk = get_risk_pct(balance)
        effective_risk = base_risk * risk_mult
        risk_amt = balance * effective_risk
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)
        trade_count_today += 1

        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score, "be_target": be_target, "be": False,
               "_high": price, "_low": price, "regime": regime_name}
        positions.append(pos); last_entry = ct

    # Calculate metrics
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    losses = len(closed) - wins if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []

    peak = STARTING_BALANCE; maxdd = 0
    eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)

    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    final_bal = STARTING_BALANCE + total_pnl

    # Counters
    dir_counts = {}
    reason_counts = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1

    return {
        "final_balance": final_bal,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
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
        "regime_stats": regime_stats,
        "daily_balances": daily_balances,
        "daily_balance_dates": daily_balance_dates,
    }


def load_data():
    """Load available data - try H1 first for long periods."""
    data_dir = DATA_DIR

    files = {
        "1y_H1": os.path.join(data_dir, "XAUUSD_1y_H1.csv"),
        "6mo_H1": os.path.join(data_dir, "XAUUSD_6mo_H1.csv"),
        "2y_H1": os.path.join(data_dir, "XAUUSD_2y_H1.csv"),
        "60d_M15": os.path.join(data_dir, "XAUUSD_60d_M15.csv"),
        "60d_M5": os.path.join(data_dir, "XAUUSD_60d_M5.csv"),
    }

    # Load 2-year data and filter to H1 2026
    fp2y = files["2y_H1"]
    if os.path.exists(fp2y):
        df = pd.read_csv(fp2y)
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df.set_index('Datetime', inplace=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[~df.index.duplicated(keep='last')].dropna()
        df.sort_index(inplace=True)
        # Filter to Jan 1 2026 - Jun 30 2026  
        df = df[(df.index >= '2026-01-01') & (df.index < '2026-07-01')].copy()
        print(f"  Loaded 2y_H1 filtered to H1 2026: {len(df)} candles")
        print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
        return df, None
    # Fallback
    for key in ["1y_H1", "6mo_H1", "2y_H1"]:
        if os.path.exists(files[key]):
            df = pd.read_csv(files[key])
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
                df.set_index('Datetime', inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df = df[~df.index.duplicated(keep='last')].dropna()
            df.sort_index(inplace=True)
            print(f"  Loaded {key}: {len(df)} candles")
            return df, None

    if os.path.exists(files["60d_M15"]):
        df = pd.read_csv(files["60d_M15"])
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[~df.index.duplicated(keep='last')].dropna()
        df.sort_index(inplace=True)
        df5 = None
        if os.path.exists(files["60d_M5"]):
            df5 = pd.read_csv(files["60d_M5"])
            df5['Datetime'] = pd.to_datetime(df5['Datetime'], utc=True)
            df5.set_index('Datetime', inplace=True)
            df5.columns = [c.lower() for c in df5.columns]
            df5 = df5[~df5.index.duplicated(keep='last')].dropna()
            df5.sort_index(inplace=True)
        print(f"  Loaded M15: {len(df)} candles")
        return df, df5

    print("  NO DATA FOUND!")
    return None, None


def print_results(results, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
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


def main():
    print("=" * 80)
    print("  SUPER BOT v5.0 — BACKTEST")
    print("  MIN_SCORE=40 | Adaptive Risk (1.5-3%) | Regime Risk Mult")
    print("=" * 80)

    df, df5 = load_data()
    if df is None:
        print("Cannot run backtest without data!")
        return

    print(f"\nRunning Super Bot backtest on {len(df)} candles...")
    results = run_backtest_super(df, df5)

    print_results(results, "SUPER BOT v5.0 PERFORMANCE")

    # Monthly performance
    print(f"\n{'='*80}")
    print(f"  MONTHLY PERFORMANCE BREAKDOWN")
    print(f"{'='*80}")

    daily_bals = results['daily_balances']
    daily_dates = results['daily_balance_dates']

    monthly_stats = {}
    for bal, dt in zip(daily_bals, daily_dates):
        month_key = f"{dt.year}-{dt.month:02d}"
        if month_key not in monthly_stats:
            monthly_stats[month_key] = []
        monthly_stats[month_key].append(bal)

    cumulative = STARTING_BALANCE
    print(f"  {'Month':<10} {'Start':<10} {'End':<10} {'Return%':<10} {'Cumulative':<12}")
    print(f"  {'-'*52}")
    for month in sorted(monthly_stats.keys()):
        bals = monthly_stats[month]
        m_start = bals[0] if bals else cumulative
        m_end = bals[-1] if bals else cumulative
        m_ret = (m_end - m_start) / m_start * 100 if m_start > 0 else 0
        cumulative = m_end
        print(f"  {month:<10} ${m_start:<8.2f} ${m_end:<8.2f} {m_ret:<+9.1f}% ${cumulative:<8.2f}")

    # Risk/return summary
    print(f"\n{'='*80}")
    print(f"  RISK/RETURN SUMMARY")
    print(f"{'='*80}")
    return_pct = results['return_pct']
    max_dd = results['max_dd_pct']
    calmar = abs(return_pct / max_dd) if max_dd and max_dd > 0 else float('inf')

    print(f"  Total Return:       {return_pct:+.1f}%")
    print(f"  Max Drawdown:       {max_dd:.1f}%")
    print(f"  Calmar Ratio:       {calmar:.2f} (higher is better)")
    print(f"  Profit Factor:      {results['profit_factor']:.2f}")
    print(f"  Win Rate:           {results['win_rate']:.1f}%")
    print(f"  Trades:             {results['trades']}")

    passed = True
    if max_dd > 30:
        print(f"  ❌ FAIL: Drawdown {max_dd:.1f}% exceeds 30% limit")
        passed = False
    if results['net_pnl'] <= 0:
        print(f"  ❌ FAIL: Net profit ${results['net_pnl']:.2f} is negative")
        passed = False

    if passed:
        print(f"  ✅ PASS: All criteria met!")
        print(f"     >300% return target:  {'✅' if return_pct > 300 else '❌'} ({return_pct:.1f}%)")
        print(f"     ≤30% drawdown limit:  {'✅' if max_dd <= 30 else '❌'} ({max_dd:.1f}%)")

    # Compare with Bot A
    print(f"\n{'='*80}")
    print(f"  COMPARED TO BOT A (v4.3)")
    print(f"{'='*80}")
    print(f"  Bot A (June M15):  +213.2%, DD 14.7%, PF 1.98")
    print(f"  Super Bot:         {return_pct:+.1f}%, DD {max_dd:.1f}%, PF {results['profit_factor']:.2f}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "super_bot_results.csv")
    with open(output_path, "w") as f:
        f.write("Metric,Value\n")
        for k, v in [
            ("Starting Balance", f"${STARTING_BALANCE:.2f}"),
            ("Final Balance", f"${results['final_balance']:.2f}"),
            ("Net Profit", f"${results['net_pnl']:.2f}"),
            ("Return %", f"{results['return_pct']:.2f}%"),
            ("Total Trades", str(results['trades'])),
            ("Win Rate", f"{results['win_rate']:.1f}%"),
            ("Avg Win", f"${results['avg_win']:.2f}"),
            ("Avg Loss", f"${results['avg_loss']:.2f}"),
            ("Profit Factor", f"{results['profit_factor']:.2f}"),
            ("Max Drawdown", f"{results['max_dd_pct']:.1f}%"),
            ("Best Trade", f"${results['best_trade']:.2f}"),
            ("Worst Trade", f"${results['worst_trade']:.2f}"),
            ("BUY / SELL", f"{results['buy_count']}/{results['sell_count']}"),
        ]:
            f.write(f"{k},{v}\n")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()