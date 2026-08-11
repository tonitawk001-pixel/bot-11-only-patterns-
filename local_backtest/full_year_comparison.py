"""
FULL YEAR COMPARISON — SL 1.5x vs SL 3.0x
===========================================
Uses existing 1-year H1 data for a full-year comparison.
H1 data: 5905 candles (2025-07-01 to 2026-07-15)

Run: python local_backtest/full_year_comparison.py
"""

import os, sys, warnings
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STARTING_BALANCE = 500.00


def compute_adx(high, low, close, period=14):
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    up_move = high - high.shift(); down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def get_risk_pct(balance):
    tiers = [(0, 500, 2.0), (500, 2000, 2.5), (2000, 10000, 3.0), (10000, float('inf'), 4.0)]
    for lo, hi, pct in tiers:
        if lo < balance <= hi:
            return pct / 100.0
    return 0.02


def run_backtest(df, sl_mult, label):
    """Run on H1 data with given SL multiplier."""
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = 3

    balance = STARTING_BALANCE
    positions = []
    daily_pnl = 0.0
    cons_losses = 0
    halt_until = None
    last_entry = None
    last_date = None
    closed = []
    trade_count_today = 0

    for idx in range(200, len(df)):
        ct = df.index[idx]
        price = float(df["close"].iloc[idx])

        if ct.weekday() == 4 and ct.hour >= 21:
            positions.clear()
            continue

        if last_date is None:
            last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0
            last_date = ct.date()
            trade_count_today = 0

        if halt_until and ct < halt_until:
            continue
        if daily_pnl <= -balance * 0.03:
            continue

        m5_window = df.iloc[max(0, idx-200):idx+1].copy()
        m15_window = df.iloc[max(0, idx-500):idx+1].copy()
        if len(m5_window) < 50 or len(m15_window) < 50:
            continue

        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None:
            continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0:
            continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5:
            continue

        # Position management
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
                ns = price - atr_val * 0.3 if direction == "BUY" else price + atr_val * 0.3
                if direction == "BUY" and ns > sl + 0.5: p["sl"] = round(ns, 2)
                elif direction == "SELL" and ns < sl - 0.5: p["sl"] = round(ns, 2)

            sl, tp = p["sl"], p["tp"]
            hit = False; pnl = 0.0; reason = ""
            if direction == "BUY":
                if tp and price >= tp: pnl = (tp-entry)*pv; reason = "TP"; hit = True
                elif sl and price <= sl: pnl = (sl-entry)*pv; reason = "TRAIL" if sl > entry else "SL"; hit = True
            else:
                if tp and price <= tp: pnl = (entry-tp)*pv; reason = "TP"; hit = True
                elif sl and price >= sl: pnl = (entry-sl)*pv; reason = "TRAIL" if sl < entry else "SL"; hit = True
            if hit:
                pnl -= 0.50 * lot * 100
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                if reason == "SL":
                    cons_losses += 1
                else:
                    cons_losses = 0
            else: surviving.append(p)
        positions = surviving

        if len(positions) >= 3: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < 0: continue

        try:
            adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except: adx_val = 0
        tp_mult = 6.0 if adx_val >= 20 else 4.0

        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5_window.tail(20)
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < 30: continue

        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40): continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60): continue
        except: pass

        sd = atr_val * sl_mult
        td = atr_val * tp_mult
        if direction == "BUY": sl = round(price - sd, 2); tp = round(price + td, 2)
        else: sl = round(price + sd, 2); tp = round(price - td, 2)

        base_risk = get_risk_pct(balance)
        risk_amt = balance * base_risk
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * 2.0 if direction == "BUY" else -atr_val * 2.0)
        trade_count_today += 1

        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score, "be_target": be_target, "be": False,
               "_high": price, "_low": price}
        positions.append(pos); last_entry = ct

    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0]
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0]

    peak = STARTING_BALANCE; maxdd = 0
    eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)

    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0

    dir_counts = {}; reason_counts = {}; monthly = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in monthly: monthly[mk] = {"pnl": 0.0, "trades": 0, "wins": 0}
        monthly[mk]["pnl"] += t["pnl"]; monthly[mk]["trades"] += 1
        if t["pnl"] > 0: monthly[mk]["wins"] += 1

    return {
        "label": label, "net_pnl": total_pnl,
        "return_pct": (total_pnl/STARTING_BALANCE)*100,
        "trades": len(closed), "wins": wins, "losses": len(closed)-wins,
        "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": maxdd,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "buys": dir_counts.get("BUY", 0), "sells": dir_counts.get("SELL", 0),
        "sl_count": reason_counts.get("SL", 0), "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
        "monthly": monthly,
    }


def main():
    print("=" * 90)
    print("  FULL YEAR COMPARISON — SL 1.5x vs SL 3.0x")
    print("  Using 1-year H1 data (exact, not resampled)")
    print("=" * 90)

    fp = os.path.join(DATA_DIR, "XAUUSD_1y_H1.csv")
    if not os.path.exists(fp):
        fp = os.path.join(DATA_DIR, "XAUUSD_6mo_H1.csv")
    if not os.path.exists(fp):
        print("  ERROR: No H1 data found.")
        return

    df = pd.read_csv(fp)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)
    print(f"\n  Data: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    configs = [
        ("SL 1.5x ATR (OLD)", 1.5),
        ("SL 3.0x ATR (NEW)", 3.0),
    ]

    results = []
    for label, sl in configs:
        print(f"\n  Running: {label}...")
        r = run_backtest(df, sl, label)
        results.append(r)

        print(f"\n  {r['label']}")
        print(f"  {'='*50}")
        print(f"  Net P&L:          ${r['net_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
        print(f"  Max Drawdown:     {r['max_dd_pct']:.1f}%")
        print(f"  Profit Factor:    {r['profit_factor']:.2f}")
        print(f"  Win Rate:         {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)")
        print(f"  Trades:           {r['trades']}")
        print(f"  Avg Win/Loss:     ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
        print(f"  Exit:             SL={r['sl_count']} TP={r['tp_count']} Trail={r['trail_count']}")
        print(f"  Direction:        BUY={r['buys']} SELL={r['sells']}")

        print(f"\n  Monthly breakdown:")
        for mk in sorted(r['monthly'].keys()):
            md = r['monthly'][mk]
            wr = md['wins']/md['trades']*100 if md['trades'] > 0 else 0
            print(f"    {mk}: {md['trades']:3d}T {md['wins']:2d}W ${md['pnl']:+8.2f} ({wr:.0f}% WR)")

    # Comparison
    print(f"\n{'='*90}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*90}")
    print(f"\n  {'Metric':<35} {'SL 1.5x (OLD)':<22} {'SL 3.0x (NEW)':<22} {'WINNER'}")
    print(f"  {'-'*90}")

    metrics = [
        ("Net Profit ($)", "net_pnl", "${:+.2f}", True),
        ("Return (%)", "return_pct", "{:+.1f}%", True),
        ("Max Drawdown (%)", "max_dd_pct", "{:.1f}%", False),
        ("Profit Factor", "profit_factor", "{:.2f}", True),
        ("Win Rate (%)", "win_rate", "{:.1f}%", True),
        ("Total Trades", "trades", "{:d}", None),
        ("Avg Win ($)", "avg_win", "${:.2f}", True),
        ("Avg Loss ($)", "avg_loss", "${:.2f}", False),
    ]

    for name, key, fmt, higher_better in metrics:
        vals = [fmt.format(r[key]) for r in results]
        if higher_better is not None:
            winner = "A" if results[0][key] > results[1][key] else "B" if results[1][key] > results[0][key] else "="
            if not higher_better:
                winner = "B" if results[0][key] > results[1][key] else "A" if results[1][key] > results[0][key] else "="
        else:
            winner = "-"
        print(f"  {name:<35} {vals[0]:<22} {vals[1]:<22} {winner}")

    v0 = f"{results[0]['buys']}/{results[0]['sells']}"
    v1 = f"{results[1]['buys']}/{results[1]['sells']}"
    print(f"  {'BUY/SELL':<35} {v0:<22} {v1:<22} {'=' if results[0]['buys']==results[1]['buys'] else ('B' if results[1]['sells']>results[0]['sells'] else 'A')}")

    v0 = f"{results[0]['sl_count']}/{results[0]['tp_count']}/{results[0]['trail_count']}"
    v1 = f"{results[1]['sl_count']}/{results[1]['tp_count']}/{results[1]['trail_count']}"
    # Lower SL is better
    sl_winner = "B" if results[1]['sl_count'] < results[0]['sl_count'] else "A"
    print(f"  {'SL/TP/Trail':<35} {v0:<22} {v1:<22} {sl_winner}")

    # Count winner points
    a_total = results[0]
    b_total = results[1]
    a_pts = 0; b_pts = 0

    # Drawdown (lower is better)
    if a_total['max_dd_pct'] < b_total['max_dd_pct']: a_pts += 1
    else: b_pts += 1

    # Profit factor (higher is better)
    if a_total['profit_factor'] > b_total['profit_factor']: a_pts += 1
    else: b_pts += 1

    # Win rate (higher is better)
    if a_total['win_rate'] > b_total['win_rate']: a_pts += 1
    else: b_pts += 1

    # SL rate (lower is better)
    a_sl_pct = a_total['sl_count']/max(a_total['trades'], 1)
    b_sl_pct = b_total['sl_count']/max(b_total['trades'], 1)
    if a_sl_pct < b_sl_pct: a_pts += 1
    else: b_pts += 1

    # Monthly profitability (more green months)
    a_green = sum(1 for mk in a_total['monthly'] if a_total['monthly'][mk]['pnl'] > 0)
    b_green = sum(1 for mk in b_total['monthly'] if b_total['monthly'][mk]['pnl'] > 0)
    if a_green > b_green: a_pts += 1
    elif b_green > a_green: b_pts += 1

    print(f"\n{'='*90}")
    print("  FINAL SCORING")
    print(f"{'='*90}")
    print(f"  SL 1.5x: {a_pts} pts | SL 3.0x: {b_pts} pts")

    if a_pts >= b_pts:
        winner = "SL 1.5x ATR"
        wr = results[0]
    else:
        winner = "SL 3.0x ATR"
        wr = results[1]

    print(f"\n  ★ RECOMMENDED: {winner}")
    print(f"    Return: {wr['return_pct']:+.1f}% | DD: {wr['max_dd_pct']:.1f}% | PF: {wr['profit_factor']:.2f} | WR: {wr['win_rate']:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()