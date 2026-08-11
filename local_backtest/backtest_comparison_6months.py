"""
V22 Gold Scalping — 6-Month COMPARISON BACKTEST
================================================
Tests BOTH bots on identical H1 data (Jan 15 - Jul 15, 2026):

BOT A (Current Bot - v4.3):
  - Fixed MIN_SCORE=45, flat 2% risk, ADX-based TP override
  - No market regime detection
  - Trailing at 0.7x ATR, breakeven at 2x ATR

BOT B (Copy Bot - v4.4):
  - Market regime detection (ADX, EMA alignment, ATR percentile)
  - Dynamic scoring (50-75), dynamic risk (0.5%-2%)
  - Dynamic TP (2.0x-5.0x based on regime + ADX override)
  - Same trailing/breakeven as Bot A

Run: python local_backtest/backtest_comparison_6months.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# ===== SHARED CONFIG =====
SYMBOL = "XAUUSD"
MAX_POSITIONS = 1
MIN_ATR = 1.0
TRADE_HOURS_START = 8
TRADE_HOURS_END = 22
TP_ATR_MULT = 3.5
TP_ATR_MULT_TREND = 5.0
SL_ATR_MULT = 1.5
BE_ATR_MULT = 2.0
TRAIL_ATR_MULT = 0.7
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
ENTRY_COOLDOWN_MINUTES = 30
DAILY_LOSS_PCT = 0.03
SPREAD_COST_PIP = 0.50
ADX_TREND_THRESHOLD = 25
MAX_TRADES_PER_DAY = 10
STARTING_BALANCE = 304.99
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ===== BOT A CONFIG (Current Bot v4.3) =====
# Fixed: MIN_SCORE=45, 2% flat risk, ADX TP override

# ===== BOT B CONFIG (Copy Bot v4.4) =====
REGIME_PARAMS = {
    "strong_trend":    {"score": 50, "tp": TP_ATR_MULT_TREND, "risk": 0.02},
    "breakout":        {"score": 50, "tp": TP_ATR_MULT_TREND, "risk": 0.02},
    "weak_trend":      {"score": 60, "tp": None,              "risk": 0.02},
    "range":           {"score": 70, "tp": 2.5,               "risk": 0.01},
    "sideways":        {"score": 70, "tp": 2.5,               "risk": 0.01},
    "high_volatility": {"score": 75, "tp": 2.0,               "risk": 0.005},
    "low_volatility":  {"score": 70, "tp": TP_ATR_MULT,      "risk": 0.01},
}


# ===== SHARED HELPERS =====

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

def detect_regime(ind5, ind15, m15w):
    """Simplified regime detection matching MarketRegimeDetector logic."""
    try:
        adx_s = ind15.get("adx") if ind15 else None
        if adx_s is not None and len(adx_s) > 0:
            adx_val = float(adx_s.iloc[-1]) if not np.isnan(float(adx_s.iloc[-1])) else 0
        else:
            adx_val = 0
    except:
        adx_val = 0

    emas = ind15.get("emas", pd.DataFrame()) if ind15 else pd.DataFrame()
    ema_bullish = False
    if not emas.empty and "EMA_20" in emas.columns and "EMA_50" in emas.columns:
        try:
            ema20 = float(emas["EMA_20"].iloc[-1]); ema50 = float(emas["EMA_50"].iloc[-1])
            ema_bullish = ema20 > ema50
        except: pass

    atr_s = ind5.get("atr") if ind5 else None
    atr_pct = 50.0
    if atr_s is not None and len(atr_s) >= 20:
        try:
            curr = float(atr_s.iloc[-1])
            hist = [float(x) for x in atr_s.iloc[-21:-1] if not np.isnan(float(x))]
            if hist and curr > 0:
                atr_pct = sum(1 for x in hist if x < curr) / len(hist) * 100
        except: pass

    if atr_pct >= 90: return "high_volatility"
    if adx_val >= 30 and ema_bullish: return "strong_trend"
    if adx_val >= 30: return "strong_trend"
    if adx_val >= 20: return "weak_trend"
    if atr_pct <= 20: return "low_volatility"
    return "sideways"


def get_regime_params(market_regime, adx_val=None):
    params = REGIME_PARAMS.get(market_regime, {"score": 60, "tp": TP_ATR_MULT, "risk": 0.02})
    min_score = params["score"]; risk_pct = params["risk"]; tp_mult = params["tp"]
    if tp_mult is None:
        tp_mult = TP_ATR_MULT_TREND if (adx_val and adx_val >= ADX_TREND_THRESHOLD) else TP_ATR_MULT
    if adx_val and adx_val >= ADX_TREND_THRESHOLD and tp_mult < TP_ATR_MULT_TREND:
        tp_mult = TP_ATR_MULT_TREND
    return min_score, tp_mult, risk_pct


def run_single_backtest(label, min_score_fixed, use_regime, m5w, m15w):
    """
    Run a backtest using H1 data for both M5 and M15.
    
    Args:
        label: Name for this backtest
        min_score_fixed: Fixed min score (for Bot A)
        use_regime: If True, use regime detection (Bot B); if False, fixed params (Bot A)
    
    Returns:
        dict with all performance metrics
    """
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
    total_entries = 0
    regime_counts = {}
    trade_count_today = 0
    current_trade_day = None

    for i in range(200, len(m15w)):
        ct = m15w.index[i]
        price = float(m15w["close"].iloc[i])

        if not in_session(ct): continue

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

        # Build indicator windows - use same data for both timeframes on H1
        m5_window = m5w.iloc[max(0, i-200):i+1].copy()
        m15_window = m15w.iloc[max(0, i-500):i+1].copy()

        if len(m5_window) < 50 or len(m15_window) < 50: continue

        ind5 = compute_all_indicators(m5_window)
        ind15 = compute_all_indicators(m15_window)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue

        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < MIN_ATR: continue

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

        # ===== STRATEGY LOGIC =====
        if use_regime:
            # Bot B: Regime detection
            market_regime = detect_regime(ind5, ind15, m15_window)
            regime_counts[market_regime] = regime_counts.get(market_regime, 0) + 1

            try:
                adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
                adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
            except: adx_val = 0

            min_score, tp_mult, risk_pct = get_regime_params(market_regime, adx_val)
        else:
            # Bot A: Fixed params
            market_regime = "fixed"
            try:
                adx_series = compute_adx(m5_window["high"], m5_window["low"], m5_window["close"])
                adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
            except: adx_val = 0

            min_score = min_score_fixed  # 45
            tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT
            risk_pct = 0.02  # 2% flat

        # Run strategy
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5_window.tail(20)
            result = strategy.analyze(m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < min_score: continue

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

        # Risk-based lot sizing
        risk_amt = balance * risk_pct
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)
        total_entries += 1
        trade_count_today += 1

        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score, "be_target": be_target, "be": False,
               "_high": price, "_low": price,
               "regime": market_regime, "adx": adx_val, "risk_pct": risk_pct}
        positions.append(pos); last_entry = ct

    # Build results
    total_pnl = 0; wins = 0; losses = 0; win_pnls = []; loss_pnls = []
    dir_counts = {}; reason_counts = {}; daily_pnls = {}; regime_trades = {}

    for t in closed:
        pnl = t["pnl"]; total_pnl += pnl
        if pnl > 0: wins += 1; win_pnls.append(pnl)
        else: losses += 1; loss_pnls.append(pnl)
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
        d = t["open_time"]; day_key = d.date() if hasattr(d, "date") else str(d)[:10]
        daily_pnls[day_key] = daily_pnls.get(day_key, 0) + pnl
        r = t.get("regime", "unknown")
        regime_trades[r] = regime_trades.get(r, 0) + 1

    peak = STARTING_BALANCE; maxdd = 0; eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq: peak = max(peak, e); dd = (peak - e) / peak * 100; maxdd = max(maxdd, dd)

    # Sort closed by close_time for equity curve
    closed_sorted = sorted(closed, key=lambda x: x.get("close_time", x["open_time"]))
    equity_curve = [STARTING_BALANCE]
    for t in closed_sorted:
        equity_curve.append(equity_curve[-1] + t["pnl"])

    final_bal = STARTING_BALANCE + total_pnl
    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0

    return {
        "label": label,
        "final_balance": final_bal,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(closed) * 100) if closed else 0,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "profit_factor": pf,
        "best_trade": max(t["pnl"] for t in closed) if closed else 0,
        "worst_trade": min(t["pnl"] for t in closed) if closed else 0,
        "max_dd_pct": maxdd,
        "total_entries": total_entries,
        "buy_count": dir_counts.get("BUY", 0),
        "sell_count": dir_counts.get("SELL", 0),
        "reason_counts": reason_counts,
        "regime_counts": regime_counts,
        "regime_trades": regime_trades,
        "daily_pnls": daily_pnls,
        "equity_curve": equity_curve,
        "closed": closed,
    }


def print_summary(results_a, results_b):
    """Print side-by-side comparison."""
    print("\n" + "=" * 100)
    print("  6-MONTH COMPARISON BACKTEST — XAUUSD (H1 Data)")
    print("=" * 100)
    print(f"  Period: Jan 15 - Jul 15, 2026")
    print(f"  Starting Balance: ${STARTING_BALANCE:.2f}")
    print(f"  Data: H1 hourly candles (2 years total, focused on 6mo)")
    print("=" * 100)

    headers = ["Metric", "BOT A (Current)", "BOT B (Copy)", "WINNER"]
    print(f"\n  {'Metric':<28} {'BOT A (Current v4.3)':<20} {'BOT B (Copy v4.4)':<20} {'WINNER':<15}")
    print(f"  {'-'*28} {'-'*20} {'-'*20} {'-'*15}")

    rows = [
        ("Final Balance",     f"${results_a['final_balance']:<8.2f}",  f"${results_b['final_balance']:<8.2f}"),
        ("Net P&L",           f"${results_a['net_pnl']:<+8.2f}",      f"${results_b['net_pnl']:<+8.2f}"),
        ("Return %",          f"{results_a['return_pct']:<+7.2f}%",   f"{results_b['return_pct']:<+7.2f}%"),
        ("Total Trades",      f"{results_a['trades']:<8d}",            f"{results_b['trades']:<8d}"),
        ("Win Rate",          f"{results_a['win_rate']:<7.1f}%",      f"{results_b['win_rate']:<7.1f}%"),
        ("Avg Win",           f"${results_a['avg_win']:<+8.2f}",      f"${results_b['avg_win']:<+8.2f}"),
        ("Avg Loss",          f"${results_a['avg_loss']:<+8.2f}",     f"${results_b['avg_loss']:<+8.2f}"),
        ("Profit Factor",     f"{results_a['profit_factor']:<8.2f}",  f"{results_b['profit_factor']:<8.2f}"),
        ("Max Drawdown",      f"{results_a['max_dd_pct']:<7.1f}%",    f"{results_b['max_dd_pct']:<7.1f}%"),
        ("Best Trade",        f"${results_a['best_trade']:<+8.2f}",   f"${results_b['best_trade']:<+8.2f}"),
        ("Worst Trade",       f"${results_a['worst_trade']:<+8.2f}",  f"${results_b['worst_trade']:<+8.2f}"),
        ("Total Signals",     f"{results_a['total_entries']:<8d}",     f"{results_b['total_entries']:<8d}"),
        ("BUY / SELL",        f"{results_a['buy_count']}/{results_a['sell_count']:<7d}", f"{results_b['buy_count']}/{results_b['sell_count']:<7d}"),
    ]

    for metric, val_a, val_b in rows:
        # Determine winner
        if metric in ("Max Drawdown", "Worst Trade", "Avg Loss"):
            # Lower is better
            val_a_num = float(val_a.replace("$", "").replace("%", ""))
            val_b_num = float(val_b.replace("$", "").replace("%", ""))
            winner = "BOT A" if val_a_num < val_b_num else ("BOT B" if val_b_num < val_a_num else "TIE")
        elif metric in ("Total Trades", "Total Signals"):
            winner = "—"
        elif metric in ("BUY / SELL"):
            winner = "—"
        else:
            # Higher is better
            val_a_num = float(val_a.replace("$", "").replace("%", "").replace("+", ""))
            val_b_num = float(val_b.replace("$", "").replace("%", "").replace("+", ""))
            winner = "BOT A" if val_a_num > val_b_num else ("BOT B" if val_b_num > val_a_num else "TIE")

        print(f"  {metric:<28} {val_a:<20} {val_b:<20} {winner:<15}")

    # Regime breakdown for Bot B
    print(f"\n  {'':-<100}")
    print(f"\n  BOT B (Copy) REGIME BREAKDOWN:")
    print(f"  {'Regime':<20} {'Trades':<10} {'Distribution':<15}")
    for r, c in sorted(results_b['regime_trades'].items(), key=lambda x: -x[1]):
        pct = c / results_b['trades'] * 100 if results_b['trades'] else 0
        print(f"  {r:<20} {c:<10} {pct:<6.1f}% ({results_b['regime_counts'].get(r, 0)} candles)")

    # Regime distribution for Bot B
    if results_b['regime_counts']:
        print(f"\n  BOT B REGIME SAMPLE DISTRIBUTION:")
        for r, c in sorted(results_b['regime_counts'].items(), key=lambda x: -x[1]):
            total = sum(results_b['regime_counts'].values())
            print(f"  {r:<20} {c:<8} ({c/total*100:.1f}%)")

    # Close reasons
    print(f"\n  BOT A CLOSE REASONS: {results_a['reason_counts']}")
    print(f"  BOT B CLOSE REASONS: {results_b['reason_counts']}")

    # Final verdict
    print(f"\n{'='*100}")
    total_return_a = results_a['net_pnl']
    total_return_b = results_b['net_pnl']
    
    print(f"  FINAL VERDICT:")
    print(f"  {'':28} {'BOT A (Current)':<20} {'BOT B (Copy)':<20}")
    print(f"  {'Net P&L':28} ${total_return_a:<+8.2f}        ${total_return_b:<+8.2f}")
    print(f"  {'Win Rate':28} {results_a['win_rate']:<7.1f}%             {results_b['win_rate']:<7.1f}%")
    print(f"  {'Profit Factor':28} {results_a['profit_factor']:<8.2f}              {results_b['profit_factor']:<8.2f}")
    print(f"  {'Max DD':28} {results_a['max_dd_pct']:<7.1f}%             {results_b['max_dd_pct']:<7.1f}%")

    if total_return_a > total_return_b:
        print(f"\n  🏆 WINNER: BOT A (Current Bot v4.3) — Higher net profit")
        print(f"     Net P&L: ${total_return_a:.2f} vs ${total_return_b:.2f}")
    elif total_return_b > total_return_a:
        print(f"\n  🏆 WINNER: BOT B (Copy Bot v4.4) — Higher net profit")
        print(f"     Net P&L: ${total_return_b:.2f} vs ${total_return_a:.2f}")
    else:
        print(f"\n  🤝 TIE — Both bots have identical net profit")


def run_backtest():
    print("=" * 80)
    print("  V22 GOLD SCALPING — 6-MONTH COMPARISON BACKTEST")
    print("  BOT A (Current v4.3) vs BOT B (Copy v4.4)")
    print("=" * 80)

    # Load 2-year H1 data
    h1_path_2y = os.path.join(DATA_DIR, "XAUUSD_2y_H1.csv")
    h1_path_6mo = os.path.join(DATA_DIR, "XAUUSD_6mo_H1.csv")
    
    # Try 2y first, then 6mo
    if os.path.exists(h1_path_2y):
        df = pd.read_csv(h1_path_2y)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
    elif os.path.exists(h1_path_6mo):
        df = pd.read_csv(h1_path_6mo)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
    else:
        print("No data! Run: python local_backtest/download_6months.py")
        return

    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)
    
    # Filter to 6 months: Jan 15 2026 to Jul 15 2026
    start_date = pd.Timestamp("2026-01-15", tz='UTC')
    end_date = pd.Timestamp("2026-07-15", tz='UTC')
    df = df[(df.index >= start_date) & (df.index < end_date)].copy()
    
    if len(df) < 500:
        print(f"Not enough data: {len(df)} candles (need 500+). Try downloading more data.")
        return

    timeframe = "H1"
    print(f"\n  Data: {timeframe} | {len(df)} candles")
    print(f"  Period: {df.index[0].strftime('%Y-%m-%d %H:%M')} -> {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Bot A: MIN_SCORE=45, Fixed 2% risk, ADX TP override")
    print(f"  Bot B: Regime-adaptive, Dynamic score/risk/TP\n")

    # Use H1 for both M5 and M15 views (since no sub-hourly for 6 months)
    m5w = df
    m15w = df

    print("Running Bot A (Current v4.3 - fixed parameters)...")
    results_a = run_single_backtest(
        label="Current Bot A",
        min_score_fixed=45,
        use_regime=False,
        m5w=m5w,
        m15w=m15w,
    )
    print(f"  Done. {results_a['trades']} trades, Net P&L: ${results_a['net_pnl']:.2f}")

    print("\nRunning Bot B (Copy v4.4 - regime adaptive)...")
    results_b = run_single_backtest(
        label="Copy Bot B",
        min_score_fixed=60,
        use_regime=True,
        m5w=m5w,
        m15w=m15w,
    )
    print(f"  Done. {results_b['trades']} trades, Net P&L: ${results_b['net_pnl']:.2f}")

    # Print comparison
    print_summary(results_a, results_b)

    # Save results to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_6mo_results.csv")
    with open(output_path, "w") as f:
        f.write("Metric,BOT A (Current v4.3),BOT B (Copy v4.4)\n")
        for metric, val_a, val_b in [
            ("Final Balance", f"${results_a['final_balance']:.2f}", f"${results_b['final_balance']:.2f}"),
            ("Net P&L", f"${results_a['net_pnl']:.2f}", f"${results_b['net_pnl']:.2f}"),
            ("Return %", f"{results_a['return_pct']:.2f}%", f"{results_b['return_pct']:.2f}%"),
            ("Total Trades", f"{results_a['trades']}", f"{results_b['trades']}"),
            ("Win Rate", f"{results_a['win_rate']:.1f}%", f"{results_b['win_rate']:.1f}%"),
            ("Avg Win", f"${results_a['avg_win']:.2f}", f"${results_b['avg_win']:.2f}"),
            ("Avg Loss", f"${results_a['avg_loss']:.2f}", f"${results_b['avg_loss']:.2f}"),
            ("Profit Factor", f"{results_a['profit_factor']:.2f}", f"{results_b['profit_factor']:.2f}"),
            ("Max Drawdown", f"{results_a['max_dd_pct']:.1f}%", f"{results_b['max_dd_pct']:.1f}%"),
            ("Best Trade", f"${results_a['best_trade']:.2f}", f"${results_b['best_trade']:.2f}"),
            ("Worst Trade", f"${results_a['worst_trade']:.2f}", f"${results_b['worst_trade']:.2f}"),
            ("Total Signals", f"{results_a['total_entries']}", f"{results_b['total_entries']}"),
            ("BUY / SELL", f"{results_a['buy_count']}/{results_a['sell_count']}", f"{results_b['buy_count']}/{results_b['sell_count']}"),
        ]:
            f.write(f"{metric},{val_a},{val_b}\n")

    print(f"\n  Results saved to: {output_path}")

    # Also save all trades
    trades_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_6mo_trades.csv")
    with open(trades_path, "w") as f:
        f.write("Bot,EntryTime,Direction,EntryPrice,ExitPrice,P&L,Reason,Lot,Score,Regime\n")
        for bot_label, results in [("BOT_A", results_a), ("BOT_B", results_b)]:
            for t in results["closed"]:
                entry_t = t["open_time"].strftime("%Y-%m-%d %H:%M") if hasattr(t["open_time"], "strftime") else str(t["open_time"])[:16]
                f.write(f"{bot_label},{entry_t},{t['dir']},{t['entry']:.2f},{t.get('close_price',0):.2f},{t['pnl']:.2f},{t.get('reason','')},{t['lot']:.2f},{t.get('score',0)},{t.get('regime','none')}\n")
    print(f"  All trades saved to: {trades_path}")


if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()