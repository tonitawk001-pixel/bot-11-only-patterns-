"""
RISK LEVEL SWEEP — 1-Year Backtest
====================================
Tests 3 risk profiles on 1-year H1 data to find the sweet spot.

Run: python local_backtest/risk_sweep_1year.py
"""

import os, sys, warnings, time
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = r'c:\visual studio code\Ai-bot-linked-to-Meta-api-MT5-made-by-deepseek'
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import trading_bot.utils.logger as lm
lm.logger.setLevel(logging.CRITICAL)
lm.logger.handlers = []
lm.logger.addHandler(logging.NullHandler())

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

DATA_DIR = os.path.join(_PROJECT_ROOT, "local_backtest", "data")
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
ADX_TREND_THRESHOLD = 20

# 3 risk profiles
RISK_PROFILES = [
    {
        "name": "SAFE (low risk)",
        "min_score": 45, "tp_mult": 5.0, "sl_mult": 1.5, "trail_mult": 0.4, "max_pos": 2,
        "risk_tiers": [(0, 500, 1.0), (500, 2000, 1.5), (2000, 10000, 2.0), (10000, 999999, 2.5)],
    },
    {
        "name": "MEDIUM (balanced)",
        "min_score": 40, "tp_mult": 5.0, "sl_mult": 1.5, "trail_mult": 0.5, "max_pos": 2,
        "risk_tiers": [(0, 500, 1.5), (500, 2000, 2.0), (2000, 10000, 2.5), (10000, 999999, 3.0)],
    },
    {
        "name": "AGGRESSIVE (max profit)",
        "min_score": 35, "tp_mult": 6.0, "sl_mult": 1.5, "trail_mult": 0.5, "max_pos": 2,
        "risk_tiers": [(0, 500, 2.0), (500, 2000, 2.5), (2000, 10000, 3.0), (10000, 999999, 4.0)],
    },
]


def in_session(ct):
    return TRADE_HOURS_START <= ct.hour < TRADE_HOURS_END


def compute_adx(high, low, close, period=14):
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, down = h - h.shift(), l.shift() - l
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=c.index).ewm(span=period, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm, index=c.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def update_positions(positions, price, atr_val, trail_mult):
    surviving = []; closed = []
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
        hit, pnl, reason = False, 0.0, ""
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
    return surviving, closed


def run_test(profile, df):
    """Run backtest with given risk profile on H1 data."""
    ms = profile["min_score"]
    tp_mult = profile["tp_mult"]
    sl_mult = profile["sl_mult"]
    trail_mult = profile["trail_mult"]
    max_pos = profile["max_pos"]
    risk_tiers = profile["risk_tiers"]
    TP_TREND = max(tp_mult * 1.4, 5.0)

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = MAX_TRADES_PER_DAY
    strategy._max_open_positions = max_pos

    balance = STARTING_BALANCE
    positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0; current_trade_day = None

    for i in range(200, len(df)):
        ct = df.index[i]
        price = float(df["close"].iloc[i])

        if not in_session(ct): continue
        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue
        if ct.weekday() == 4 and ct.hour >= 18: continue

        if last_date is None: last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0; current_trade_day = ct.date()
        if current_trade_day is None: current_trade_day = ct.date()
        if ct.date() == current_trade_day: pass
        else: trade_count_today = 0; current_trade_day = ct.date()

        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS_PCT: continue
        if trade_count_today >= MAX_TRADES_PER_DAY: continue

        m5w = df.iloc[max(0, i - 200):i + 1].copy()
        m15w = df.iloc[max(0, i - 500):i + 1].copy()
        if len(m5w) < 50 or len(m15w) < 50: continue

        ind5 = compute_all_indicators(m5w)
        ind15 = compute_all_indicators(m15w)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue
        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5: continue

        positions, new_closed = update_positions(positions, price, atr_val, trail_mult)
        closed.extend(new_closed)
        for t in new_closed:
            pnl = t["pnl"]; daily_pnl += pnl; balance += pnl
            if t["reason"] == "SL":
                cons_losses += 1
                if cons_losses >= HALT_AFTER_LOSSES: halt_until = ct + timedelta(hours=HALT_HOURS); cons_losses = 0
            else: cons_losses = 0

        if cons_losses >= HALT_AFTER_LOSSES and halt_until is None:
            halt_until = ct + timedelta(hours=HALT_HOURS); continue
        if len(positions) >= max_pos: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES: continue

        try:
            adx_s = compute_adx(m5w["high"], m5w["low"], m5w["close"])
            adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val = 0
        tp_use = TP_TREND if adx_val >= ADX_TREND_THRESHOLD else tp_mult

        empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
        try:
            result = strategy.analyze(m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=m5w.tail(20), m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < ms: continue

        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40): continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60): continue
        except: pass

        closes = m15w["close"].values
        if len(closes) >= 200:
            ema200 = pd.Series(closes).ewm(200, adjust=False).mean().values
            if len(ema200) >= 10:
                rising = ema200[-1] > ema200[-10]
                if direction == "BUY" and not rising: continue
                if direction == "SELL" and rising: continue

        sd = atr_val * sl_mult; td = atr_val * tp_use
        if direction == "BUY": sl = round(price - sd, 2); tp = round(price + td, 2)
        else: sl = round(price + sd, 2); tp = round(price - td, 2)

        # Risk from tiers
        risk_pct = 0.02
        for lo, hi, rp in risk_tiers:
            if lo < balance <= hi: risk_pct = rp / 100.0; break

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
        return {"net_pnl": 0, "max_dd": 0, "trades": 0, "win_rate": 0, "pf": 0}

    total_pnl = sum(t["pnl"] for t in closed)
    wins = sum(1 for t in closed if t["pnl"] > 0)
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0]
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0]
    peak = STARTING_BALANCE; maxdd = 0; eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else (float('inf') if win_pnls else 0)
    wr = wins / len(closed) * 100 if closed else 0

    return {
        "net_pnl": total_pnl,
        "max_dd": maxdd,
        "trades": len(closed),
        "win_rate": wr,
        "pf": pf,
        "final_balance": STARTING_BALANCE + total_pnl,
    }


def main():
    print("=" * 80)
    print("  RISK LEVEL SWEEP — 1-Year Backtest")
    print("=" * 80)

    # Load data
    fp = os.path.join(DATA_DIR, "XAUUSD_1y_H1.csv")
    if not os.path.exists(fp):
        print("ERROR: 1-year H1 data not found!")
        return

    df = pd.read_csv(fp)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)
    print(f"Data: {len(df)} candles, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")

    results = []
    for profile in RISK_PROFILES:
        print(f"\n{'='*60}")
        print(f"  Testing: {profile['name']}")
        print(f"  Score={profile['min_score']}, TP={profile['tp_mult']}x, "
              f"SL={profile['sl_mult']}x, Trail={profile['trail_mult']}x, "
              f"MaxPos={profile['max_pos']}")
        print(f"  Risk tiers: {profile['risk_tiers']}")
        print(f"{'='*60}")

        t0 = time.time()
        r = run_test(profile, df)
        elapsed = time.time() - t0

        if r["trades"] == 0:
            print("  No trades!")
            continue

        ret = (r["net_pnl"] / STARTING_BALANCE) * 100
        print(f"  Net P&L:     ${r['net_pnl']:+.2f} ({ret:+.1f}%)")
        print(f"  Max DD:      {r['max_dd']:.1f}%")
        print(f"  Trades:      {r['trades']}")
        print(f"  Win Rate:    {r['win_rate']:.1f}%")
        print(f"  PF:          {r['pf']:.2f}")
        print(f"  Final Bal:   ${r['final_balance']:.2f}")
        print(f"  Time:        {elapsed:.1f}s")
        r["name"] = profile["name"]
        r["return_pct"] = ret
        results.append(r)

    # Summary table
    print(f"\n{'='*100}")
    print(f"  🏆 RISK LEVEL COMPARISON — 1-Year H1 Data")
    print(f"{'='*100}")
    print(f"  {'Profile':<25} {'Net P&L':<12} {'Return%':<12} {'DD%':<8} {'Trades':<8} {'PF':<8} {'Status':<15}")
    print(f"  {'-'*85}")

    best = None
    for r in sorted(results, key=lambda x: x["net_pnl"], reverse=True):
        dd_ok = "✅ ≤30%" if r["max_dd"] <= 30 else "⚠️ >30%"
        print(f"  {r['name']:<25} ${r['net_pnl']:<+8.2f} {r['return_pct']:<+8.1f}%  {r['max_dd']:<6.1f}% {r['trades']:<8} {r['pf']:<8.2f} {dd_ok:<15}")
        if r["max_dd"] <= 30 and (best is None or r["net_pnl"] > best["net_pnl"]):
            best = r

    # Winner
    print(f"\n{'='*80}")
    profiles_with_dd = [r for r in results if r["max_dd"] <= 30]
    if profiles_with_dd:
        winner = max(profiles_with_dd, key=lambda x: x["net_pnl"])
        print(f"  🏆 BEST WITH ≤30% DD: {winner['name']}")
        print(f"     Net P&L: ${winner['net_pnl']:+.2f} ({winner['return_pct']:+.1f}%), DD: {winner['max_dd']:.1f}%")
    else:
        winner = max(results, key=lambda x: x["net_pnl"])
        print(f"  ⚠️ NO profile within 30% DD. Best profit: {winner['name']}")
        print(f"     Net P&L: ${winner['net_pnl']:+.2f} ({winner['return_pct']:+.1f}%), DD: {winner['max_dd']:.1f}%")

    print(f"\n{'='*80}")
    print(f"  SUGGESTED LIVE BOT PARAMETERS:")
    if winner:
        for p in RISK_PROFILES:
            if p["name"] == winner["name"]:
                print(f"  MIN_SCORE = {p['min_score']}")
                print(f"  TP_MULT = {p['tp_mult']}")
                print(f"  SL_MULT = {p['sl_mult']}")
                print(f"  TRAIL_MULT = {p['trail_mult']}")
                print(f"  MAX_POSITIONS = {p['max_pos']}")
                print(f"  RISK_TIERS = {p['risk_tiers']}")
                break
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()