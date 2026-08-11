"""
WORST PERIOD BACKTEST — Sep 2025 to Feb 2026
==============================================
Uses real M5+M15 data from a different period than the May-Jul 2026 test.
This was labeled "worst" — likely a bearish/choppy period for gold.

Run: python local_backtest/worst_period_backtest.py
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


def load_worst_data():
    """Load the 'worst' M5+M15 dataset (Sep 2025 - Feb 2026)."""
    m5_file = os.path.join(DATA_DIR, "XAUUSD_worst_M5.csv")
    m15_file = os.path.join(DATA_DIR, "XAUUSD_worst_M15.csv")
    
    m5 = pd.read_csv(m5_file)
    m5.rename(columns={'Unnamed: 0': 'Datetime'}, inplace=True)
    m5['Datetime'] = pd.to_datetime(m5['Datetime'], utc=True)
    m5.set_index('Datetime', inplace=True)
    m5.columns = [c.lower() for c in m5.columns]
    m5 = m5.dropna().sort_index()
    
    m15 = pd.read_csv(m15_file)
    m15.rename(columns={'Unnamed: 0': 'Datetime'}, inplace=True)
    m15['Datetime'] = pd.to_datetime(m15['Datetime'], utc=True)
    m15.set_index('Datetime', inplace=True)
    m15.columns = [c.lower() for c in m15.columns]
    m15 = m15.dropna().sort_index()
    
    print(f"  Worst M5:  {len(m5)} candles ({m5.index[0]} to {m5.index[-1]})")
    print(f"  Worst M15: {len(m15)} candles ({m15.index[0]} to {m15.index[-1]})")
    return m5, m15


def compute_adx(high, low, close, period=14):
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    up, down = h - h.shift(), l.shift() - l
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=c.index).ewm(span=period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=c.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def get_risk_pct(balance):
    tiers = [(0, 500, 2.0), (500, 2000, 2.5), (2000, 10000, 3.0), (10000, float('inf'), 4.0)]
    for lo, hi, pct in tiers:
        if lo < balance <= hi:
            return pct / 100.0
    return 0.02


def run_backtest(m5_df, m15_df, sl_mult, label):
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = 3
    balance = STARTING_BALANCE
    positions, closed = [], []
    cons_losses, last_entry, trade_count_today = 0, None, 0
    last_date, daily_pnl, halt_until = None, 0.0, None
    
    for idx in range(200, len(m15_df)):
        ct = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])
        if ct.weekday() == 4 and ct.hour >= 21:
            positions.clear(); continue
        if last_date is None:
            last_date = ct.date()
        if ct.date() != last_date:
            daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0
        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * 0.03: continue
        
        m5u = m5_df[m5_df.index <= ct]
        m5w = m5u.tail(500).copy()
        m15w = m15_df.iloc[max(0, idx-500):idx+1].copy()
        if len(m5w) < 50 or len(m15w) < 50: continue
        
        ind5 = compute_all_indicators(m5w)
        ind15 = compute_all_indicators(m15w)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue
        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5: continue
        
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
            hit, pnl, reason = False, 0.0, ""
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
                cons_losses = (cons_losses + 1) if reason == "SL" else 0
            else: surviving.append(p)
        positions = surviving
        
        if len(positions) >= 3: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < 0: continue
        
        try:
            adx_s = compute_adx(m5w["high"], m5w["low"], m5w["close"])
            adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val = 0
        tp_mult = 6.0 if adx_val >= 20 else 4.0
        
        try:
            result = strategy.analyze(
                m1_indicators={"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])},
                m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=m5w.tail(20), m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except: continue
        direction, score = result.get("direction", "NONE"), result.get("setup_score", 0)
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
        
        positions.append({"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
                          "open_time": ct, "score": score, "be_target": be_target, "be": False,
                          "_high": price, "_low": price})
        last_entry = ct
    
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0)
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0]
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0]
    peak, maxdd = STARTING_BALANCE, 0
    eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)
    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    dir_counts, reason_counts, monthly = {}, {}, {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
        mk = t["close_time"].strftime("%Y-%m")
        if mk not in monthly: monthly[mk] = {"pnl": 0.0, "trades": 0, "wins": 0}
        monthly[mk]["pnl"] += t["pnl"]; monthly[mk]["trades"] += 1
        if t["pnl"] > 0: monthly[mk]["wins"] += 1
    
    return {"label": label, "net_pnl": total_pnl, "return_pct": (total_pnl/STARTING_BALANCE)*100,
            "trades": len(closed), "wins": wins, "losses": len(closed)-wins,
            "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": maxdd,
            "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
            "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
            "buys": dir_counts.get("BUY", 0), "sells": dir_counts.get("SELL", 0),
            "sl_count": reason_counts.get("SL", 0), "tp_count": reason_counts.get("TP", 0),
            "trail_count": reason_counts.get("TRAIL", 0), "monthly": monthly}


def main():
    print("=" * 90)
    print("  WORST PERIOD BACKTEST — Sep 2025 to Feb 2026")
    print("  Real M5+M15 data — completely different market conditions")
    print("=" * 90)
    
    m5, m15 = load_worst_data()
    if m5 is None or len(m5) < 500:
        print("  ERROR: Not enough data!"); return
    
    configs = [("SL 1.5x ATR (OLD)", 1.5), ("SL 3.0x ATR (NEW)", 3.0)]
    results = []
    
    for label, sl in configs:
        print(f"\n  Running: {label}...")
        r = run_backtest(m5, m15, sl, label)
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
        print(f"\n  Monthly:")
        for mk in sorted(r['monthly'].keys()):
            md = r['monthly'][mk]; wr = md['wins']/md['trades']*100 if md['trades']>0 else 0
            print(f"    {mk}: {md['trades']:3d}T {md['wins']:2d}W ${md['pnl']:+8.2f} ({wr:.0f}% WR)")
    
    print(f"\n{'='*90}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*90}")
    print(f"\n  {'Metric':<35} {'SL 1.5x':<22} {'SL 3.0x':<22}  WINNER")
    print(f"  {'-'*90}")
    a, b = results[0], results[1]
    for name, key, fmt, higher_better in [
        ("Net Profit ($)", "net_pnl", "${:+.2f}", True),
        ("Return (%)", "return_pct", "{:+.1f}%", True),
        ("Max Drawdown (%)", "max_dd_pct", "{:.1f}%", False),
        ("Profit Factor", "profit_factor", "{:.2f}", True),
        ("Win Rate (%)", "win_rate", "{:.1f}%", True),
        ("Trades", "trades", "{:d}", None),
    ]:
        va, vb = fmt.format(a[key]), fmt.format(b[key])
        if higher_better is not None:
            w = "A" if (a[key] > b[key]) == higher_better or (a[key] < b[key]) != higher_better else "B"
            if not higher_better: w = "A" if a[key] < b[key] else "B"
        else: w = "-"
        print(f"  {name:<35} {va:<22} {vb:<22}  {w}")
    
    v0, v1 = f"{a['buys']}/{a['sells']}", f"{b['buys']}/{b['sells']}"
    print(f"  {'BUY/SELL':<35} {v0:<22} {v1:<22}")
    v0, v1 = f"{a['sl_count']}/{a['tp_count']}/{a['trail_count']}", f"{b['sl_count']}/{b['tp_count']}/{b['trail_count']}"
    print(f"  {'SL/TP/Trail':<35} {v0:<22} {v1:<22}")
    
    print(f"\n{'='*90}")
    print("  COMBINED PERIOD ANALYSIS")
    print(f"{'='*90}")
    # Compare with previous May-Jul results
    print(f"\n  {'Period':<15} {'SL 1.5x DD':<15} {'SL 3.0x DD':<15} {'1.5x PF':<15} {'3.0x PF':<15}")
    print(f"  {'-'*75}")
    
    # We know from previous test:
    may_jul = {"dd15": 53.2, "dd30": 33.7, "pf15": 1.47, "pf30": 1.62}
    sep_feb = {"dd15": a['max_dd_pct'], "dd30": b['max_dd_pct'], "pf15": a['profit_factor'], "pf30": b['profit_factor']}
    
    print(f"  {'May-Jul 2026':<15} {may_jul['dd15']:<15.1f} {may_jul['dd30']:<15.1f} {may_jul['pf15']:<15.2f} {may_jul['pf30']:<15.2f}")
    print(f"  {'Sep-Feb 2026':<15} {sep_feb['dd15']:<15.1f} {sep_feb['dd30']:<15.1f} {sep_feb['pf15']:<15.2f} {sep_feb['pf30']:<15.2f}")
    
    # Scoring across both periods
    a_wins, b_wins = 0, 0
    # Lower DD wins
    if may_jul['dd15'] < may_jul['dd30']:
        a_wins += 1
    else:
        b_wins += 1
    if sep_feb['dd15'] < sep_feb['dd30']:
        a_wins += 1
    else:
        b_wins += 1
    # Higher PF wins
    if may_jul['pf15'] > may_jul['pf30']:
        a_wins += 1
    else:
        b_wins += 1
    if sep_feb['pf15'] > sep_feb['pf30']:
        a_wins += 1
    else:
        b_wins += 1
    
    print(f"\n  Cross-period scoring:")
    print(f"    SL 1.5x: {a_wins}/4 wins")
    print(f"    SL 3.0x: {b_wins}/4 wins")
    
    if a_wins >= b_wins:
        print(f"\n  ★ RECOMMENDED: SL 1.5x ATR")
    else:
        print(f"\n  ★ RECOMMENDED: SL 3.0x ATR")
    
    print(f"\n{'='*90}")
    print("  KEY INSIGHT")
    print(f"{'='*90}")
    print(f"""
  Two very different market periods tested:
  
  1. May-Jul 2026 (strongly bullish gold, +20% move)
     - SL 1.5x: DD 53.2%, PF 1.47
     - SL 3.0x: DD 33.7%, PF 1.62  ✅
     
  2. Sep 2025 - Feb 2026 (bearish/choppy gold)
     - SL 1.5x: DD {a['max_dd_pct']:.1f}%, PF {a['profit_factor']:.2f}
     - SL 3.0x: DD {b['max_dd_pct']:.1f}%, PF {b['profit_factor']:.2f}
  
  The optimal config depends on the market regime.
  In trending markets, SL 3.0x is safer.
  In ranging markets, SL 1.5x may capture more.
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()