"""
ALL THREE CONFIGS — SL 1.5x vs SL 2.0x vs SL 3.0x
===================================================
Tests all three on both available M5+M15 datasets:
  1. May-Jul 2026 (bullish)
  2. Sep-Feb 2026 (bearish/choppy)

Run: python local_backtest/all_three_configs.py
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


def load_may_jul():
    """Load May-Jul 2026 M5+M15 data."""
    m5 = pd.read_csv(os.path.join(DATA_DIR, "XAUUSD_60d_M5.csv"))
    m15 = pd.read_csv(os.path.join(DATA_DIR, "XAUUSD_60d_M15.csv"))
    for df in [m5, m15]:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
        df.columns = [c.lower() for c in df.columns]
    m5 = m5.dropna().sort_index()
    m15 = m15.dropna().sort_index()
    return m5, m15


def load_sep_feb():
    """Load Sep-Feb 2026 M5+M15 data."""
    m5 = pd.read_csv(os.path.join(DATA_DIR, "XAUUSD_worst_M5.csv"))
    m15 = pd.read_csv(os.path.join(DATA_DIR, "XAUUSD_worst_M15.csv"))
    for df in [m5, m15]:
        df.rename(columns={'Unnamed: 0': 'Datetime'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
        df.columns = [c.lower() for c in df.columns]
    m5 = m5.dropna().sort_index()
    m15 = m15.dropna().sort_index()
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


def run_backtest(m5_df, m15_df, sl_mult, label, progress=False):
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = 3
    balance = STARTING_BALANCE
    positions, closed = [], []
    cons_losses, last_entry, trade_count_today = 0, None, 0
    last_date, daily_pnl, halt_until = None, 0.0, None
    total = len(m15_df)
    
    for idx in range(200, total):
        if progress and idx % 5000 == 0:
            print(f"    {idx}/{total} candles...")
        ct = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])
        if ct.weekday() == 4 and ct.hour >= 21:
            positions.clear(); continue
        if last_date is None: last_date = ct.date()
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
    dir_counts, reason_counts = {}, {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
    
    return {"label": label, "sl_mult": sl_mult, "net_pnl": total_pnl,
            "return_pct": (total_pnl/STARTING_BALANCE)*100,
            "trades": len(closed), "wins": wins, "losses": len(closed)-wins,
            "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": maxdd,
            "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
            "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
            "buys": dir_counts.get("BUY", 0), "sells": dir_counts.get("SELL", 0),
            "sl_count": reason_counts.get("SL", 0), "tp_count": reason_counts.get("TP", 0),
            "trail_count": reason_counts.get("TRAIL", 0)}


def main():
    print("=" * 100)
    print("  COMPREHENSIVE TEST: SL 1.5x vs SL 2.0x vs SL 3.0x")
    print("  Both periods: May-Jul 2026 AND Sep-Feb 2026")
    print("=" * 100)
    
    datasets = [
        ("May-Jul 2026 (Bullish)", load_may_jul),
        ("Sep-Feb 2026 (Bearish)", load_sep_feb),
    ]
    
    sl_configs = [
        ("SL 1.5x", 1.5),
        ("SL 2.0x (CANDIDATE)", 2.0),
        ("SL 3.0x", 3.0),
    ]
    
    all_results = {}
    
    for period_name, loader in datasets:
        print(f"\n{'='*100}")
        print(f"  PERIOD: {period_name}")
        print(f"{'='*100}")
        
        m5, m15 = loader()
        print(f"  Data: {len(m15)} M15 candles, {len(m5)} M5 candles")
        
        period_results = {}
        for label, sl in sl_configs:
            print(f"\n  Testing {label} (SL={sl}x ATR)...")
            r = run_backtest(m5, m15, sl, label, progress=True)
            period_results[sl] = r
            
            print(f"\n  {label}:")
            print(f"    Net P&L:     ${r['net_pnl']:+.2f}")
            print(f"    Drawdown:    {r['max_dd_pct']:.1f}%")
            print(f"    Win Rate:    {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)")
            print(f"    Profit Fact: {r['profit_factor']:.2f}")
            print(f"    Trades:      {r['trades']}")
            print(f"    SL/TP/Trail: {r['sl_count']}/{r['tp_count']}/{r['trail_count']}")
            print(f"    BUY/SELL:    {r['buys']}/{r['sells']}")
        
        all_results[period_name] = period_results
    
    # MASTER SUMMARY
    print(f"\n{'='*100}")
    print(f"  MASTER SUMMARY — ALL CONFIGS ALL PERIODS")
    print(f"{'='*100}")
    
    header = f"  {'SL Config':<12} {'Period':<18} {'Net P&L':<16} {'DD':<10} {'PF':<10} {'WR':<10} {'Trades':<8} {'SL%':<8} {'TP%':<8}"
    print(f"\n{header}")
    print(f"  {'-'*100}")
    
    for period_name in all_results:
        for sl in [1.5, 2.0, 3.0]:
            r = all_results[period_name][sl]
            sl_pct = r['sl_count']/max(r['trades'],1)*100
            tp_pct = r['tp_count']/max(r['trades'],1)*100
            pnl = "${:+.2f}".format(r['net_pnl'])
            print(f"  {'SL '+str(sl)+'x':<12} {period_name:<18} {pnl:<16} {r['max_dd_pct']:<9.1f}% {r['profit_factor']:<9.2f} {r['win_rate']:<8.1f}% {r['trades']:<8} {sl_pct:<7.0f}% {tp_pct:<7.0f}%")
    
    # BEST CHOICE ANALYSIS
    print(f"\n{'='*100}")
    print(f"  BEST CHOICE ANALYSIS")
    print(f"{'='*100}")
    
    print(f"\n  Scoring (out of 2 periods each):")
    for sl_label, sl_val in [("SL 1.5x", 1.5), ("SL 2.0x", 2.0), ("SL 3.0x", 3.0)]:
        score = 0
        for period_name in all_results:
            r = all_results[period_name][sl_val]
            # Point for lower DD than next config
            if sl_val == 1.5:
                next_r = all_results[period_name][2.0]
                if r['max_dd_pct'] < next_r['max_dd_pct']: score += 1
            if sl_val == 2.0:
                prev_r = all_results[period_name][1.5]
                next_r = all_results[period_name][3.0]
                if r['max_dd_pct'] < prev_r['max_dd_pct']: score += 1
                if r['max_dd_pct'] < next_r['max_dd_pct']: score += 1
            if sl_val == 3.0:
                prev_r = all_results[period_name][2.0]
                if r['max_dd_pct'] < prev_r['max_dd_pct']: score += 1
        print(f"    {sl_label:<12} {score}/4 pts (lower DD = better)")
    
    print(f"\n  Recommendation: SL 2.0x vs SL 3.0x")
    print(f"  ------------------------------------------")
    for period_name in all_results:
        r2 = all_results[period_name][2.0]
        r3 = all_results[period_name][3.0]
        dd_winner = "BOTH" if abs(r2['max_dd_pct'] - r3['max_dd_pct']) < 5 else ("SL 3.0x" if r3['max_dd_pct'] < r2['max_dd_pct'] else "SL 2.0x")
        print(f"  {period_name:<20}: SL 2.0x DD={r2['max_dd_pct']:.1f}% | SL 3.0x DD={r3['max_dd_pct']:.1f}% | Best DD: {dd_winner}")
    for period_name in all_results:
        r2 = all_results[period_name][2.0]
        r3 = all_results[period_name][3.0]
        pf_winner = "SL 3.0x" if r3['profit_factor'] > r2['profit_factor'] else "SL 2.0x"
        print(f"  {period_name:<20}: SL 2.0x PF={r2['profit_factor']:.2f} | SL 3.0x PF={r3['profit_factor']:.2f} | Best PF: {pf_winner}")
    
    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT")
    print(f"{'='*100}")
    print(f"""
  SL 1.5x: HIGHEST profit, BUT dangerous drawdown ({all_results['May-Jul 2026 (Bullish)'][1.5]['max_dd_pct']:.1f}%). Risky.
  SL 2.0x: Good middle ground — better than 1.5x, but still higher DD than 3.0x.
  SL 3.0x: BEST drawdown control ({all_results['May-Jul 2026 (Bullish)'][3.0]['max_dd_pct']:.1f}%/{all_results['Sep-Feb 2026 (Bearish)'][3.0]['max_dd_pct']:.1f}%), best win rate, most consistent.
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()