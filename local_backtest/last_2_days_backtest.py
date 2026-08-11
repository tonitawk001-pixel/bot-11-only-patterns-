"""
LAST 2 DAYS BACKTEST — Jul 21-22, 2026
========================================
Uses M1 data resampled to M5+M15 to test the exact days
your VPS bot traded. Compares SL 1.5x vs SL 3.0x.

Run: python local_backtest/last_2_days_backtest.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STARTING_BALANCE = 269.47  # Match VPS starting balance


def load_and_resample():
    """Load M1 data and resample to M5+M15 for Jul 21-22."""
    m1_file = os.path.join(DATA_DIR, "XAUUSD_7d_M1.csv")
    if not os.path.exists(m1_file):
        print("  ERROR: M1 data not found!")
        return None, None
    
    df = pd.read_csv(m1_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()
    
    # Data only goes to Jul 21 15:09 UTC
    print(f"  M1 data: {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    print(f"  (Note: M1 data ends Jul 21 15:09 UTC — no Jul 22 data exists)")
    
    # Resample to M5
    m5 = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    
    # Resample to M15
    m15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    
    print(f"  M5:  {len(m5)} candles ({m5.index[0]} to {m5.index[-1]})")
    print(f"  M15: {len(m15)} candles ({m15.index[0]} to {m15.index[-1]})")
    
    return m5, m15


def run_backtest(m5_df, m15_df, sl_mult, label):
    """Run bot strategy on given data."""
    os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
    import logging
    logging.disable(logging.CRITICAL)
    
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    
    import trading_bot.utils.logger as logger_module
    logger_module.logger.setLevel(logging.CRITICAL)
    logger_module.logger.handlers = []
    logger_module.logger.addHandler(logging.NullHandler())
    
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
    
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = 3
    
    balance = STARTING_BALANCE
    positions = []
    closed = []
    cons_losses = 0
    halt_until = None
    last_entry = None
    trade_count_today = 0
    last_date = None
    daily_pnl = 0.0
    
    # Need more warmup candles for M15 vs M1 resampled
    for idx in range(50, len(m15_df)):
        ct = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])
        
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
        
        m5u = m5_df[m5_df.index <= ct]
        m5w = m5u.tail(500).copy()
        m15w = m15_df.iloc[max(0, idx-500):idx+1].copy()
        if len(m5w) < 30 or len(m15w) < 30:
            continue
        
        ind5 = compute_all_indicators(m5w)
        ind15 = compute_all_indicators(m15w)
        if ind5 is None or ind15 is None:
            continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0:
            continue
        
        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.3:
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
            high_s = pd.Series(m5w["high"].values, index=m5w.index)
            low_s = pd.Series(m5w["low"].values, index=m5w.index)
            close_s = pd.Series(m5w["close"].values, index=m5w.index)
            adx_s = compute_adx_simple(high_s, low_s, close_s)
            adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val = 0
        tp_mult = 6.0 if adx_val >= 20 else 4.0
        
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5w.tail(20)
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
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
        
        risk_amt = balance * 0.02
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
    
    dir_counts = {}; reason_counts = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
    
    return {
        "label": label, "balance": balance,
        "net_pnl": total_pnl, "return_pct": (total_pnl/STARTING_BALANCE)*100,
        "trades": len(closed), "wins": wins, "losses": len(closed)-wins,
        "win_rate": win_rate, "profit_factor": pf, "max_dd_pct": maxdd,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "buys": dir_counts.get("BUY", 0), "sells": dir_counts.get("SELL", 0),
        "sl_count": reason_counts.get("SL", 0), "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
    }


def compute_adx_simple(high, low, close, period=14):
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift(); down = low.shift() - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def main():
    print("=" * 70)
    print("  LAST 2 DAYS BACKTEST — Jul 21-22, 2026")
    print("  Starting Balance: $269.47 (same as VPS)")
    print("  M1 resampled to M5+M15")
    print("=" * 70)
    
    m5, m15 = load_and_resample()
    if m5 is None or len(m5) < 100:
        print("\n  Not enough data. Need at least 100 M5 candles.")
        return
    
    # Use ALL available M1 data (Jul 14-21 15:09 UTC) for backtest
    # First 100 M5 candles used for indicator warmup
    m15_test = m15.copy()
    m5_test = m5.copy()
    
    configs = [
        ("SL 1.5x ATR (OLD)", 1.5),
        ("SL 3.0x ATR (NEW)", 3.0),
    ]
    
    results = []
    for label, sl in configs:
        print(f"\n  Running {label} on Jul 22 data...")
        r = run_backtest(m5_test, m15_test, sl, label)
        results.append(r)
        
        print(f"\n  {r['label']}")
        print(f"  {'='*40}")
        print(f"  Final Balance:      ${r['balance']:.2f}")
        print(f"  Net P&L:            ${r['net_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
        print(f"  Max Drawdown:       {r['max_dd_pct']:.1f}%")
        print(f"  Profit Factor:      {r['profit_factor']:.2f}")
        print(f"  Win Rate:           {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)")
        print(f"  Trades:             {r['trades']}")
        print(f"  Avg Win/Loss:       ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
        print(f"  Exit:               SL={r['sl_count']} TP={r['tp_count']} Trail={r['trail_count']}")
        print(f"  Direction:          BUY={r['buys']} SELL={r['sells']}")
    
    # Comparison
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<30} {'SL 1.5x (OLD)':<20} {'SL 3.0x (NEW)':<20}")
    print(f"  {'-'*70}")
    
    metrics = [
        ("P&L ($)", "net_pnl", "${:+.2f}"),
        ("Return (%)", "return_pct", "{:+.1f}%"),
        ("Drawdown (%)", "max_dd_pct", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Win Rate (%)", "win_rate", "{:.1f}%"),
        ("Trades", "trades", "{:d}"),
        ("Avg Win ($)", "avg_win", "${:.2f}"),
        ("Avg Loss ($)", "avg_loss", "${:.2f}"),
    ]
    for name, key, fmt in metrics:
        vals = [fmt.format(r[key]) for r in results]
        print(f"  {name:<30} {vals[0]:<20} {vals[1]:<20}")
    
    v0 = [f"{r['buys']}/{r['sells']}" for r in results]
    v1 = [f"{r['sl_count']}/{r['tp_count']}/{r['trail_count']}" for r in results]
    print(f"  {'BUY/SELL':<30} {v0[0]:<20} {v0[1]:<20}")
    print(f"  {'SL/TP/Trail':<30} {v1[0]:<20} {v1[1]:<20}")
    
    # Compare with VPS
    print(f"\n{'='*70}")
    print("  VS YOUR VPS TRADES")
    print(f"{'='*70}")
    
    vps_trades = [
        {"pnl": 21.08, "reason": "TP"}, {"pnl": 2.88, "reason": "TRAIL"},
        {"pnl": -0.17, "reason": "TRAIL"}, {"pnl": -0.19, "reason": "TRAIL"},
        {"pnl": 0.65, "reason": "TRAIL"}, {"pnl": 0.61, "reason": "TRAIL"},
        {"pnl": -0.20, "reason": "TRAIL"}, {"pnl": 3.53, "reason": "TRAIL"},
        {"pnl": -5.91, "reason": "SL"}, {"pnl": 9.10, "reason": "TRAIL"},
        {"pnl": 0.98, "reason": "TRAIL"}, {"pnl": 3.26, "reason": "TRAIL"},
        {"pnl": 7.10, "reason": "TRAIL"}, {"pnl": 2.82, "reason": "TRAIL"},
        {"pnl": -6.84, "reason": "SL"},
    ]
    vps_pnl = sum(t["pnl"] for t in vps_trades)
    vps_wins = sum(1 for t in vps_trades if t["pnl"] > 0)
    
    print(f"\n  {'Metric':<30} {'VPS ACTUAL':<20} {'SL 1.5x':<20} {'SL 3.0x':<20}")
    print(f"  {'-'*70}")
    a_pnl_str = "${:+.2f}".format(results[0]["net_pnl"])
    b_pnl_str = "${:+.2f}".format(results[1]["net_pnl"])
    a_dir_str = "{}/{}".format(results[0]["buys"], results[0]["sells"])
    b_dir_str = "{}/{}".format(results[1]["buys"], results[1]["sells"])
    
    print(f"  {'Trades':<30} {'15':<20} {results[0]['trades']:<20} {results[1]['trades']:<20}")
    print(f"  {'P&L':<30} {'${:+.2f}'.format(vps_pnl):<20} {a_pnl_str:<20} {b_pnl_str:<20}")
    print(f"  {'Wins':<30} {vps_wins:<20} {results[0]['wins']:<20} {results[1]['wins']:<20}")
    print(f"  {'BUY/SELL':<30} {'15/0':<20} {a_dir_str:<20} {b_dir_str:<20}")
    
    print(f"\n{'='*70}")
    print("  DONE.")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()