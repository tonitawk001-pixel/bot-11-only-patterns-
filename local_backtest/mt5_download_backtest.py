"""
MT5 DOWNLOAD + BACKTEST — Real M5+M15 data from MT5
=====================================================
Downloads fresh M5 and M15 candles from MT5 (no resampling),
then runs the bot's strategy with both old and new configs.

Run: python local_backtest/mt5_download_backtest.py
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

# ===== CONFIG =====
SYMBOL = "XAUUSD"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
DAYS_TO_DOWNLOAD = 60
STARTING_BALANCE = 500.00


def download_from_mt5():
    """Download real M5 and M15 candles from MT5."""
    import MetaTrader5 as mt5
    
    print(f"\n  [MT5] Initializing...")
    if not mt5.initialize():
        print(f"  [MT5] ERROR: initialize() returned False")
        return None, None
    
    term_info = mt5.terminal_info()
    print(f"  [MT5] Connected. Terminal: {term_info.name if hasattr(term_info, 'name') else '?'}")
    
    # Download M5
    print(f"\n  [MT5] Downloading M5 data for {SYMBOL} (last {DAYS_TO_DOWNLOAD} days)...")
    rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 10000)
    if rates_m5 is None or len(rates_m5) == 0:
        print(f"  [MT5] ERROR: No M5 data. Error: {mt5.last_error()}")
        mt5.shutdown()
        return None, None
    
    df_m5 = pd.DataFrame(rates_m5)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s', utc=True)
    df_m5.set_index('time', inplace=True)
    df_m5.columns = [c.lower() for c in df_m5.columns]
    df_m5 = df_m5[~df_m5.index.duplicated(keep='last')].sort_index()
    print(f"  [MT5] M5: {len(df_m5)} candles ({df_m5.index[0]} to {df_m5.index[-1]})")
    
    # Download M15
    print(f"  [MT5] Downloading M15 data for {SYMBOL}...")
    rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 5000)
    if rates_m15 is None or len(rates_m15) == 0:
        print(f"  [MT5] ERROR: No M15 data. Error: {mt5.last_error()}")
        mt5.shutdown()
        return None, None
    
    df_m15 = pd.DataFrame(rates_m15)
    df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s', utc=True)
    df_m15.set_index('time', inplace=True)
    df_m15.columns = [c.lower() for c in df_m15.columns]
    df_m15 = df_m15[~df_m15.index.duplicated(keep='last')].sort_index()
    print(f"  [MT5] M15: {len(df_m15)} candles ({df_m15.index[0]} to {df_m15.index[-1]})")
    
    mt5.shutdown()
    
    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    m5_path = os.path.join(DATA_DIR, f"XAUUSD_fresh_M5.csv")
    m15_path = os.path.join(DATA_DIR, f"XAUUSD_fresh_M15.csv")
    df_m5.to_csv(m5_path)
    df_m15.to_csv(m15_path)
    print(f"  [MT5] Saved: {m5_path}")
    print(f"  [MT5] Saved: {m15_path}")
    
    return df_m5, df_m15


def run_backtest(m15_df, m5_df, label, sl_mult=3.0, risk_mult=1.0):
    """Run exact bot strategy on real M15+M5 data."""
    os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
    import logging
    logging.disable(logging.CRITICAL)
    
    import sys as _sys
    if _PROJECT_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJECT_ROOT)
    
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
    
    for idx in range(200, len(m15_df)):
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
        if len(m5w) < 50 or len(m15w) < 50:
            continue
        
        ind5 = compute_all_indicators(m5w)
        ind15 = compute_all_indicators(m15w)
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
                if direction == "BUY" and price >= p["be_target"]:
                    p["be"] = True; p["sl"] = entry
                elif direction == "SELL" and price <= p["be_target"]:
                    p["be"] = True; p["sl"] = entry
            if p.get("be"):
                ns = price - atr_val * 0.3 if direction == "BUY" else price + atr_val * 0.3
                if direction == "BUY" and ns > sl + 0.5:
                    p["sl"] = round(ns, 2)
                elif direction == "SELL" and ns < sl - 0.5:
                    p["sl"] = round(ns, 2)
            
            sl, tp = p["sl"], p["tp"]
            hit = False; pnl = 0.0; reason = ""
            if direction == "BUY":
                if tp and price >= tp:
                    pnl = (tp-entry)*pv; reason = "TP"; hit = True
                elif sl and price <= sl:
                    pnl = (sl-entry)*pv; reason = "TRAIL" if sl > entry else "SL"; hit = True
            else:
                if tp and price <= tp:
                    pnl = (entry-tp)*pv; reason = "TP"; hit = True
                elif sl and price >= sl:
                    pnl = (entry-sl)*pv; reason = "TRAIL" if sl < entry else "SL"; hit = True
            if hit:
                pnl -= 0.50 * lot * 100
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price; p["close_time"] = ct
                closed.append(p)
                if reason == "SL":
                    cons_losses += 1
                else:
                    cons_losses = 0
            else:
                surviving.append(p)
        positions = surviving
        
        if len(positions) >= 3:
            continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < 0:
            continue
        
        try:
            adx_series = compute_adx_simple(m5w["high"], m5w["low"], m5w["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
        except:
            adx_val = 0
        tp_mult = 6.0 if adx_val >= 20 else 4.0
        
        try:
            empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
            eo = m5w.tail(20)
            result = strategy.analyze(
                m1_indicators=empty_m1, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=eo, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except:
            continue
        
        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)
        if direction == "NONE" or score < 30:
            continue
        
        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40):
                continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60):
                continue
        except:
            pass
        
        sd = atr_val * sl_mult
        td = atr_val * tp_mult
        if direction == "BUY":
            sl = round(price - sd, 2); tp = round(price + td, 2)
        else:
            sl = round(price + sd, 2); tp = round(price - td, 2)
        
        risk_amt = balance * 0.02 * risk_mult
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        be_target = price + (atr_val * 2.0 if direction == "BUY" else -atr_val * 2.0)
        trade_count_today += 1
        
        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score, "be_target": be_target, "be": False,
               "_high": price, "_low": price}
        positions.append(pos)
        last_entry = ct
    
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0]
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0]
    
    peak = STARTING_BALANCE; maxdd = 0
    eq = [STARTING_BALANCE]
    for t in closed:
        eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)
    
    pf = abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf') if win_pnls else 0
    win_rate = wins / len(closed) * 100 if closed else 0
    
    dir_counts = {}
    reason_counts = {}
    monthly = {}
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
        mk = t["close_time"].strftime("%Y-%m-%d")
        if mk not in monthly:
            monthly[mk] = {"pnl": 0.0, "trades": 0, "wins": 0}
        monthly[mk]["pnl"] += t["pnl"]
        monthly[mk]["trades"] += 1
        if t["pnl"] > 0:
            monthly[mk]["wins"] += 1
    
    return {
        "label": label,
        "final_balance": STARTING_BALANCE + total_pnl,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "trades": len(closed), "wins": wins, "losses": len(closed) - wins,
        "win_rate": win_rate,
        "avg_win": (sum(win_pnls)/len(win_pnls)) if win_pnls else 0,
        "avg_loss": (sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0,
        "profit_factor": pf, "max_dd_pct": maxdd,
        "buys": dir_counts.get("BUY", 0), "sells": dir_counts.get("SELL", 0),
        "sl_count": reason_counts.get("SL", 0),
        "tp_count": reason_counts.get("TP", 0),
        "trail_count": reason_counts.get("TRAIL", 0),
        "monthly": monthly,
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


def print_results(r):
    print(f"\n  {r['label']}")
    print(f"  {'='*50}")
    print(f"  Net Profit:      ${r['net_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
    print(f"  Max Drawdown:    {r['max_dd_pct']:.1f}%")
    print(f"  Profit Factor:   {r['profit_factor']:.2f}")
    print(f"  Win Rate:        {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)")
    print(f"  Trades:          {r['trades']}")
    print(f"  Avg Win/Loss:    ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
    print(f"  Exit:            SL={r['sl_count']} TP={r['tp_count']} Trail={r['trail_count']}")
    print(f"  Direction:       BUY={r['buys']} SELL={r['sells']}")
    if r.get('monthly'):
        print(f"\n  Daily breakdown:")
        for dk, dd in sorted(r['monthly'].items()):
            wr = dd['wins']/dd['trades']*100 if dd['trades'] > 0 else 0
            print(f"    {dk}: ${dd['pnl']:+.2f} ({dd['trades']}T, {dd['wins']}W, {wr:.0f}% WR)")


def main():
    print("=" * 70)
    print("  MT5 DOWNLOAD + BACKTEST")
    print("  Real M5+M15 data from MT5 — no resampling")
    print("  Bot code NOT modified")
    print("=" * 70)
    
    # Step 1: Download from MT5
    print(f"\n{'='*70}")
    print("  STEP 1: Download real M5+M15 from MT5")
    print(f"{'='*70}")
    m5, m15 = download_from_mt5()
    
    if m5 is None or m15 is None:
        print("\n  Could not download from MT5. Checking existing data...")
        m5_file = os.path.join(DATA_DIR, "XAUUSD_60d_M5.csv")
        m15_file = os.path.join(DATA_DIR, "XAUUSD_60d_M15.csv")
        if os.path.exists(m5_file) and os.path.exists(m15_file):
            print("  Loading existing M5+M15 data...")
            m5 = pd.read_csv(m5_file, index_col=0, parse_dates=True)
            m15 = pd.read_csv(m15_file, index_col=0, parse_dates=True)
            m5.index = pd.to_datetime(m5.index, utc=True)
            m15.index = pd.to_datetime(m15.index, utc=True)
        else:
            print("  ERROR: No data available.")
            return
    
    # Step 2: Run backtest comparisons
    print(f"\n{'='*70}")
    print("  STEP 2: Run backtest on real MT5 data")
    print(f"{'='*70}")
    
    configs = [
        ("OLD BOT (SL 1.5x)", {"sl_mult": 1.5, "risk_mult": 1.0}),
        ("NEW BOT (SL 3.0x)", {"sl_mult": 3.0, "risk_mult": 1.0}),
    ]
    
    results = []
    for label, params in configs:
        r = run_backtest(m15, m5, label, **params)
        results.append(r)
        print_results(r)
    
    # Comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<30} {'OLD (SL 1.5x)':<22} {'NEW (SL 3.0x)':<22}")
    print(f"  {'-'*74}")
    for name, key in [
        ("Net Profit ($)", "net_pnl"),
        ("Return (%)", "return_pct"),
        ("Max Drawdown (%)", "max_dd_pct"),
        ("Profit Factor", "profit_factor"),
        ("Win Rate (%)", "win_rate"),
        ("Total Trades", "trades"),
        ("Avg Win ($)", "avg_win"),
        ("Avg Loss ($)", "avg_loss"),
    ]:
        vals = [f"${r[key]:+.2f}" if key in ["net_pnl","avg_win","avg_loss"] else 
                f"{r[key]:.1f}%" if key in ["return_pct","max_dd_pct","win_rate"] else
                f"{r[key]:.2f}" if key == "profit_factor" else
                f"{r[key]:d}" if key == "trades" else
                f"{r[key]:.2f}" for r in results]
        print(f"  {name:<30} {vals[0]:<22} {vals[1]:<22}")
    
    # BUY/SELL and exit reasons
    v0 = [f"{r['buys']}/{r['sells']}" for r in results]
    v1 = [f"{r['sl_count']}/{r['tp_count']}/{r['trail_count']}" for r in results]
    print(f"  {'BUY / SELL':<30} {v0[0]:<22} {v0[1]:<22}")
    print(f"  {'SL / TP / Trail':<30} {v1[0]:<22} {v1[1]:<22}")
    
    # Recommendation
    old = results[0]
    new = results[1]
    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print(f"{'='*70}")
    
    improvements = []
    if new['max_dd_pct'] < old['max_dd_pct']:
        improvements.append(f"Drawdown reduced: {old['max_dd_pct']:.1f}% -> {new['max_dd_pct']:.1f}%")
    if new['profit_factor'] > old['profit_factor']:
        improvements.append(f"Profit Factor improved: {old['profit_factor']:.2f} -> {new['profit_factor']:.2f}")
    if new['win_rate'] > old['win_rate']:
        improvements.append(f"Win Rate improved: {old['win_rate']:.1f}% -> {new['win_rate']:.1f}%")
    if new['net_pnl'] > old['net_pnl']:
        improvements.append(f"Net Profit increased: ${old['net_pnl']:.2f} -> ${new['net_pnl']:.2f}")
    
    for imp in improvements:
        print(f"  ✅ {imp}")
    
    if len(improvements) >= 3:
        print(f"\n  ★ SL 3.0x OUTPERFORMS — Recommended config")
    elif new['net_pnl'] > old['net_pnl']:
        print(f"\n  ★ SL 3.0x performs better — Recommended")
    else:
        print(f"\n  ⚠️ Results are mixed — SL 3.0x may need adjustment on this specific period")
    
    # Compare with VPS data
    print(f"\n{'='*70}")
    print("  COMPARISON WITH YOUR VPS (16 trades, Jul 21-22)")
    print(f"{'='*70}")
    vps_pnl = sum(t.get("pnl", 0) for t in GROUND_TRUTH)
    vps_wins = sum(1 for t in GROUND_TRUTH if t.get("pnl", 0) > 0)
    print(f"  VPS Actual:  {len(GROUND_TRUTH)} trades, {vps_wins}W, P&L ${vps_pnl:+.2f}")
    print(f"  Old Bot:     {old['trades']} trades, {old['wins']}W, P&L ${old['net_pnl']:+.2f}")
    print(f"  New Bot:     {new['trades']} trades, {new['wins']}W, P&L ${new['net_pnl']:+.2f}")
    
    print(f"\n{'='*70}")
    print("  DONE.")
    print(f"{'='*70}")


# Your actual VPS trades (from bot_state_super.json you sent)
GROUND_TRUTH = [
    {"entry": 4083.28, "dir": "BUY", "lot": 0.02, "pnl": 21.08, "reason": "TP"},
    {"entry": 4114.82, "dir": "BUY", "lot": 0.01, "pnl": 2.88, "reason": "TRAIL"},
    {"entry": 4120.98, "dir": "BUY", "lot": 0.01, "pnl": -0.17, "reason": "TRAIL"},
    {"entry": 4116.39, "dir": "BUY", "lot": 0.01, "pnl": -0.19, "reason": "TRAIL"},
    {"entry": 4118.22, "dir": "BUY", "lot": 0.01, "pnl": 0.65, "reason": "TRAIL"},
    {"entry": 4116.50, "dir": "BUY", "lot": 0.01, "pnl": 0.61, "reason": "TRAIL"},
    {"entry": 4118.93, "dir": "BUY", "lot": 0.01, "pnl": -0.20, "reason": "TRAIL"},
    {"entry": 4123.06, "dir": "BUY", "lot": 0.01, "pnl": 3.53, "reason": "TRAIL"},
    {"entry": 4133.76, "dir": "BUY", "lot": 0.01, "pnl": -5.91, "reason": "SL"},
    {"entry": 4133.31, "dir": "BUY", "lot": 0.01, "pnl": 9.10, "reason": "TRAIL"},
    {"entry": 4153.65, "dir": "BUY", "lot": 0.01, "pnl": 0.98, "reason": "TRAIL"},
    {"entry": 4147.75, "dir": "BUY", "lot": 0.01, "pnl": 3.26, "reason": "TRAIL"},
    {"entry": 4149.78, "dir": "BUY", "lot": 0.01, "pnl": 7.10, "reason": "TRAIL"},
    {"entry": 4156.89, "dir": "BUY", "lot": 0.01, "pnl": 2.82, "reason": "TRAIL"},
    {"entry": 4158.54, "dir": "BUY", "lot": 0.01, "pnl": -6.84, "reason": "SL"},
    {"entry": 4142.47, "dir": "BUY", "lot": 0.01, "pnl": 0, "reason": "OPEN"},
]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()