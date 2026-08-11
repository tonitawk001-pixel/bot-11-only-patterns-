"""
DATA DOWNLOADER + BACKTEST — Standalone research tool
======================================================
Downloads gold data from Yahoo Finance (daily) and runs the bot's strategy
logic on it. Does NOT modify any bot code.

For M15/M5 accuracy: uses existing local data + resamples M1 data.
For fresh data (full year): uses Yahoo Finance daily bars.

Run:  python local_backtest/data_downloader.py

Requirements: pip install yfinance
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

# ===== CONFIG =====
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTING_BALANCE = 500.00


def download_yahoo_daily():
    """Download 1 year of daily gold data from Yahoo Finance."""
    try:
        import yfinance as yf
        print("\n  [Download] Downloading 1 year daily gold data from Yahoo Finance...")
        df = yf.download("GC=F", period="1y", interval="1d", progress=False)
        if df.empty:
            df = yf.download("GLD", period="1y", interval="1d", progress=False)
        if df.empty:
            print("  [Download] ERROR: No data from Yahoo Finance")
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        df = df.dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        print(f"  [Download] {len(df)} daily candles: {df.index[0]} to {df.index[-1]}")
        return df
    except ImportError:
        print("  [Download] yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"  [Download] Error: {e}")
        return None


def load_local_data():
    """Load existing local M15, M5, M1 data."""
    m15_file = os.path.join(DATA_DIR, "XAUUSD_60d_M15.csv")
    m5_file = os.path.join(DATA_DIR, "XAUUSD_60d_M5.csv")
    m1_file = os.path.join(DATA_DIR, "XAUUSD_7d_M1.csv")
    
    data = {}
    
    for name, fp in [("m15", m15_file), ("m5", m5_file), ("m1", m1_file)]:
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df.set_index('Datetime', inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df = df[~df.index.duplicated(keep='last')].dropna()
            df.sort_index(inplace=True)
            data[name] = df
            print(f"  [Local] {name}: {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    
    return data


def resample_m1_to_m5_m15(m1_df):
    """Resample 1-minute data to M5 and M15."""
    if m1_df is None or m1_df.empty:
        return None, None
    
    print(f"  [Resample] Creating M5 from M1 data...")
    m5 = m1_df.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
        'spread': 'last', 'real_volume': 'sum'
    }).dropna()
    print(f"  [Resample] M5: {len(m5)} candles ({m5.index[0]} to {m5.index[-1]})")
    
    print(f"  [Resample] Creating M15 from M1 data...")
    m15 = m1_df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
        'spread': 'last', 'real_volume': 'sum'
    }).dropna()
    print(f"  [Resample] M15: {len(m15)} candles ({m15.index[0]} to {m15.index[-1]})")
    
    return m5, m15


def combine_datasets(local_m15, local_m5, resampled_m15, resampled_m5):
    """Combine local data with resampled M1 data (resampled takes priority for overlap)."""
    combined_m15 = local_m15.copy() if local_m15 is not None else pd.DataFrame()
    combined_m5 = local_m5.copy() if local_m5 is not None else pd.DataFrame()
    
    if resampled_m15 is not None and not resampled_m15.empty:
        # Remove overlapping older data
        overlap_start = resampled_m15.index[0]
        combined_m15 = combined_m15[combined_m15.index < overlap_start]
        # Append resampled data
        combined_m15 = pd.concat([combined_m15, resampled_m15])
        combined_m15 = combined_m15[~combined_m15.index.duplicated(keep='last')]
        combined_m15.sort_index(inplace=True)
        print(f"  [Combine] M15 now: {len(combined_m15)} candles ({combined_m15.index[0]} to {combined_m15.index[-1]})")
    
    if resampled_m5 is not None and not resampled_m5.empty:
        overlap_start = resampled_m5.index[0]
        combined_m5 = combined_m5[combined_m5.index < overlap_start]
        combined_m5 = pd.concat([combined_m5, resampled_m5])
        combined_m5 = combined_m5[~combined_m5.index.duplicated(keep='last')]
        combined_m5.sort_index(inplace=True)
        print(f"  [Combine] M5 now: {len(combined_m5)} candles ({combined_m5.index[0]} to {combined_m5.index[-1]})")
    
    return combined_m15, combined_m5


def run_quick_backtest(m15_df, m5_df, label, sl_mult=3.0, risk_mult=1.0, min_score=30):
    """Simplified backtest running bot strategy logic."""
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
                if reason == "SL": cons_losses += 1
                else: cons_losses = 0
            else: surviving.append(p)
        positions = surviving
        
        if len(positions) >= 3: continue
        if last_entry and (ct - last_entry).total_seconds() / 60 < 0: continue
        
        try:
            adx_series = compute_adx_simple(m5w["high"], m5w["low"], m5w["close"])
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
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
        if direction == "NONE" or score < min_score: continue
        
        try:
            if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40): continue
            if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60): continue
        except: pass
        
        sd = atr_val * sl_mult
        td = atr_val * tp_mult
        if direction == "BUY": sl = round(price - sd, 2); tp = round(price + td, 2)
        else: sl = round(price + sd, 2); tp = round(price - td, 2)
        
        risk_amt = balance * 0.02 * risk_mult
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
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []
    
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
    for t in closed:
        dir_counts[t["dir"]] = dir_counts.get(t["dir"], 0) + 1
        reason_counts[t.get("reason", "?")] = reason_counts.get(t.get("reason", "?"), 0) + 1
    
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


def save_daily_to_csv(daily_df):
    """Save daily data to CSV for reference."""
    out = os.path.join(OUTPUT_DIR, "gold_daily_1year.csv")
    daily_df.to_csv(out)
    print(f"  [Save] Daily data saved to: {out}")
    return out


def main():
    print("=" * 70)
    print("  DATA DOWNLOADER + BACKTEST — RESEARCH TOOL")
    print("  Does NOT modify any bot code")
    print("=" * 70)
    
    # Step 1: Download Yahoo data
    print(f"\n{'='*70}")
    print("  STEP 1: Download fresh data from Yahoo Finance")
    print(f"{'='*70}")
    daily_df = download_yahoo_daily()
    if daily_df is not None:
        save_daily_to_csv(daily_df)
        
        # Run daily backtest (approximate - not as accurate as M15)
        print(f"\n  NOTE: Daily data gives approximate results only.")
        print(f"  For accurate M15+M5 backtest, use the local data below.")
    
    # Step 2: Load local data
    print(f"\n{'='*70}")
    print("  STEP 2: Load local M15/M5/M1 data")
    print(f"{'='*70}")
    local = load_local_data()
    
    # Step 3: Resample M1 to extend M5/M15
    print(f"\n{'='*70}")
    print("  STEP 3: Resample M1 → M5 + M15 (extends data to recent)")
    print(f"{'='*70}")
    resampled_m5 = None
    resampled_m15 = None
    if "m1" in local:
        resampled_m5, resampled_m15 = resample_m1_to_m5_m15(local["m1"])
    
    # Step 4: Combine datasets
    print(f"\n{'='*70}")
    print("  STEP 4: Combine local + resampled data")
    print(f"{'='*70}")
    combined_m15, combined_m5 = combine_datasets(
        local.get("m15"), local.get("m5"),
        resampled_m15, resampled_m5
    )
    
    # Step 5: Run backtest comparisons
    print(f"\n{'='*70}")
    print("  STEP 5: Run backtest comparisons")
    print(f"{'='*70}")
    
    if combined_m15 is None or combined_m5 is None or len(combined_m15) < 500:
        print("  Not enough data for M15+M5 backtest.")
        print("  Using local data only (if available)...")
        combined_m15 = local.get("m15")
        combined_m5 = local.get("m5")
    
    if combined_m15 is not None and combined_m5 is not None and len(combined_m15) >= 500:
        configs = [
            ("OLD BOT (SL 1.5x, Risk 100%)", {"sl_mult": 1.5, "risk_mult": 1.0, "min_score": 30}),
            ("NEW BOT (SL 3.0x, Risk 100%)", {"sl_mult": 3.0, "risk_mult": 1.0, "min_score": 30}),
            ("SAFE BOT (SL 2.0x, Risk 50%)", {"sl_mult": 2.0, "risk_mult": 0.5, "min_score": 30}),
        ]
        
        results = []
        for label, params in configs:
            r = run_quick_backtest(combined_m15, combined_m5, label, **params)
            results.append(r)
            print_results(r)
        
        # Comparison table
        print(f"\n{'='*70}")
        print("  COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"\n  {'Metric':<30} {'OLD (SL1.5)':<18} {'NEW (SL3.0)':<18} {'SAFE (SL2+R50)':<18}")
        print(f"  {'-'*84}")
        metrics = [
            ("Net Profit ($)", "net_pnl", "${:+.2f}"),
            ("Return (%)", "return_pct", "{:+.1f}%"),
            ("Max Drawdown (%)", "max_dd_pct", "{:.1f}%"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
            ("Win Rate (%)", "win_rate", "{:.1f}%"),
            ("Total Trades", "trades", "{:d}"),
            ("BUY / SELL", None, None),  # Special case
            ("SL/TP/Trail", None, None),  # Special case
        ]
        for name, key, fmt in metrics:
            vals = []
            if key:
                for r in results:
                    if isinstance(r[key], float):
                        vals.append(fmt.format(r[key]))
                    else:
                        vals.append(str(r[key]))
                print(f"  {name:<30} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")
            else:
                if "BUY" in name:
                    for r in results:
                        vals.append(f"{r['buys']}/{r['sells']}")
                else:
                    for r in results:
                        vals.append(f"{r['sl_count']}/{r['tp_count']}/{r['trail_count']}")
                print(f"  {name:<30} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")
        
        # Which is best?
        baseline = results[0]
        best = max(results[1:], key=lambda r: r['profit_factor'] * abs(r['return_pct']) / max(r['max_dd_pct'], 0.1))
        print(f"\n  ★ BEST CONFIG: {best['label']}")
        print(f"    Return: {best['return_pct']:+.1f}% | DD: {best['max_dd_pct']:.1f}% | PF: {best['profit_factor']:.2f}")
        
    else:
        print("  ERROR: Not enough M15 data for analysis.")
    
    # Step 6: Also show daily approximate backtest
    if daily_df is not None:
        print(f"\n{'='*70}")
        print("  STEP 6: Daily data approximate analysis")
        print(f"  (Not as accurate as M15, but covers full year)")
        print(f"{'='*70}")
        
        # Simple trend analysis on daily data
        daily_close = daily_df["close"]
        returns = daily_close.pct_change().dropna()
        
        max_drawdown_daily = 0
        peak = daily_close.iloc[0]
        for price in daily_close:
            peak = max(peak, price)
            dd = (peak - price) / peak * 100
            max_drawdown_daily = max(max_drawdown_daily, dd)
        
        print(f"\n  Gold (GC=F) Daily Statistics (1 year):")
        print(f"  Period:         {daily_df.index[0]} to {daily_df.index[-1]}")
        print(f"  Start Price:    ${daily_close.iloc[0]:.2f}")
        print(f"  End Price:      ${daily_close.iloc[-1]:.2f}")
        print(f"  Price Change:   {(daily_close.iloc[-1]/daily_close.iloc[0]-1)*100:+.1f}%")
        print(f"  Max Drawdown:   {max_drawdown_daily:.1f}%")
        print(f"  Avg Daily Vol:  {returns.std()*100:.2f}%")
        print(f"  Trading Days:   {len(daily_df)}")
    
    print(f"\n{'='*70}")
    print("  DONE. All data saved to local_backtest/ folder")
    print("  Bot code was NOT modified.")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()