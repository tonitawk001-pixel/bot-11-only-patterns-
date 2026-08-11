"""
FREQUENCY COMPARISON — More Trades vs Better Trades
====================================================
Tests multiple strategies + shows last 30 days in detail.

Run: python local_backtest/freq_comparison.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = r'c:\visual studio code\Ai-bot-linked-to-Meta-api-MT5-made-by-deepseek'
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

# Load H1 data
fp = os.path.join(DATA_DIR, "XAUUSD_1y_H1.csv")
df = pd.read_csv(fp)
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df.set_index('Datetime', inplace=True)
df.columns = [c.lower() for c in df.columns]
df = df[~df.index.duplicated(keep='last')].dropna()
df.sort_index(inplace=True)

# Also load M5 data for M5 analysis mode
m5_fp = os.path.join(DATA_DIR, "XAUUSD_60d_M5.csv")
m5_df = None
if os.path.exists(m5_fp):
    m5_df = pd.read_csv(m5_fp)
    m5_df['Datetime'] = pd.to_datetime(m5_df['Datetime'], utc=True)
    m5_df.set_index('Datetime', inplace=True)
    m5_df.columns = [c.lower() for c in m5_df.columns]
    m5_df = m5_df[~m5_df.index.duplicated(keep='last')].dropna()
    m5_df.sort_index(inplace=True)


def compute_adx(h, l, c, p=14):
    if len(c) < p*2: return pd.Series([np.nan]*len(c), index=c.index)
    h=h.astype(float); l=l.astype(float); c=c.astype(float)
    tr=pd.concat([(h-l).abs(),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up=h-h.shift(); down=l.shift()-l
    pdm=np.where((up>down)&(up>0),up,0.0)
    ndm=np.where((down>up)&(down>0),down,0.0)
    atr=tr.ewm(span=p,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=c.index).ewm(span=p,adjust=False).mean()/atr
    ndi=100*pd.Series(ndm,index=c.index).ewm(span=p,adjust=False).mean()/atr
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False).mean()


def run_test(label, ms, tp_mult, sl_mult, trail_mult, max_pos, risk_tiers, df_ohlcv, df_m5=None):
    """Run backtest and return results + daily breakdown."""
    TP_TREND = max(tp_mult * 1.4, 5.0)
    ADX_TH = 20; BE = 2.0; DAILY_LOSS = 0.03

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = max_pos

    balance = STARTING_BALANCE
    positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0

    daily_data = {}  # date -> [list of trade dicts]

    for i in range(200, len(df_ohlcv)):
        ct = df_ohlcv.index[i]
        price = float(df_ohlcv["close"].iloc[i])

        if not (8 <= ct.hour < 22): continue
        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue
        if ct.weekday() == 4 and ct.hour >= 18: continue
        if last_date is None: last_date = ct.date()
        if ct.date() != last_date: daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0
        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS: continue
        if trade_count_today >= 50: continue

        # Build windows
        m5w = df_ohlcv.iloc[max(0, i-200):i+1].copy()
        m15w = df_ohlcv.iloc[max(0, i-500):i+1].copy()
        if len(m5w) < 50 or len(m15w) < 50: continue

        ind5 = compute_all_indicators(m5w)
        ind15 = compute_all_indicators(m15w)
        if not ind5 or not ind15: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue
        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5: continue

        # Update positions
        surv = []
        for p in positions:
            e, d, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
            pv = lot * 100
            if not p.get("be", False) and p.get("be_target"):
                if d == "BUY" and price >= p["be_target"]: p["be"] = True; p["sl"] = e
                elif d == "SELL" and price <= p["be_target"]: p["be"] = True; p["sl"] = e
            if p.get("be"):
                ns = price - atr_val * trail_mult if d == "BUY" else price + atr_val * trail_mult
                if d == "BUY" and ns > sl + 0.5: p["sl"] = round(ns, 2)
                elif d == "SELL" and ns < sl - 0.5: p["sl"] = round(ns, 2)
            sl, tp = p["sl"], p["tp"]
            hit, pnl, reason = False, 0.0, ""
            if d == "BUY":
                if tp and price >= tp: pnl = (tp - e) * pv; reason = "TP"; hit = True
                elif sl and price <= sl: pnl = (sl - e) * pv; reason = "TRAIL" if sl > e else "SL"; hit = True
            else:
                if tp and price <= tp: pnl = (e - tp) * pv; reason = "TP"; hit = True
                elif sl and price >= sl: pnl = (e - sl) * pv; reason = "TRAIL" if sl < e else "SL"; hit = True
            if hit:
                pnl -= 0.5 * lot * 100
                daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = price
                closed.append(p)
                # Add to daily data
                day_key = ct.date()
                if day_key not in daily_data: daily_data[day_key] = []
                daily_data[day_key].append(p)
                if reason == "SL":
                    cons_losses += 1
                    if cons_losses >= 3: halt_until = ct + timedelta(hours=6); cons_losses = 0
                else: cons_losses = 0
            else: surv.append(p)
        positions = surv
        if cons_losses >= 3 and halt_until is None: halt_until = ct + timedelta(hours=6); continue
        if len(positions) >= max_pos: continue

        try:
            adx_s = compute_adx(m5w["high"], m5w["low"], m5w["close"])
            adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val = 0
        tp_use = TP_TREND if adx_val >= ADX_TH else tp_mult

        empty = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
        try:
            result = strategy.analyze(m1_indicators=empty, m5_indicators=ind5, m15_indicators=ind15,
                m1_ohlcv=m5w.tail(20), m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except: continue

        direction = result.get("direction", "NONE"); score = result.get("setup_score", 0)
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

        risk_pct = 0.02
        for lo, hi, rp in risk_tiers:
            if lo < balance <= hi: risk_pct = rp / 100.0; break
        risk_amt = balance * risk_pct
        risk_per_lot = sd * 100
        raw_lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        lot = max(0.01, min(round(raw_lot / 0.01) * 0.01, 10.0))
        trade_count_today += 1

        pos = {"entry": price, "sl": sl, "tp": tp, "lot": lot, "dir": direction,
               "open_time": ct, "score": score,
               "be_target": price + (atr_val * BE if direction == "BUY" else -atr_val * BE),
               "be": False}
        positions.append(pos); last_entry = ct

    # Calculate results
    total_pnl = sum(t["pnl"] for t in closed) if closed else 0
    wins = sum(1 for t in closed if t["pnl"] > 0) if closed else 0
    peak = STARTING_BALANCE; maxdd = 0; eq = [STARTING_BALANCE]
    for t in closed:
        eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else (float('inf') if win_pnls else 0)
    wr = wins / len(closed) * 100 if closed else 0

    return {
        "label": label,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "max_dd": maxdd,
        "trades": len(closed),
        "win_rate": wr,
        "pf": pf,
        "final_balance": STARTING_BALANCE + total_pnl,
        "daily_data": daily_data,
        "trading_days": len(daily_data),
    }


def print_30_days(daily_data, label):
    """Print last 30 days of trades."""
    sorted_days = sorted(daily_data.keys())
    last_30 = sorted_days[-30:] if len(sorted_days) >= 30 else sorted_days

    print(f"\n{'='*100}")
    print(f"  LAST 30 TRADING DAYS — {label}")
    print(f"{'='*100}")
    print(f"  {'Date':<12} {'Trades':<7} {'P&L':<12} {'Wins':<6} {'Losses':<7} {'Avg P&L':<10}")
    print(f"  {'-'*55}")

    total_pnl_30 = 0
    for day in last_30:
        trades = daily_data[day]
        day_pnl = sum(t["pnl"] for t in trades)
        day_wins = sum(1 for t in trades if t["pnl"] > 0)
        day_losses = len(trades) - day_wins
        avg_pnl = day_pnl / len(trades) if trades else 0
        total_pnl_30 += day_pnl
        print(f"  {str(day):<12} {len(trades):<7} ${day_pnl:<+8.2f} {day_wins:<6} {day_losses:<7} ${avg_pnl:<+7.2f}")

    print(f"  {'-'*55}")
    win_days = sum(1 for d in last_30 if sum(t["pnl"] for t in daily_data[d]) > 0)
    loss_days = len(last_30) - win_days
    print(f"  30-DAY TOTAL: ${total_pnl_30:+.2f} | Win days: {win_days} | Loss days: {loss_days}")


def main():
    print("=" * 80)
    print("  FREQUENCY COMPARISON — 1 Year H1 Data")
    print("=" * 80)
    print(f"  Data: {len(df)} candles")

    # Define tests
    tests = [
        {
            "label": "A: AGGRESSIVE (Score=35)",
            "ms": 35, "tp": 6.0, "sl": 1.5, "trail": 0.5, "max_pos": 2,
            "risk_tiers": [(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)],
        },
        {
            "label": "B: SUPER AGGRESSIVE (Score=30)",
            "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.5, "max_pos": 2,
            "risk_tiers": [(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)],
        },
        {
            "label": "C: LOW SCORE + TIGHT TRAIL (Score=30, Trail=0.3)",
            "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "max_pos": 2,
            "risk_tiers": [(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)],
        },
    ]

    results = []
    for t in tests:
        print(f"\n{'='*60}")
        print(f"  Testing: {t['label']}")
        print(f"{'='*60}")
        r = run_test(t["label"], t["ms"], t["tp"], t["sl"], t["trail"], t["max_pos"], t["risk_tiers"], df)
        ret = r["return_pct"]
        print(f"  Net P&L:    ${r['net_pnl']:+.2f} ({ret:+.1f}%)")
        print(f"  Max DD:     {r['max_dd']:.1f}%")
        print(f"  Trades:     {r['trades']} ({r['trading_days']} trading days)")
        print(f"  Avg/day:    {r['trades']/r['trading_days']:.2f}" if r['trading_days'] > 0 else "  Avg/day: N/A")
        print(f"  Win Rate:   {r['win_rate']:.1f}%")
        print(f"  PF:         {r['pf']:.2f}")
        print(f"  Final Bal:  ${r['final_balance']:.2f}")
        results.append(r)

    # Print 30 days for each
    for r in results:
        print_30_days(r["daily_data"], r["label"])

    # Comparison table
    print(f"\n{'='*110}")
    print(f"  🏆 FINAL COMPARISON")
    print(f"{'='*110}")
    print(f"  {'Version':<40} {'Net P&L':<12} {'Return%':<14} {'DD%':<8} {'Trades':<8} {'Days':<6} {'PF':<8}")
    print(f"  {'-'*92}")

    best = None
    for r in sorted(results, key=lambda x: x["net_pnl"], reverse=True):
        print(f"  {r['label']:<40} ${r['net_pnl']:<+8.2f} {r['return_pct']:<+9.1f}%  {r['max_dd']:<6.1f}% {r['trades']:<8} {r['trading_days']:<6} {r['pf']:<8.2f}")
        if best is None or r["net_pnl"] > best["net_pnl"]:
            best = r

    if best:
        print(f"\n  🏆 WINNER: {best['label']}")
        print(f"     Return: {best['return_pct']:+.1f}% | DD: {best['max_dd']:.1f}% | Trades: {best['trades']}")

        # Extract params
        for t in tests:
            if t["label"] == best["label"]:
                print(f"\n  ✅ UPDATE main_super.py WITH:")
                print(f"  MIN_SCORE = {t['ms']}")
                print(f"  TP_ATR_MULT = 4.0")
                print(f"  TP_ATR_MULT_TREND = 6.0")
                print(f"  SL_ATR_MULT = {t['sl']}")
                print(f"  TRAIL_ATR_MULT = {t['trail']}")
                print(f"  MAX_POSITIONS = {t['max_pos']}")
                print(f"  RISK_TIERS = {t['risk_tiers']}")
                break

    print(f"\n{'='*80}")
    print(f"  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()