"""
DEEP ANALYSIS — Can we increase trades without increasing DD?
==============================================================
Tests 8 variations against Version C (baseline: +$201K, DD 36.6%).

Run: python local_backtest/deep_analysis.py
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

# Load H1 data for analysis
fp = os.path.join(DATA_DIR, "XAUUSD_1y_H1.csv")
df = pd.read_csv(fp)
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df.set_index('Datetime', inplace=True)
df.columns = [c.lower() for c in df.columns]
df = df[~df.index.duplicated(keep='last')].dropna()
df.sort_index(inplace=True)


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


def run_test(label, ms, tp_mult, sl_mult, trail_mult, max_pos, risk_tiers,
             session_8_22=True, use_rsi_filter=True, use_ema200=True, df_ohlcv=None):
    """Run backtest with configurable filters."""
    TP_TREND = max(tp_mult * 1.4, 5.0)
    ADX_TH = 20; BE = 2.0; DAILY_LOSS = 0.03

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = max_pos

    balance = STARTING_BALANCE
    positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0; blocked_reasons = {}

    for i in range(200, len(df_ohlcv)):
        ct = df_ohlcv.index[i]
        price = float(df_ohlcv["close"].iloc[i])

        if session_8_22:
            if not (8 <= ct.hour < 22):
                # Still update positions outside session
                if positions:
                    m5w = df_ohlcv.iloc[max(0, i-200):i+1].copy()
                    ind5 = compute_all_indicators(m5w)
                    atr_v = 0
                    if ind5 and ind5.get("atr") is not None and len(ind5["atr"]) > 0:
                        atr_v = float(ind5["atr"].iloc[-1])
                    surv = []
                    for p in positions:
                        e, d, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
                        pv = lot * 100
                        if not p.get("be", False) and p.get("be_target"):
                            if d == "BUY" and price >= p["be_target"]: p["be"] = True; p["sl"] = e
                            elif d == "SELL" and price <= p["be_target"]: p["be"] = True; p["sl"] = e
                        if p.get("be"):
                            ns = price - atr_v * trail_mult if d == "BUY" else price + atr_v * trail_mult
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
                            pnl -= 0.5 * lot * 100; daily_pnl += pnl; balance += pnl
                            p["pnl"] = pnl; p["reason"] = reason; closed.append(p)
                            if reason == "SL":
                                cons_losses += 1
                                if cons_losses >= 3: halt_until = ct + timedelta(hours=6); cons_losses = 0
                            else: cons_losses = 0
                        else: surv.append(p)
                    positions = surv
                continue

        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue
        if ct.weekday() == 4 and ct.hour >= 18 and session_8_22: continue

        if last_date is None: last_date = ct.date()
        if ct.date() != last_date: daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0
        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS: continue
        if trade_count_today >= 50: continue

        m5w = df_ohlcv.iloc[max(0, i-200):i+1].copy()
        m15w = df_ohlcv.iloc[max(0, i-500):i+1].copy()
        if len(m5w) < 50 or len(m15w) < 50: continue

        ind5 = compute_all_indicators(m5w); ind15 = compute_all_indicators(m15w)
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
                pnl -= 0.5 * lot * 100; daily_pnl += pnl; balance += pnl
                p["pnl"] = pnl; p["reason"] = reason; closed.append(p)
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
        if direction == "NONE" or score < ms:
            blocked_reasons["low_score"] = blocked_reasons.get("low_score", 0) + 1
            continue

        if use_rsi_filter:
            try:
                if direction == "BUY" and not (ind5["rsi"].iloc[-1] > 40 and ind15["rsi"].iloc[-1] > 40):
                    blocked_reasons["rsi"] = blocked_reasons.get("rsi", 0) + 1; continue
                if direction == "SELL" and not (ind5["rsi"].iloc[-1] < 60 and ind15["rsi"].iloc[-1] < 60):
                    blocked_reasons["rsi"] = blocked_reasons.get("rsi", 0) + 1; continue
            except: pass

        if use_ema200:
            closes = m15w["close"].values
            if len(closes) >= 200:
                ema200 = pd.Series(closes).ewm(200, adjust=False).mean().values
                if len(ema200) >= 10:
                    rising = ema200[-1] > ema200[-10]
                    if direction == "BUY" and not rising:
                        blocked_reasons["ema200"] = blocked_reasons.get("ema200", 0) + 1; continue
                    if direction == "SELL" and rising:
                        blocked_reasons["ema200"] = blocked_reasons.get("ema200", 0) + 1; continue

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
               "be_target": price + (atr_val * BE if direction == "BUY" else -atr_val * BE), "be": False}
        positions.append(pos); last_entry = ct

    if not closed:
        return {"net_pnl": 0, "max_dd": 0, "trades": 0, "win_rate": 0, "pf": 0, "trading_days": 0}

    total_pnl = sum(t["pnl"] for t in closed)
    wins = sum(1 for t in closed if t["pnl"] > 0)
    peak = STARTING_BALANCE; maxdd = 0; eq = [STARTING_BALANCE]
    for t in closed: eq.append(eq[-1] + t["pnl"])
    for e in eq:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        maxdd = max(maxdd, dd)
    win_pnls = [t["pnl"] for t in closed if t["pnl"] > 0] if closed else []
    loss_pnls = [t["pnl"] for t in closed if t["pnl"] <= 0] if closed else []
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else (float('inf') if win_pnls else 0)
    wr = wins / len(closed) * 100 if closed else 0
    trading_days = len(set(t["open_time"].date() for t in closed if hasattr(t["open_time"], "date")))

    return {
        "label": label, "net_pnl": total_pnl,
        "return_pct": (total_pnl / STARTING_BALANCE) * 100,
        "max_dd": maxdd, "trades": len(closed), "win_rate": wr, "pf": pf,
        "trading_days": trading_days, "blocked_reasons": blocked_reasons,
    }


def main():
    print("=" * 100)
    print("  DEEP ANALYSIS — Can we increase trades without increasing DD?")
    print("=" * 100)

    base_tiers = [(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)]

    tests = [
        {"label": "C (baseline)", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 2,
         "session": True, "rsi": True, "ema": True},
        {"label": "D: MaxPos=3", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 3,
         "session": True, "rsi": True, "ema": True},
        {"label": "E: No RSI filter", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 2,
         "session": True, "rsi": False, "ema": True},
        {"label": "F: 24h trading", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 2,
         "session": False, "rsi": True, "ema": True},
        {"label": "G: MaxPos=3 + No RSI", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 3,
         "session": True, "rsi": False, "ema": True},
        {"label": "H: Score=25", "ms": 25, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 2,
         "session": True, "rsi": True, "ema": True},
        {"label": "I: No EMA200", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 2,
         "session": True, "rsi": True, "ema": False},
        {"label": "J: MaxPos=3 + 24h", "ms": 30, "tp": 6.0, "sl": 1.5, "trail": 0.3, "mp": 3,
         "session": False, "rsi": True, "ema": True},
    ]

    results = []
    for t in tests:
        print(f"\n  Testing: {t['label']}...", end="", flush=True)
        r = run_test(t["label"], t["ms"], t["tp"], t["sl"], t["trail"], t["mp"], base_tiers,
                     t["session"], t["rsi"], t["ema"], df)
        print(f" Done")
        results.append(r)

    # Find baseline (C)
    baseline = [r for r in results if "baseline" in r["label"]][0]
    b_ratio = baseline["net_pnl"] / baseline["max_dd"] if baseline["max_dd"] > 0 else 0

    # Results table
    print(f"\n{'='*130}")
    print(f"  📊 RESULTS — Sorted by Net Profit (within ≤baseline DD if possible)")
    print(f"{'='*130}")
    print(f"  {'Version':<28} {'Net P&L':<12} {'Return%':<12} {'DD%':<8} {'Trades':<8} {'Days':<6} {'Avg/D':<7} {'PF':<7} {'Ratio':<8} {'Status':<15}")
    print(f"  {'-'*105}")

    # Sort by a score: if DD <= baseline DD, sort by profit; else penalize
    def score_func(r):
        dd_penalty = 1.0 if r["max_dd"] <= baseline["max_dd"] * 1.05 else (baseline["max_dd"] / max(r["max_dd"], 0.1))
        return r["net_pnl"] * dd_penalty

    sorted_results = sorted(results, key=score_func, reverse=True)

    best = None
    best_score_val = -999999

    for r in sorted_results:
        ret = r["return_pct"]
        avg_day = r["trades"] / r["trading_days"] if r["trading_days"] > 0 else 0
        ratio = r["net_pnl"] / r["max_dd"] if r["max_dd"] > 0 else 0
        dd_ok = r["max_dd"] <= baseline["max_dd"] * 1.05

        status = ""
        if r == baseline:
            status = "🏆 BASELINE"
        elif dd_ok and r["net_pnl"] > baseline["net_pnl"]:
            status = "⭐ BEATS C!"
        elif dd_ok:
            status = "✅ DD OK"
        else:
            status = f"⚠️ DD +{r['max_dd'] - baseline['max_dd']:.1f}%"

        print(f"  {r['label']:<28} ${r['net_pnl']:<+8.2f} {ret:<+8.1f}%  {r['max_dd']:<6.1f}% {r['trades']:<8} {r['trading_days']:<6} {avg_day:<6.1f}  {r['pf']:<7.2f} {ratio:<7.1f} {status:<15}")

        s = score_func(r)
        if s > best_score_val:
            best_score_val = s
            best = r

    # Recommendation
    print(f"\n{'='*80}")
    if best is not None and best != baseline:
        print(f"  🏆 BEST IMPROVEMENT: {best['label']}")
        print(f"     Net P&L: ${best['net_pnl']:+.2f} vs baseline ${baseline['net_pnl']:+.2f}")
        print(f"     DD: {best['max_dd']:.1f}% vs baseline {baseline['max_dd']:.1f}%")
        print(f"     Trades: {best['trades']} vs baseline {best['trades']}")
    else:
        print(f"  No version beats baseline C. Keeping current settings.")

    print(f"\n  ✅ Final recommendation:")
    if best is not None and best != baseline and best["max_dd"] <= baseline["max_dd"] * 1.05:
        print(f"  UPDATE main_super.py with: {best['label']}")
        for t in tests:
            if t["label"] == best["label"]:
                print(f"    MIN_SCORE = {t['ms']}")
                print(f"    MAX_POSITIONS = {t['mp']}")
                print(f"    session_8_22 = {t['session']}")
                print(f"    use_rsi_filter = {t['rsi']}")
                print(f"    use_ema200 = {t['ema']}")
    else:
        print(f"  Keep current settings (Version C). No version improves trades without increasing DD.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()