"""Compare: Current config vs BE-at-$40 config — 2 months"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'trading_bot_mt5'))

import pandas as pd, numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
import candle_patterns as cp

SYMBOL = "XAUUSD"
MIN_SCORE_BUY = 35; MIN_SCORE_SELL = 35
MAX_POSITIONS = 1; FIXED_RISK = 0.05
TP_ATR_MULT = 5.0; SL_ATR_MULT = 2.5
HIGH_SCORE_THRESHOLD = 70; HIGH_SCORE_RISK = 0.10
ATR_VOL_THRESHOLD = 4.0; SESSION_COOLDOWN_MIN = 10
STARTING_BALANCE = 500.00

print("Downloading XAUUSD data...")
mt5.initialize()
end = datetime.now(); start = end - timedelta(days=70)
m15_r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
m5_r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()

m15 = pd.DataFrame(m15_r); m15['time'] = pd.to_datetime(m15['time'], unit='s'); m15.set_index('time', inplace=True)
m5 = pd.DataFrame(m5_r); m5['time'] = pd.to_datetime(m5['time'], unit='s'); m5.set_index('time', inplace=True)
cutoff = m15.index.max() - timedelta(days=60)
m15 = m15[m15.index >= cutoff]; m5 = m5[m5.index >= m15.index.min()]
print(f"M15: {len(m15)} | {m15.index.min()} -> {m15.index.max()}")

def h4_trend(w):
    if len(w) < 34: return "NEUTRAL"
    e = w['close'].ewm(span=80, adjust=False).mean().iloc[-1]
    return "BULLISH" if w['close'].iloc[-1] > e else "BEARISH"

def blocked(d, s, pos, i15, i5, ht, sr, curr):
    if len(pos) >= MAX_POSITIONS: return True
    if s < MIN_SCORE_BUY: return True
    try:
        if (d == "BUY" and float(i5["rsi"].iloc[-1]) < 20) or (d == "SELL" and float(i5["rsi"].iloc[-1]) > 80): return True
    except: pass
    if pos and d in set(p["dir"] for p in pos): return True
    at = i15.get("atr")
    if at is not None and len(at) >= 20:
        ca, ma = at.iloc[-1], at.rolling(20).mean().iloc[-1]
        if not np.isnan(ca) and ca > ma * ATR_VOL_THRESHOLD: return True
    if sr:
        if d == "BUY" and (sr['resistance'] - curr) < 2.5: return True
        if d == "SELL" and (curr - sr['support']) < 2.5: return True
    if pos and d not in set(p["dir"] for p in pos) and len(pos) > 0: return True
    if d != ("BUY" if ht == "BULLISH" else "SELL"):
        rv = i15['rsi'].iloc[-1]
        if not (rv < 25 or rv > 75): return True
    try:
        bb = i15.get('bb')
        if bb is not None:
            if d == "BUY" and curr < bb['lower'].iloc[-1]: return True
            if d == "SELL" and curr > bb['upper'].iloc[-1]: return True
    except: pass
    return False

def run_sim(label, be_atr, be_buf, trail_atr, be_profit_usd=None, partial_trigger=None, partial_sl=None):
    balance = STARTING_BALANCE; peak = STARTING_BALANCE
    positions = []; trades_list = []; daily_pnl = 0.0
    last_date = ""
    strategy = GoldScalpingStrategy()
    bcount = scount = blcount = tcount = 0

    for idx in range(100, len(m15)):
        bar = m15.iloc[idx]; bt = bar.name; curr = bar['close']; h = bt.hour
        date_str = bt.strftime("%Y-%m-%d")
        if date_str != last_date:
            daily_pnl = 0.0; last_date = date_str; strategy.reset_daily()
        if balance > peak: peak = balance

        # Position management
        if positions:
            tc = []
            for i, p in enumerate(positions):
                hp, lp = (bar['high'], bar['low']) if p["dir"] == "BUY" else (bar['low'], bar['high'])
                sd = abs(p["entry"] - p["sl"])
                if (p["dir"] == "BUY" and hp >= p["tp"]) or (p["dir"] == "SELL" and lp <= p["tp"]):
                    pnl = (p["tp"] - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - p["tp"]) * p["lots"] * 100
                    p["pnl"] = round(pnl, 2); p["exit"] = "tp"
                    trades_list.append(p); tc.append(i)
                    balance += pnl; daily_pnl += pnl
                elif (p["dir"] == "BUY" and lp <= p["sl"]) or (p["dir"] == "SELL" and hp >= p["sl"]):
                    pnl = (p["sl"] - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - p["sl"]) * p["lots"] * 100
                    p["pnl"] = round(pnl, 2); p["exit"] = "sl"
                    trades_list.append(p); tc.append(i)
                    balance += pnl; daily_pnl += pnl
                elif p["sl"] and p["sl"] > 0:
                    # BE trigger (ATR-based)
                    bt_trigger = p["entry"] + (sd * be_atr / SL_ATR_MULT) if p["dir"] == "BUY" else p["entry"] - (sd * be_atr / SL_ATR_MULT)
                    if (p["dir"] == "BUY" and hp >= bt_trigger) or (p["dir"] == "SELL" and lp <= bt_trigger):
                        ns = p["entry"] + (be_buf * 0.01) if p["dir"] == "BUY" else p["entry"] - (be_buf * 0.01)
                        if (p["dir"] == "BUY" and ns > p["sl"]) or (p["dir"] == "SELL" and ns < p["sl"]): p["sl"] = ns
                    # BE-at-profit trigger (if enabled) — moves SL to entry
                    if be_profit_usd:
                        profit_pts = be_profit_usd / (p["lots"] * 100)
                        be_price = p["entry"] + profit_pts if p["dir"] == "BUY" else p["entry"] - profit_pts
                        if (p["dir"] == "BUY" and hp >= be_price) or (p["dir"] == "SELL" and lp <= be_price):
                            ns = p["entry"] + (be_buf * 0.01) if p["dir"] == "BUY" else p["entry"] - (be_buf * 0.01)
                            if (p["dir"] == "BUY" and ns > p["sl"]) or (p["dir"] == "SELL" and ns < p["sl"]): p["sl"] = ns
                    # Partial profit lock (if enabled) — SL to +$X when profit reaches +$Y
                    if partial_trigger and partial_sl:
                        trigger_pts = partial_trigger / (p["lots"] * 100)
                        sl_pts = partial_sl / (p["lots"] * 100)
                        trigger_price = p["entry"] + trigger_pts if p["dir"] == "BUY" else p["entry"] - trigger_pts
                        if (p["dir"] == "BUY" and hp >= trigger_price) or (p["dir"] == "SELL" and lp <= trigger_price):
                            ns = p["entry"] + sl_pts if p["dir"] == "BUY" else p["entry"] - sl_pts
                            if (p["dir"] == "BUY" and ns > p["sl"]) or (p["dir"] == "SELL" and ns < p["sl"]): p["sl"] = ns
                    # Trail
                    if trail_atr > 0:
                        if p["dir"] == "BUY":
                            profit = hp - p["entry"]
                            if profit > sd * 1.2:
                                ns = max(p["sl"], hp - sd * trail_atr * 0.3)
                                if ns > p["sl"]: p["sl"] = ns
                        else:
                            profit = p["entry"] - lp
                            if profit > sd * 1.2:
                                ns = min(p["sl"], lp + sd * trail_atr * 0.3)
                                if ns < p["sl"]: p["sl"] = ns
            for i in sorted(tc, reverse=True):
                positions.pop(i)

        if balance < 50: break
        if len(positions) >= MAX_POSITIONS: continue
        # 24h trading
        if h == 0 and bt.minute < SESSION_COOLDOWN_MIN: continue

        mw = m15[max(0, idx-100):idx+1]
        if len(mw) < 50: continue
        m5w = m5[m5.index <= bt].tail(100)
        if len(m5w) < 20: continue
        m15r = mw.rename(columns=lambda x: x.lower()); m5r = m5w.rename(columns=lambda x: x.lower())

        try:
            i15 = compute_all_indicators(m15r); i5 = compute_all_indicators(m5r)
            ht = h4_trend(mw); sr = compute_sr_levels(mw)
        except: continue
        try: sw = cp.detect_swing_levels(mw); ca = cp.analyze_full(mw, sw); csig = ca.get("signal", "NONE"); ccon = ca.get("confidence", 0)
        except: csig = "NONE"; ccon = 0
        try: res = strategy.analyze(i5, i5, i15, m5r.tail(5), m5r, m15r)
        except: continue

        d = res.get("direction", "NONE"); sc = res.get("setup_score", 0)
        if d == "NONE": bcount += 1; continue
        scount += 1
        if csig == d: sc = min(100, sc + max(1, ccon // 3))
        if sc < MIN_SCORE_BUY: bcount += 1; continue
        if blocked(d, sc, positions, i15, i5, ht, sr, curr): blcount += 1; bcount += 1; continue

        av = i15["atr"].iloc[-1]
        if np.isnan(av) or av <= 0: continue
        sl_dist = av * SL_ATR_MULT
        sl_p = curr - sl_dist if d == "BUY" else curr + sl_dist
        tp = curr + av * TP_ATR_MULT if d == "BUY" else curr - av * TP_ATR_MULT
        rp = FIXED_RISK
        if sc >= HIGH_SCORE_THRESHOLD: rp = HIGH_SCORE_RISK
        lot = max(0.01, round((balance * rp) / (sl_dist * 100), 2))
        positions.append({"dir": d, "entry": curr, "sl": sl_p, "tp": tp, "lots": lot, "entry_time": bt, "score": sc})
        tcount += 1; bcount += 1

    # Close remaining
    if positions:
        lp = m15['close'].iloc[-1]
        for p in positions:
            pnl = (lp - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - lp) * p["lots"] * 100
            p["pnl"] = round(pnl, 2); p["exit"] = "eod"; trades_list.append(p); balance += pnl

    wins = [t for t in trades_list if t.get("pnl", 0) > 0]; losses = [t for t in trades_list if t.get("pnl", 0) <= 0]
    tt = len(trades_list); wr = len(wins)/tt*100 if tt > 0 else 0; net = sum(t.get("pnl", 0) for t in trades_list)
    dd = (peak - min(balance, peak)) / peak * 100 if peak > 0 else 0
    gp = sum(t.get("pnl", 0) for t in wins); gl = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 0.01
    pf = gp/gl if gl > 0 else 999
    aw = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
    al = sum(t.get("pnl", 0) for t in losses) / max(len(losses), 1)
    tp_ex = sum(1 for t in trades_list if t.get("exit") == "tp"); sl_ex = sum(1 for t in trades_list if t.get("exit") == "sl")
    pct = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    print(f"\n  {label}")
    print(f"  FINAL: ${balance:,.2f} ({pct:+.1f}%) | Trades: {tt} | WR: {wr:.1f}% | PF: {pf:.2f} | DD: {dd:.1f}%")
    print(f"  Win: ${aw:+,.2f} | Loss: ${al:+,.2f} | TP={tp_ex} SL={sl_ex}")
    return {"final": balance, "net": net, "trades": tt, "wr": wr, "pf": pf, "dd": dd, "tp": tp_ex, "sl": sl_ex}

# Config A: Current v9.2 (BE_ATR=2.0, BE_BUF=50, TRAIL=0.5, no BE-at-profit)
# Config B: With BE-at-$40 (BE_ATR=2.0, BE_BUF=50, TRAIL=0.5, +$40 BE trigger)
# Config C: With BE-at-$30 (BE_ATR=2.0, BE_BUF=50, TRAIL=0.5, +$30 BE trigger)

print("\n" + "="*55)
print("2-MONTH COMPARISON: Current vs BE-at-Profit")
print("="*55)

# Run current config
r_curr = run_sim("CONFIG A: Current v9.2 (BE+Trail)", be_atr=2.0, be_buf=50, trail_atr=0.5)
r_nobe = run_sim("CONFIG B: No BE (Pure SL/TP)", be_atr=0, be_buf=0, trail_atr=0)
r_partial = run_sim("CONFIG C: SL-to-$10 at +$20", be_atr=0, be_buf=0, trail_atr=0, partial_trigger=20, partial_sl=10)
r_be40 = run_sim("CONFIG D: BE-at-$40 (current)", be_atr=2.0, be_buf=50, trail_atr=0.5, be_profit_usd=40.0)

print(f"\n{'='*55}")
print(f"SUMMARY")
print(f"{'='*55}")
best = max([(r_curr['final'], 'v9.2 BE+Trail'), (r_nobe['final'], 'No BE'), (r_partial['final'], 'SL-to-$10 at +$20'), (r_be40['final'], 'BE-at-$40')])
print(f"  v9.2 BE+Trail:     ${r_curr['final']:,.2f} ({r_curr['trades']} trades, {r_curr['wr']:.0f}% WR, PF {r_curr['pf']:.2f})")
print(f"  No BE (pure):      ${r_nobe['final']:,.2f} ({r_nobe['trades']} trades, {r_nobe['wr']:.0f}% WR, PF {r_nobe['pf']:.2f})")
print(f"  SL-to-$10@+$20:    ${r_partial['final']:,.2f} ({r_partial['trades']} trades, {r_partial['wr']:.0f}% WR, PF {r_partial['pf']:.2f})")
print(f"  BE-at-$40:         ${r_be40['final']:,.2f} ({r_be40['trades']} trades, {r_be40['wr']:.0f}% WR, PF {r_be40['pf']:.2f})")
print(f"  WINNER: {best[1]} at ${best[0]:,.2f}")
print(f"{'='*55}")
