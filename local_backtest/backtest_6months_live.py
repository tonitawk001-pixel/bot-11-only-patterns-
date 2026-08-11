"""6-MONTH CONTINUOUS BACKTEST — v8.9 Config — Single run with compounding"""
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
BE_ATR_MULT = 1.5; BE_BUFFER_POINTS = 20
HIGH_SCORE_THRESHOLD = 70; HIGH_SCORE_RISK = 0.10
TRAIL_ATR_MULT = 1.0
ATR_VOL_THRESHOLD = 4.0; SESSION_COOLDOWN_MIN = 10
STARTING_BALANCE = 500.00

print("Downloading XAUUSD M15/M5 data...")
mt5.initialize()
end = datetime.now(); start = end - timedelta(days=200)
m15_r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
m5_r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()

m15 = pd.DataFrame(m15_r); m15['time'] = pd.to_datetime(m15['time'], unit='s'); m15.set_index('time', inplace=True)
m5 = pd.DataFrame(m5_r); m5['time'] = pd.to_datetime(m5['time'], unit='s'); m5.set_index('time', inplace=True)

# Filter to exactly 6 months
cutoff = m15.index.max() - timedelta(days=180)
m15 = m15[m15.index >= cutoff]
m5 = m5[m5.index >= m15.index.min()]

print(f"M15: {len(m15)} candles | {m15.index.min()} -> {m15.index.max()}")
print(f"M5:  {len(m5)} candles")

balance = STARTING_BALANCE; peak = STARTING_BALANCE; positions = []; trades = []
daily_pnl = 0.0; last_date = ""
strategy = GoldScalpingStrategy()
bcount = scount = blcount = tcount = 0
monthly_pnl = {}

def h4_trend(w):
    if len(w)<34: return "NEUTRAL"
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

def manage_positions(bar_high, bar_low):
    global balance, daily_pnl, trades
    tc = []
    for i, p in enumerate(positions):
        hp, lp = (bar_high, bar_low) if p["dir"] == "BUY" else (bar_low, bar_high)
        sd = abs(p["entry"] - p["sl"])
        if (p["dir"] == "BUY" and hp >= p["tp"]) or (p["dir"] == "SELL" and lp <= p["tp"]):
            pnl = (p["tp"] - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - p["tp"]) * p["lots"] * 100
            p["pnl"] = round(pnl, 2); p["exit"] = "tp"; tc.append(i)
            balance += pnl; daily_pnl += pnl
        elif (p["dir"] == "BUY" and lp <= p["sl"]) or (p["dir"] == "SELL" and hp >= p["sl"]):
            pnl = (p["sl"] - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - p["sl"]) * p["lots"] * 100
            p["pnl"] = round(pnl, 2); p["exit"] = "sl"; tc.append(i)
            balance += pnl; daily_pnl += pnl
        elif p["sl"] and p["sl"] > 0:
            bt = p["entry"] + (sd * BE_ATR_MULT / SL_ATR_MULT) if p["dir"] == "BUY" else p["entry"] - (sd * BE_ATR_MULT / SL_ATR_MULT)
            if (p["dir"] == "BUY" and hp >= bt) or (p["dir"] == "SELL" and lp <= bt):
                ns = p["entry"] + (BE_BUFFER_POINTS * 0.01) if p["dir"] == "BUY" else p["entry"] - (BE_BUFFER_POINTS * 0.01)
                if (p["dir"] == "BUY" and ns > p["sl"]) or (p["dir"] == "SELL" and ns < p["sl"]): p["sl"] = ns
            if p["dir"] == "BUY":
                profit = hp - p["entry"]
                if profit > sd * 1.2:
                    ns = max(p["sl"], hp - sd * TRAIL_ATR_MULT * 0.3)
                    if ns > p["sl"]: p["sl"] = ns
            else:
                profit = p["entry"] - lp
                if profit > sd * 1.2:
                    ns = min(p["sl"], lp + sd * TRAIL_ATR_MULT * 0.3)
                    if ns < p["sl"]: p["sl"] = ns
    for i in sorted(tc, reverse=True):
        trades.append(positions.pop(i))

print("\n" + "="*55)
print("6-MONTH CONTINUOUS BACKTEST (v8.9)")
print("="*55)

for idx in range(100, len(m15)):
    bar = m15.iloc[idx]; bt = bar.name; curr = bar['close']; h = bt.hour
    
    date_str = bt.strftime("%Y-%m-%d")
    if date_str != last_date:
        daily_pnl = 0.0; last_date = date_str; strategy.reset_daily()
    if balance > peak: peak = balance
    if positions: manage_positions(bar['high'], bar['low'])
    if balance < 50: print(f"BALANCE FLOOR at {bt}"); break
    if len(positions) >= MAX_POSITIONS: continue
    if not (12 <= h < 16): continue
    if h == 12 and bt.minute < SESSION_COOLDOWN_MIN: continue
    
    mw = m15[max(0, idx-100):idx+1]
    if len(mw) < 50: continue
    m5w = m5[m5.index <= bt].tail(100)
    if len(m5w) < 20: continue
    m15r = mw.rename(columns=lambda x: x.lower()); m5r = m5w.rename(columns=lambda x: x.lower())
    
    try: i15 = compute_all_indicators(m15r); i5 = compute_all_indicators(m5r); ht = h4_trend(mw); sr = compute_sr_levels(mw)
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
    sl_dist = av * SL_ATR_MULT; sl_p = curr - sl_dist if d == "BUY" else curr + sl_dist
    tp = curr + av * TP_ATR_MULT if d == "BUY" else curr - av * TP_ATR_MULT
    rp = FIXED_RISK
    if sc >= HIGH_SCORE_THRESHOLD: rp = HIGH_SCORE_RISK
    lot = max(0.01, round((balance * rp) / (sl_dist * 100), 2))
    positions.append({"dir": d, "entry": curr, "sl": sl_p, "tp": tp, "lots": lot, "entry_time": bt, "score": sc})
    tcount += 1; bcount += 1
    
    # Track monthly P&L as we go
    m_key = bt.strftime("%Y-%m")
    monthly_pnl[m_key] = monthly_pnl.get(m_key, 0) + 0  # placeholder, updated on close
    
    if bcount % 800 == 0:
        print(f"  Bar {bcount} | Bal: ${balance:.2f} | Peak: ${peak:.2f} | Trades: {tcount}")

# Close remaining
if positions:
    lp = m15['close'].iloc[-1]
    for p in positions:
        pnl = (lp - p["entry"]) * p["lots"] * 100 if p["dir"] == "BUY" else (p["entry"] - lp) * p["lots"] * 100
        p["pnl"] = round(pnl, 2); p["exit"] = "eod"; trades.append(p); balance += pnl

# Recompute monthly P&L from closed trades
monthly_pnl = {}
for t in trades:
    m = t["entry_time"].strftime("%Y-%m")
    monthly_pnl[m] = monthly_pnl.get(m, 0) + t.get("pnl", 0)

wins = [t for t in trades if t.get("pnl", 0) > 0]; losses = [t for t in trades if t.get("pnl", 0) <= 0]
tt = len(trades); wr = len(wins)/tt*100 if tt > 0 else 0; net = sum(t.get("pnl", 0) for t in trades)
dd = (peak - min(balance, peak)) / peak * 100 if peak > 0 else 0
gp = sum(t.get("pnl", 0) for t in wins); gl = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 0.01
pf = gp/gl if gl > 0 else 999
bw = sum(1 for t in trades if t.get("dir") == "BUY"); sw = tt - bw
bwr = len([t for t in wins if t.get("dir") == "BUY"]) / max(bw, 1) * 100
swr = len([t for t in wins if t.get("dir") == "SELL"]) / max(sw, 1) * 100
aw = sum(t.get("pnl", 0) for t in wins) / max(len(wins), 1)
al = sum(t.get("pnl", 0) for t in losses) / max(len(losses), 1)
tp_ex = sum(1 for t in trades if t.get("exit") == "tp")
sl_ex = sum(1 for t in trades if t.get("exit") == "sl")

print(f"""
{'='*55}
FINAL RESULTS
{'='*55}
Period: {m15.index.min().strftime('%Y-%m-%d')} -> {m15.index.max().strftime('%Y-%m-%d')}
Bars: {bcount}

FINAL: ${balance:,.2f} | Net: ${net:+,.2f} ({(balance-STARTING_BALANCE)/STARTING_BALANCE*100:+.1f}%)
PEAK: ${peak:,.2f} | Max DD: {dd:.1f}%

TRADES: {tt} ({bw}B/{sw}S) | ~{tt/6:.0f}/mo
WIN RATE: {wr:.1f}% | BUY: {bwr:.1f}% | SELL: {swr:.1f}%
PROFIT FACTOR: {pf:.2f}
AVG WIN: ${aw:+,.2f} | AVG LOSS: ${al:+,.2f}
EXITS: TP={tp_ex} SL={sl_ex}
""")

print("MONTHLY P&L:")
for m in sorted(monthly_pnl.keys()):
    pnl_m = monthly_pnl[m]
    print(f"  {m}: ${pnl_m:+,.2f} {'UP' if pnl_m > 0 else 'DOWN'}")

print("\nLAST 5 TRADES:")
for t in trades[-5:]:
    print(f"  {t['entry_time']}: {t['dir']} @ {t['entry']:.2f} -> {t['exit']} | PnL: ${t['pnl']:+,.2f}")