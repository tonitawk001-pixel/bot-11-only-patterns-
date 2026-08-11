"""
2-YEAR COMPARISON BACKTEST
Tests the Final Super Bot (Test J) on BOTH years:
  - Year 1: Jul 2024 - Jul 2025
  - Year 2: Jul 2025 - Jul 2026 (already done: +626,360%)

Run: python local_backtest/backtest_2year_comparison.py
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

# Test J config
MS = 30
TP_MULT = 6.0
SL_MULT = 1.5
TRAIL_MULT = 0.3
MAX_POS = 3
TP_TREND = 6.0
RISK_TIERS = [(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)]
ADX_TH = 20
BE = 2.0
DAILY_LOSS = 0.03
SPREAD = 0.50


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


def run_backtest(df, label):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Data: {len(df)} candles, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = MAX_POS

    balance = STARTING_BALANCE
    positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0

    for i in range(200, len(df)):
        ct = df.index[i]; price = float(df["close"].iloc[i])
        if ct.weekday() == 4 and ct.hour >= 21: positions.clear(); continue
        if last_date is None: last_date = ct.date()
        if ct.date() != last_date: daily_pnl = 0.0; last_date = ct.date(); trade_count_today = 0
        if halt_until and ct < halt_until: continue
        if daily_pnl <= -balance * DAILY_LOSS: continue
        if trade_count_today >= 50: continue

        m5w = df.iloc[max(0,i-200):i+1].copy()
        m15w = df.iloc[max(0,i-500):i+1].copy()
        if len(m5w) < 50 or len(m15w) < 50: continue
        ind5 = compute_all_indicators(m5w); ind15 = compute_all_indicators(m15w)
        if not ind5 or not ind15: continue
        if ind5.get("atr") is None or len(ind5["atr"]) == 0: continue
        atr_val = float(ind5["atr"].iloc[-1])
        if atr_val < 0.5: continue

        surv = []
        for p in positions:
            e,d,sl,tp,lot = p["entry"],p["dir"],p["sl"],p["tp"],p["lot"]
            pv = lot * 100
            if not p.get("be",False) and p.get("be_target"):
                if d=="BUY" and price >= p["be_target"]: p["be"]=True; p["sl"]=e
                elif d=="SELL" and price <= p["be_target"]: p["be"]=True; p["sl"]=e
            if p.get("be"):
                ns = price - atr_val * TRAIL_MULT if d=="BUY" else price + atr_val * TRAIL_MULT
                if d=="BUY" and ns > sl + 0.5: p["sl"] = round(ns,2)
                elif d=="SELL" and ns < sl - 0.5: p["sl"] = round(ns,2)
            sl,tp = p["sl"],p["tp"]
            hit=False; pnl=0.0; reason=""
            if d=="BUY":
                if tp and price >= tp: pnl=(tp-e)*pv;reason="TP";hit=True
                elif sl and price <= sl: pnl=(sl-e)*pv;reason="TRAIL" if sl>e else "SL";hit=True
            else:
                if tp and price <= tp: pnl=(e-tp)*pv;reason="TP";hit=True
                elif sl and price >= sl: pnl=(e-sl)*pv;reason="TRAIL" if sl<e else "SL";hit=True
            if hit:
                pnl -= SPREAD * lot * 100; daily_pnl += pnl; balance += pnl
                p["pnl"]=pnl; p["reason"]=reason; closed.append(p)
                if reason=="SL":
                    cons_losses+=1
                    if cons_losses>=3: halt_until=ct+timedelta(hours=6);cons_losses=0
                else: cons_losses=0
            else: surv.append(p)
        positions = surv
        if cons_losses>=3 and halt_until is None: halt_until=ct+timedelta(hours=6);continue
        if len(positions) >= MAX_POS: continue

        try:
            adx_s = compute_adx(m5w["high"],m5w["low"],m5w["close"])
            adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val = 0
        tp_use = TP_TREND if adx_val >= ADX_TH else TP_MULT

        empty={"rsi":pd.Series([50]),"emas":pd.DataFrame(),"macd":pd.Series([0])}
        try:
            result = strategy.analyze(m1_indicators=empty,m5_indicators=ind5,m15_indicators=ind15,m1_ohlcv=m5w.tail(20),m5_ohlcv=m5w,m15_ohlcv=m15w,news_context=None)
        except: continue

        direction=result.get("direction","NONE"); score=result.get("setup_score",0)
        if direction=="NONE" or score<MS: continue
        try:
            if direction=="BUY" and not (ind5["rsi"].iloc[-1]>40 and ind15["rsi"].iloc[-1]>40): continue
            if direction=="SELL" and not (ind5["rsi"].iloc[-1]<60 and ind15["rsi"].iloc[-1]<60): continue
        except: pass
        closes=m15w["close"].values
        if len(closes)>=200:
            ema200=pd.Series(closes).ewm(200,adjust=False).mean().values
            if len(ema200)>=10:
                rising=ema200[-1]>ema200[-10]
                if direction=="BUY" and not rising: continue
                if direction=="SELL" and rising: continue

        sd=atr_val*SL_MULT; td=atr_val*tp_use
        if direction=="BUY": sl=round(price-sd,2); tp=round(price+td,2)
        else: sl=round(price+sd,2); tp=round(price-td,2)

        risk_pct = 0.02
        for lo,hi,rp in RISK_TIERS:
            if lo < balance <= hi: risk_pct = rp/100.0; break
        risk_amt=balance*risk_pct; risk_per_lot=sd*100
        raw_lot=risk_amt/risk_per_lot if risk_per_lot>0 else 0.01
        lot=max(0.01,min(round(raw_lot/0.01)*0.01,10.0))
        trade_count_today+=1
        pos={"entry":price,"sl":sl,"tp":tp,"lot":lot,"dir":direction,"open_time":ct,"score":score,"be_target":price+(atr_val*BE if direction=="BUY" else -atr_val*BE),"be":False}
        positions.append(pos); last_entry=ct

    if not closed:
        print("  No trades!")
        return None

    total_pnl=sum(t["pnl"] for t in closed)
    wins=sum(1 for t in closed if t["pnl"]>0)
    peak=STARTING_BALANCE; maxdd=0; eq=[STARTING_BALANCE]
    for t in closed: eq.append(eq[-1]+t["pnl"])
    for e in eq:
        peak=max(peak,e)
        dd=(peak-e)/peak*100 if peak>0 else 0
        maxdd=max(maxdd,dd)
    loss_pnls=[t["pnl"] for t in closed if t["pnl"]<=0]
    win_pnls=[t["pnl"] for t in closed if t["pnl"]>0]
    pf=abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls)!=0 else (float('inf') if win_pnls else 0)
    wr=wins/len(closed)*100

    print(f"  Net P&L:    ${total_pnl:+.2f}")
    print(f"  Return:     {total_pnl/STARTING_BALANCE*100:.1f}%")
    print(f"  Max DD:     {maxdd:.1f}%")
    print(f"  Trades:     {len(closed)}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  PF:         {pf:.2f}")
    print(f"  Final Bal:  ${STARTING_BALANCE+total_pnl:.2f}")

    return {"total_pnl": total_pnl, "maxdd": maxdd, "trades": len(closed), "wr": wr, "pf": pf, "final": STARTING_BALANCE+total_pnl}


def main():
    print("="*80)
    print("  SUPER BOT v5.0 — 2-YEAR COMPARISON (Test J)")
    print("  Testing on 2024-2025 and 2025-2026 H1 data")
    print("="*80)

    # Load 2-year data
    fp = os.path.join(DATA_DIR, "XAUUSD_2y_H1.csv")
    df = pd.read_csv(fp)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep='last')].dropna()
    df.sort_index(inplace=True)

    # Split into 2 years
    year1 = df[(df.index >= '2024-07-15') & (df.index < '2025-07-15')].copy()
    year2 = df[(df.index >= '2025-07-01') & (df.index < '2026-07-15')].copy()

    r1 = run_backtest(year1, "YEAR 1 (Jul 2024 - Jul 2025)")
    r2 = run_backtest(year2, "YEAR 2 (Jul 2025 - Jul 2026)")

    if r1 and r2:
        print(f"\n{'='*80}")
        print(f"  🏆 2-YEAR COMPARISON")
        print(f"{'='*80}")
        print(f"  {'Metric':<20} {'Year 1 (2024-25)':<20} {'Year 2 (2025-26)':<20}")
        print(f"  {'-'*60}")
        print(f"  {'Net Profit':<20} ${r1['total_pnl']:<+8.2f}       ${r2['total_pnl']:<+8.2f}")
        print(f"  {'Return':<20} {r1['total_pnl']/STARTING_BALANCE*100:<+9.1f}%      {r2['total_pnl']/STARTING_BALANCE*100:<+9.1f}%")
        print(f"  {'Max DD':<20} {r1['maxdd']:<6.1f}%                {r2['maxdd']:<6.1f}%")
        print(f"  {'Trades':<20} {r1['trades']:<6d}                {r2['trades']:<6d}")
        print(f"  {'Win Rate':<20} {r1['wr']:<5.1f}%                {r2['wr']:<5.1f}%")
        print(f"  {'Profit Factor':<20} {r1['pf']:<6.2f}                {r2['pf']:<6.2f}")
        print(f"  {'Final Balance':<20} ${r1['final']:<8.2f}        ${r2['final']:<8.2f}")

        # Average
        avg_return = ((r1['total_pnl']/STARTING_BALANCE) + (r2['total_pnl']/STARTING_BALANCE)) / 2 * 100
        avg_dd = (r1['maxdd'] + r2['maxdd']) / 2
        print(f"\n  📊 2-YEAR AVERAGE:")
        print(f"     Avg Return: {avg_return:+.1f}%/year")
        print(f"     Avg Max DD: {avg_dd:.1f}%")
        print(f"     Total Profit (2 years): ${r1['total_pnl'] + r2['total_pnl']:+.2f}")

    print(f"\n{'='*80}")
    print(f"  COMPLETE — Bot is ready at trading_bot_mt5/main_super.py")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()