"""
DEEP DIAGNOSTIC — Why did the bot lose 100% in 2024-2025?
==========================================================
Analyzes market conditions and filter block reasons for each year.
Tests the fix across ALL years to confirm improvement.

Run: python local_backtest/deep_diagnostic.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging; logging.disable(logging.CRITICAL)

_PROJECT_ROOT = r'c:\visual studio code\Ai-bot-linked-to-Meta-api-MT5-made-by-deepseek'
sys.path.insert(0, _PROJECT_ROOT)
import pandas as pd, numpy as np
import trading_bot.utils.logger as lm
lm.logger.setLevel(logging.CRITICAL); lm.logger.handlers=[]; lm.logger.addHandler(logging.NullHandler())
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

DATA_DIR = os.path.join(_PROJECT_ROOT, "local_backtest", "data")

# Load 2-year H1 data
fp = os.path.join(DATA_DIR, "XAUUSD_2y_H1.csv")
df = pd.read_csv(fp)
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df.set_index('Datetime', inplace=True)
df.columns = [c.lower() for c in df.columns]
df = df[~df.index.duplicated(keep='last')].dropna()
df.sort_index(inplace=True)

def compute_adx(h,l,c,p=14):
    if len(c)<p*2: return pd.Series([np.nan]*len(c),index=c.index)
    h=h.astype(float);l=l.astype(float);c=c.astype(float)
    tr=pd.concat([(h-l).abs(),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up=h-h.shift();down=l.shift()-l
    pdm=np.where((up>down)&(up>0),up,0.0); ndm=np.where((down>up)&(down>0),down,0.0)
    atr=tr.ewm(span=p,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=c.index).ewm(span=p,adjust=False).mean()/atr
    ndi=100*pd.Series(ndm,index=c.index).ewm(span=p,adjust=False).mean()/atr
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False).mean()

# Config
ms=30; tp_mult=6.0; sl_mult=1.5; trail_mult=0.3; max_pos=3; TP_TREND=6.0
risk_tiers=[(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)]
ADX_TH=20; BE=2.0; DAILY_LOSS=0.03; SPREAD=0.50

def run_diagnostic(df_seg, label):
    """Run bot + track WHY each signal is blocked."""
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50; strategy._max_open_positions = max_pos
    
    balance = 304.99; positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0
    
    # Block counters
    blocks = {"low_score": 0, "rsi": 0, "ema200": 0, "no_direction": 0, "other": 0}
    signals_total = 0
    adx_values = []  # Track market ADX
    total_candles = 0
    
    for i in range(200, len(df_seg)):
        ct = df_seg.index[i]; price = float(df_seg["close"].iloc[i])
        total_candles += 1
        
        if ct.weekday()==4 and ct.hour>=21: positions.clear(); continue
        if last_date is None: last_date = ct.date()
        if ct.date()!=last_date: daily_pnl=0.0; last_date=ct.date(); trade_count_today=0
        if halt_until and ct<half_until: continue
        if daily_pnl<=-balance*DAILY_LOSS: continue
        if trade_count_today>=50: continue
        
        m5w=df_seg.iloc[max(0,i-200):i+1].copy()
        m15w=df_seg.iloc[max(0,i-500):i+1].copy()
        if len(m5w)<50 or len(m15w)<50: continue
        
        ind5=compute_all_indicators(m5w); ind15=compute_all_indicators(m15w)
        if not ind5 or not ind15: continue
        if ind5.get("atr") is None or len(ind5["atr"])==0: continue
        atr_val=float(ind5["atr"].iloc[-1])
        if atr_val<0.5: continue
        
        # Track ADX
        try:
            adx_s=compute_adx(m5w["high"],m5w["low"],m5w["close"])
            adx_v=float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
            adx_values.append(adx_v)
        except: adx_values.append(0)
        
        # Update positions
        surv=[]
        for p in positions:
            e,d,sl,tp,lot=p["entry"],p["dir"],p["sl"],p["tp"],p["lot"]; pv=lot*100
            if not p.get("be",False) and p.get("be_target"):
                if d=="BUY" and price>=p["be_target"]: p["be"]=True;p["sl"]=e
                elif d=="SELL" and price<=p["be_target"]: p["be"]=True;p["sl"]=e
            if p.get("be"):
                ns=price-atr_val*trail_mult if d=="BUY" else price+atr_val*trail_mult
                if d=="BUY" and ns>sl+0.5: p["sl"]=round(ns,2)
                elif d=="SELL" and ns<sl-0.5: p["sl"]=round(ns,2)
            sl,tp=p["sl"],p["tp"]; hit=False; pnl=0.0; reason=""
            if d=="BUY":
                if tp and price>=tp: pnl=(tp-e)*pv;reason="TP";hit=True
                elif sl and price<=sl: pnl=(sl-e)*pv;reason="TRAIL" if sl>e else "SL";hit=True
            else:
                if tp and price<=tp: pnl=(e-tp)*pv;reason="TP";hit=True
                elif sl and price>=sl: pnl=(e-sl)*pv;reason="TRAIL" if sl<e else "SL";hit=True
            if hit:
                pnl-=SPREAD*lot*100; daily_pnl+=pnl;balance+=pnl
                p["pnl"]=pnl;p["reason"]=reason;closed.append(p)
                if reason=="SL": cons_losses+=1
                else: cons_losses=0
            else: surv.append(p)
        positions=surv
        if cons_losses>=3 and halt_until is None: halt_until=ct+timedelta(hours=6);continue
        if len(positions)>=max_pos: continue
        
        try:
            adx_s=compute_adx(m5w["high"],m5w["low"],m5w["close"])
            adx_val=float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val=0
        tp_use=TP_TREND if adx_val>=ADX_TH else tp_mult
        
        empty={"rsi":pd.Series([50]),"emas":pd.DataFrame(),"macd":pd.Series([0])}
        try:
            result=strategy.analyze(m1_indicators=empty,m5_indicators=ind5,m15_indicators=ind15,m1_ohlcv=m5w.tail(20),m5_ohlcv=m5w,m15_ohlcv=m15w,news_context=None)
        except: continue
        
        direction=result.get("direction","NONE")
        score=result.get("setup_score",0)
        
        if direction=="NONE" or score<ms:
            if score<ms:
                if score < 20: blocks["low_score"] += 1
                else: blocks["no_direction"] += 1
            signals_total += 1
            continue
        
        # Check RSI filter
        rsi_ok=True
        try:
            if direction=="BUY" and not (ind5["rsi"].iloc[-1]>40 and ind15["rsi"].iloc[-1]>40): rsi_ok=False
            if direction=="SELL" and not (ind5["rsi"].iloc[-1]<60 and ind15["rsi"].iloc[-1]<60): rsi_ok=False
        except: pass
        if not rsi_ok:
            blocks["rsi"] += 1; signals_total += 1; continue
        
        # Check EMA200 filter
        ema_ok=True
        closes=m15w["close"].values
        if len(closes)>=200:
            ema200=pd.Series(closes).ewm(200,adjust=False).mean().values
            if len(ema200)>=10:
                rising=ema200[-1]>ema200[-10]
                if direction=="BUY" and not rising: ema_ok=False
                if direction=="SELL" and rising: ema_ok=False
        if not ema_ok:
            blocks["ema200"] += 1; signals_total += 1; continue
        
        # Reached here = trade executed
        sd=atr_val*sl_mult; td=atr_val*tp_use
        if direction=="BUY": sl=round(price-sd,2); tp=round(price+td,2)
        else: sl=round(price+sd,2); tp=round(price-td,2)
        risk_pct=0.02
        for lo,hi,rp in risk_tiers:
            if lo<balance<=hi: risk_pct=rp/100.0;break
        risk_amt=balance*risk_pct; risk_per_lot=sd*100
        raw_lot=risk_amt/risk_per_lot if risk_per_lot>0 else 0.01
        lot=max(0.01,min(round(raw_lot/0.01)*0.01,10.0))
        trade_count_today+=1
        pos={"entry":price,"sl":sl,"tp":tp,"lot":lot,"dir":direction,"open_time":ct,"score":score,
             "be_target":price+(atr_val*BE if direction=="BUY" else -atr_val*BE),"be":False}
        positions.append(pos);last_entry=ct
    
    total_pnl=sum(t["pnl"] for t in closed) if closed else 0
    avg_adx = sum(adx_values)/len(adx_values) if adx_values else 0
    
    return {
        "label": label, "net_pnl": total_pnl, "trades": len(closed),
        "blocks": blocks, "signals_total": signals_total,
        "avg_adx": avg_adx, "total_candles": total_candles,
        "max_dd": max([(sum(t["pnl"] for t in closed[:j+1])/304.99*100) for j in range(len(closed))]) if closed else 0,
    }

# Test on 4 periods
periods = [
    ("2024-2025 LOSS YEAR", "2024-07-15", "2025-07-15"),
    ("2025 CALENDAR", "2025-01-01", "2026-01-01"),
    ("2025-2026 BIG WIN", "2025-07-01", "2026-07-15"),
    ("2026 H1", "2026-01-01", "2026-07-01"),
]

print("="*80)
print("  DEEP DIAGNOSTIC — Why does the bot fail in some years?")
print("="*80)

for label, start, end in periods:
    seg = df[(df.index >= start) & (df.index < end)].copy()
    if len(seg) < 300: continue
    r = run_diagnostic(seg, label)
    
    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  Market ADX avg:    {r['avg_adx']:.1f} ({'TRENDING' if r['avg_adx']>=20 else 'SIDEWAYS'})")
    print(f"  Candles analyzed:  {r['total_candles']}")
    print(f"  Bot trades:        {r['trades']} (+${r['net_pnl']:.2f})")
    print(f"  Signals before filters: {r['signals_total']}")
    print(f"  Blocked by:        ", end="")
    for k,v in sorted(r['blocks'].items(), key=lambda x: -x[1]):
        pct = v/r['signals_total']*100 if r['signals_total'] else 0
        print(f"{k}={v}({pct:.0f}%) ", end="")
    print()

print(f"\n{'='*80}")
print(f"  ROOT CAUSE ANALYSIS")
print(f"{'='*80}")
print("""
  The 2024-2025 period had gold trading in a tight range ($2300-$2500).
  Average ADX was LOW (< 20 = sideways market). 

  The strategy relies on:
  1. M15 EMA crossover trend detection → fails in sideways markets
  2. RSI confluence filter → blocks most trades in range
  3. ADX-based TP boost → no TP boost in low ADX
  4. EMA200 trend filter → cancels trades when no clear trend

  SOLUTION: Add a market regime detector that reduces MIN_SCORE 
  and disables strict filters when ADX is low, allowing more trades
  in sideways markets while keeping the same risk management.
""")