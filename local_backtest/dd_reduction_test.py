"""
DD REDUCTION TEST — Can we reduce drawdown without cutting profit?
Tests 5 strategies on both years. Run: python local_backtest/dd_reduction_test.py
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

# 5 configs
CONFIGS = [
    {"name": "BASELINE (current)", "ms": 30, "mp": 3, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.03, "grad": False},
    {"name": "A: Daily loss=2%", "ms": 30, "mp": 3, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.02, "grad": False},
    {"name": "B: Grad risk reduction", "ms": 30, "mp": 3, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.03, "grad": True},
    {"name": "C: MaxPos=2", "ms": 30, "mp": 2, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.03, "grad": False},
    {"name": "D: A + B combined", "ms": 30, "mp": 3, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.02, "grad": True},
    {"name": "E: A + B + MaxPos=2", "ms": 30, "mp": 2, "tr": 0.3, "slm": 1.5, "risk_t": [(0,500,2),(500,2000,2.5),(2000,10000,3),(10000,999999,4)], "dl": 0.02, "grad": True},
]

def run_test(seg, cfg):
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50; strategy._max_open_positions = cfg["mp"]
    balance = 304.99; positions = []; daily_pnl = 0.0; cons_losses = 0
    halt_until = None; last_entry = None; last_date = None
    closed = []; trade_count_today = 0
    grad_active = False; grad_losses = 0
    
    # Graduated risk: after 2 losses in a row, reduce risk by 50%
    # Reset after any win
    
    for i in range(200, len(seg)):
        ct = seg.index[i]; price = float(seg["close"].iloc[i])
        if ct.weekday()==4 and ct.hour>=21: positions.clear(); continue
        if last_date is None: last_date = ct.date()
        if ct.date()!=last_date: daily_pnl=0.0; last_date=ct.date(); trade_count_today=0
        if halt_until and ct<halt_until: continue
        if daily_pnl<=-balance*cfg["dl"]: continue
        if trade_count_today>=50: continue
        
        m5w=seg.iloc[max(0,i-200):i+1].copy(); m15w=seg.iloc[max(0,i-500):i+1].copy()
        if len(m5w)<50 or len(m15w)<50: continue
        ind5=compute_all_indicators(m5w); ind15=compute_all_indicators(m15w)
        if not ind5 or not ind15: continue
        if ind5.get("atr") is None or len(ind5["atr"])==0: continue
        atr_val=float(ind5["atr"].iloc[-1])
        if atr_val<0.5: continue
        
        surv=[]
        for p in positions:
            e,d,sl,tp,lot=p["entry"],p["dir"],p["sl"],p["tp"],p["lot"]; pv=lot*100
            if not p.get("be",False) and p.get("be_target"):
                if d=="BUY" and price>=p["be_target"]: p["be"]=True;p["sl"]=e
                elif d=="SELL" and price<=p["be_target"]: p["be"]=True;p["sl"]=e
            if p.get("be"):
                ns=price-atr_val*cfg["tr"] if d=="BUY" else price+atr_val*cfg["tr"]
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
                pnl-=0.5*lot*100; daily_pnl+=pnl;balance+=pnl
                p["pnl"]=pnl;p["reason"]=reason;closed.append(p)
                if reason=="SL":
                    cons_losses+=1; grad_losses+=1
                    if cons_losses>=3: halt_until=ct+timedelta(hours=6);cons_losses=0
                else: cons_losses=0; grad_losses=0
            else: surv.append(p)
        positions=surv
        if cons_losses>=3 and halt_until is None: halt_until=ct+timedelta(hours=6);continue
        if len(positions)>=cfg["mp"]: continue
        
        try:
            adx_s=compute_adx(m5w["high"],m5w["low"],m5w["close"])
            adx_val=float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val=0
        tp_use=6.0 if adx_val>=20 else 4.0
        
        empty={"rsi":pd.Series([50]),"emas":pd.DataFrame(),"macd":pd.Series([0])}
        try: result=strategy.analyze(m1_indicators=empty,m5_indicators=ind5,m15_indicators=ind15,m1_ohlcv=m5w.tail(20),m5_ohlcv=m5w,m15_ohlcv=m15w,news_context=None)
        except: continue
        
        direction=result.get("direction","NONE"); score=result.get("setup_score",0)
        if direction=="NONE" or score<cfg["ms"]: continue
        try:
            if direction=="BUY" and not (ind5["rsi"].iloc[-1]>40 and ind15["rsi"].iloc[-1]>40): continue
            if direction=="SELL" and not (ind5["rsi"].iloc[-1]<60 and ind15["rsi"].iloc[-1]<60): continue
        except: pass
        
        sd=atr_val*cfg["slm"]; td=atr_val*tp_use
        if direction=="BUY": sl=round(price-sd,2); tp=round(price+td,2)
        else: sl=round(price+sd,2); tp=round(price-td,2)
        
        # Risk with graduated reduction
        risk_pct = 0.02
        for lo,hi,rp in cfg["risk_t"]:
            if lo<balance<=hi: risk_pct=rp/100.0;break
        if cfg["grad"] and grad_losses >= 2:
            risk_pct *= 0.5  # 50% risk reduction after 2+ losses
        
        risk_amt=balance*risk_pct; risk_per_lot=sd*100
        raw_lot=risk_amt/risk_per_lot if risk_per_lot>0 else 0.01
        lot=max(0.01,min(round(raw_lot/0.01)*0.01,10.0))
        trade_count_today+=1
        pos={"entry":price,"sl":sl,"tp":tp,"lot":lot,"dir":direction,"open_time":ct,"score":score,
             "be_target":price+(atr_val*2 if direction=="BUY" else -atr_val*2),"be":False}
        positions.append(pos);last_entry=ct
    
    total_pnl=sum(t["pnl"] for t in closed) if closed else 0
    wins=sum(1 for t in closed if t["pnl"]>0) if closed else 0
    peak=304.99; maxdd=0; eq=[304.99]
    for t in closed: eq.append(eq[-1]+t["pnl"])
    for e in eq: peak=max(peak,e); dd=(peak-e)/peak*100 if peak>0 else 0; maxdd=max(maxdd,dd)
    loss_pnls=[t["pnl"] for t in closed if t["pnl"]<=0] if closed else []
    win_pnls=[t["pnl"] for t in closed if t["pnl"]>0] if closed else []
    pf=abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls)!=0 else (float('inf') if win_pnls else 0)
    wr=wins/len(closed)*100 if closed else 0
    return total_pnl, maxdd, len(closed), wr, pf

# Test on both years
tests = [
    ("2024-2025 (was losing)", "2024-07-15", "2025-07-15"),
    ("2025-2026 (best year)", "2025-07-01", "2026-07-15"),
]

print("="*100)
print("  DD REDUCTION TEST — 6 Strategies × 2 Years")
print("="*100)

for label, start, end in tests:
    seg = df[(df.index >= start) & (df.index < end)].copy()
    if len(seg) < 300: continue
    
    print(f"\n  {label}:")
    print(f"  {'Strategy':<30} {'Net P&L':<12} {'DD%':<8} {'Trades':<8} {'WR%':<6} {'PF':<7} {'Score':<8}")
    print(f"  {'-'*75}")
    
    results = []
    for cfg in CONFIGS:
        total_pnl, maxdd, trades, wr, pf = run_test(seg, cfg)
        # Score: profit per DD ratio (higher = better risk-adjusted)
        score = total_pnl / maxdd if maxdd > 0 else 0
        results.append((cfg["name"], total_pnl, maxdd, trades, wr, pf, score))
    
    # Sort by score (best risk-adjusted first)
    results.sort(key=lambda x: x[6], reverse=True)
    
    for name, pnl, dd, trades, wr, pf, score in results:
        dd_ok = "✅" if dd <= 40 else ("⚠️" if dd <= 50 else "❌")
        print(f"  {name:<30} ${pnl:<+8.2f} {dd:<6.1f}%{dd_ok} {trades:<8} {wr:<5.1f}% {pf:<6.2f} {score:<8.1f}")
    
    best_name = results[0][0]
    print(f"  🏆 Best: {best_name}")

print(f"\n{'='*100}")
print(f"  RECOMMENDATION:")
print(f"  If a version beats BASELINE on BOTH profit AND DD, it will be adopted.")
print(f"{'='*100}")