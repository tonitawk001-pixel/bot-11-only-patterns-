"""2015 Gold Backtest using actual Super Bot strategy on daily data."""
import os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging; logging.disable(logging.CRITICAL)

_PROJECT_ROOT = r'c:\visual studio code\Ai-bot-linked-to-Meta-api-MT5-made-by-deepseek'
sys.path.insert(0, _PROJECT_ROOT)
import pandas as pd, numpy as np
import yfinance as yf
import trading_bot.utils.logger as lm
lm.logger.setLevel(logging.CRITICAL); lm.logger.handlers=[]; lm.logger.addHandler(logging.NullHandler())
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

# Download 2015 daily data
print("Downloading 2015 gold daily data from Yahoo Finance...")
df = yf.download('GC=F', start='2015-01-01', end='2016-01-01', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df.rename(columns=str.lower)
df.index = pd.to_datetime(df.index)
print(f"2015: {len(df)} daily candles, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")

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

# Bot config
ms=30; tp_mult=6.0; sl_mult=1.5; trail_mult=0.3; max_pos=3; TP_TREND=6.0
risk_tiers=[(0,500,2.0),(500,2000,2.5),(2000,10000,3.0),(10000,999999,4.0)]
ADX_TH=20; BE=2.0; DAILY_LOSS=0.03; SPREAD=0.50

strategy=GoldScalpingStrategy()
strategy._max_trades_per_day=50; strategy._max_open_positions=max_pos

balance=304.99; positions=[]; daily_pnl=0.0; cons_losses=0
halt_until=None; last_entry=None; last_date=None; closed=[]; trade_count_today=0
trading_days_set=set()

for i in range(200, len(df)):
    ct=df.index[i]; price=float(df["close"].iloc[i])
    trading_days_set.add(ct)
    if ct.weekday()==4 and ct.hour>=21: positions.clear(); continue
    if last_date is None: last_date=ct.date()
    if ct.date()!=last_date: daily_pnl=0.0; last_date=ct.date(); trade_count_today=0
    if halt_until and ct<halt_until: continue
    if daily_pnl<=-balance*DAILY_LOSS: continue
    if trade_count_today>=50: continue
    
    # On daily, use same data for both M5 and M15 views
    m5w=df.iloc[max(0,i-200):i+1].copy()
    m15w=df.iloc[max(0,i-500):i+1].copy()
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
        result=strategy.analyze(m1_indicators=empty,m5_indicators=ind5,m15_indicators=ind15,
            m1_ohlcv=m5w.tail(20),m5_ohlcv=m5w,m15_ohlcv=m15w,news_context=None)
    except: continue
    
    direction=result.get("direction","NONE"); score=result.get("setup_score",0)
    if direction=="NONE" or score<ms: continue
    try:
        if direction=="BUY" and not (ind5["rsi"].iloc[-1]>40 and ind15["rsi"].iloc[-1]>40): continue
        if direction=="SELL" and not (ind5["rsi"].iloc[-1]<60 and ind15["rsi"].iloc[-1]<60): continue
    except: pass
    
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
wins=sum(1 for t in closed if t["pnl"]>0) if closed else 0
peak=304.99; maxdd=0; eq=[304.99]
for t in closed: eq.append(eq[-1]+t["pnl"])
for e in eq: peak=max(peak,e); dd=(peak-e)/peak*100 if peak>0 else 0; maxdd=max(maxdd,dd)
loss_pnls=[t["pnl"] for t in closed if t["pnl"]<=0] if closed else []
win_pnls=[t["pnl"] for t in closed if t["pnl"]>0] if closed else []
pf=abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls)!=0 else (float('inf') if win_pnls else 0)
wr=wins/len(closed)*100 if closed else 0

print(f"\n{'='*60}")
print(f"  2015 BACKTEST — Super Bot on Daily Data")
print(f"{'='*60}")
print(f"  Data:         Daily candles (252)")
print(f"  ⚠️  Note:     Daily data is approximate")
print(f"                Bot uses M15/M5 in real trading")
print(f"")
print(f"  Net P&L:      ${total_pnl:+.2f}")
print(f"  Return:       {total_pnl/304.99*100:.1f}%")
print(f"  Max DD:       {maxdd:.1f}%")
print(f"  Trades:       {len(closed)}")
print(f"  Win Rate:     {wr:.1f}%")
print(f"  PF:           {pf:.2f}")
print(f"  Final Bal:    ${304.99+total_pnl:.2f}")
print(f"  Trades/Day:   {len(closed)/252:.2f}")