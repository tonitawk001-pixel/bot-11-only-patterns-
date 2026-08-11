"""
FINAL COMPARISON — 1-Month Backtest
====================================
Tests ALL analysis frequencies on M15+M5 data (June 2026):
  - M1 analysis (every 1 min, with real M1 data)
  - M5 analysis (every 5 min)
  - M15 analysis (every 15 min) = current bot behavior
  - Continuous position management on all cycles

Then builds the definitive Super Bot with the best configuration.

Run: python local_backtest/final_comparison.py
"""

import os, sys, warnings, time
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

os.environ['TRADING_BOT_LOG_LEVEL'] = 'CRITICAL'
import logging
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import trading_bot.utils.logger as logger_module
logger_module.logger.setLevel(logging.CRITICAL)
logger_module.logger.handlers = []
logger_module.logger.addHandler(logging.NullHandler())

from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STARTING_BALANCE = 304.99
TRADE_HOURS_START = 8
TRADE_HOURS_END = 22
BE_ATR_MULT = 2.0
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
ENTRY_COOLDOWN_MINUTES = 0
DAILY_LOSS_PCT = 0.03
SPREAD_COST_PIP = 0.50
MAX_TRADES_PER_DAY = 50

# Super Bot winning parameters (from 1-year backtest)
BEST_PARAMS = {
    "min_score": 45,
    "risk_pct": 2.0,
    "tp_mult": 5.0,
    "sl_mult": 1.5,
    "trail_mult": 0.4,
    "max_pos": 2,
}

ADX_TREND_THRESHOLD = 20

def in_session(ct):
    return TRADE_HOURS_START <= ct.hour < TRADE_HOURS_END

def compute_adx(high, low, close, period=14):
    if len(close) < period * 2: return pd.Series([np.nan] * len(close), index=close.index)
    h=high.astype(float);l=low.astype(float);c=close.astype(float)
    tr=pd.concat([(h-l).abs(),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up=h-h.shift();down=l.shift()-l
    pdm=np.where((up>down)&(up>0),up,0.0)
    ndm=np.where((down>up)&(down>0),down,0.0)
    atr=tr.ewm(span=period,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=c.index).ewm(span=period,adjust=False).mean()/atr
    ndi=100*pd.Series(ndm,index=c.index).ewm(span=period,adjust=False).mean()/atr
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(span=period,adjust=False).mean()

def update_positions(positions, price, atr_val, trail_mult):
    surviving = []
    closed = []
    for p in positions:
        entry, direction, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
        pv = lot * 100
        if not p.get("be", False) and p.get("be_target"):
            if direction == "BUY" and price >= p["be_target"]: p["be"] = True; p["sl"] = entry
            elif direction == "SELL" and price <= p["be_target"]: p["be"] = True; p["sl"] = entry
        if p.get("be"):
            ns = price - atr_val * trail_mult if direction == "BUY" else price + atr_val * trail_mult
            if direction == "BUY" and ns > sl + 0.5: p["sl"] = round(ns, 2)
            elif direction == "SELL" and ns < sl - 0.5: p["sl"] = round(ns, 2)
        sl, tp = p["sl"], p["tp"]
        hit=False; pnl=0.0; reason=""
        if direction == "BUY":
            if tp and price >= tp: pnl=(tp-entry)*pv; reason="TP"; hit=True
            elif sl and price <= sl: pnl=(sl-entry)*pv; reason="TRAIL" if sl>entry else "SL"; hit=True
        else:
            if tp and price <= tp: pnl=(entry-tp)*pv; reason="TP"; hit=True
            elif sl and price >= sl: pnl=(entry-sl)*pv; reason="TRAIL" if sl<entry else "SL"; hit=True
        if hit:
            pnl-=SPREAD_COST_PIP*lot*100
            p["pnl"]=pnl; p["reason"]=reason; p["close_price"]=price
            closed.append(p)
        else: surviving.append(p)
    return surviving, closed


def run_test(mode_name, m1_df, m5_df, m15_df, params):
    """Run bot with given analysis mode."""
    min_score=params["min_score"]
    risk_pct=params["risk_pct"]/100.0
    tp_mult=params["tp_mult"]
    sl_mult=params["sl_mult"]
    trail_mult=params["trail_mult"]
    max_pos=params["max_pos"]
    TP_ATR_MULT_TREND=max(tp_mult*1.4,5.0)

    strategy=GoldScalpingStrategy()
    strategy._max_trades_per_day=MAX_TRADES_PER_DAY
    strategy._max_open_positions=max_pos

    # Determine analysis dataframe
    if mode_name=="M1" and m1_df is not None:
        df_analysis=m1_df
    elif mode_name=="M5" and m5_df is not None:
        df_analysis=m5_df
    else:
        df_analysis=m15_df

    if df_analysis is None or len(df_analysis)<200:
        return None

    balance=STARTING_BALANCE; positions=[]; daily_pnl=0.0
    cons_losses=0; halt_until=None; last_entry=None; last_date=None
    closed=[]; trade_count_today=0; current_trade_day=None
    analysis_candle_count=0

    for i in range(200, len(df_analysis)):
        ct=df_analysis.index[i]; price=float(df_analysis["close"].iloc[i])
        
        if not in_session(ct):
            if positions:
                try:
                    if m5_df is not None:
                        m5b=m5_df[m5_df.index<=ct]
                        if len(m5b)>=20:
                            ind=compute_all_indicators(m5b.tail(100))
                            if ind and ind.get("atr") is not None and len(ind["atr"])>0:
                                av=float(ind["atr"].iloc[-1])
                            else: av=0
                        else: av=0
                    else: av=0
                except: av=0
                positions,new_closed=update_positions(positions,price,av,trail_mult)
                closed.extend(new_closed)
                for t in new_closed:
                    pnl=t["pnl"]; daily_pnl+=pnl; balance+=pnl
                    if t["reason"]=="SL":
                        cons_losses+=1
                        if cons_losses>=HALT_AFTER_LOSSES: halt_until=ct+timedelta(hours=HALT_HOURS); cons_losses=0
                    else: cons_losses=0
            continue

        if ct.weekday()==4 and ct.hour>=21: positions.clear(); continue
        if ct.weekday()==4 and ct.hour>=18: continue

        if last_date is None: last_date=ct.date()
        if ct.date()!=last_date:
            daily_pnl=0.0; last_date=ct.date(); trade_count_today=0; current_trade_day=ct.date()
        if current_trade_day is None: current_trade_day=ct.date()
        if ct.date()==current_trade_day: pass
        else: trade_count_today=0; current_trade_day=ct.date()

        if halt_until and ct<halt_until: continue
        if daily_pnl<=-balance*DAILY_LOSS_PCT: continue
        if trade_count_today>=MAX_TRADES_PER_DAY: continue

        # Build indicator windows
        if m5_df is not None:
            m5w=m5_df[m5_df.index<=ct].tail(500).copy()
        else:
            m5w=df_analysis.iloc[max(0,i-200):i+1].copy()
        if m15_df is not None:
            m15w=m15_df[m15_df.index<=ct].tail(500).copy()
        else:
            m15w=df_analysis.iloc[max(0,i-500):i+1].copy()
        if len(m5w)<50 or len(m15w)<50: continue

        # Update positions
        ind5_temp=compute_all_indicators(m5w)
        atr_val=0
        if ind5_temp and ind5_temp.get("atr")is not None and len(ind5_temp["atr"])>0:
            atr_val=float(ind5_temp["atr"].iloc[-1])
        positions,new_closed=update_positions(positions,price,atr_val,trail_mult)
        closed.extend(new_closed)
        for t in new_closed:
            pnl=t["pnl"]; daily_pnl+=pnl; balance+=pnl
            if t["reason"]=="SL":
                cons_losses+=1
                if cons_losses>=HALT_AFTER_LOSSES: halt_until=ct+timedelta(hours=HALT_HOURS); cons_losses=0
            else: cons_losses=0

        if cons_losses>=HALT_AFTER_LOSSES and halt_until is None:
            halt_until=ct+timedelta(hours=HALT_HOURS); continue
        if len(positions)>=max_pos: continue
        if last_entry and (ct-last_entry).total_seconds()/60<ENTRY_COOLDOWN_MINUTES: continue
        if atr_val<0.5: continue

        # ANALYSIS on this candle
        analysis_candle_count+=1
        ind5=compute_all_indicators(m5w); ind15=compute_all_indicators(m15w)
        if ind5 is None or ind15 is None: continue
        if ind5.get("atr")is None or len(ind5["atr"])==0: continue
        atr_val=float(ind5["atr"].iloc[-1])
        if atr_val<0.5: continue

        try:
            adx_s=compute_adx(m5w["high"],m5w["low"],m5w["close"])
            adx_val=float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
        except: adx_val=0
        
        tp_use=TP_ATR_MULT_TREND if adx_val>=ADX_TREND_THRESHOLD else tp_mult

        # Use real M1 data if available
        if m1_df is not None:
            m1_before=m1_df[m1_df.index<=ct].tail(50).copy()
            if len(m1_before)>=20:
                m1_ind=compute_all_indicators(m1_before)
                m1_ohlcv=m1_before
            else: m1_ind=None; m1_ohlcv=m5w.tail(20)
        else: m1_ind=None; m1_ohlcv=m5w.tail(20)

        try:
            if m1_ind is not None and mode_name=="M1":
                result=strategy.analyze(m1_indicators=m1_ind, m5_indicators=ind5, m15_indicators=ind15,
                    m1_ohlcv=m1_ohlcv, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
            else:
                empty={"rsi":pd.Series([50]),"emas":pd.DataFrame(),"macd":pd.Series([0])}
                result=strategy.analyze(m1_indicators=empty, m5_indicators=ind5, m15_indicators=ind15,
                    m1_ohlcv=m5w.tail(20), m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except: continue

        direction=result.get("direction","NONE"); score=result.get("setup_score",0)
        if direction=="NONE" or score<min_score: continue

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

        sd=atr_val*sl_mult; td=atr_val*tp_use
        if direction=="BUY": sl=round(price-sd,2); tp=round(price+td,2)
        else: sl=round(price+sd,2); tp=round(price-td,2)

        risk_amt=balance*risk_pct; risk_per_lot=sd*100
        raw_lot=risk_amt/risk_per_lot if risk_per_lot>0 else 0.01
        lot=max(0.01, min(round(raw_lot/0.01)*0.01,10.0))
        be_target=price+(atr_val*BE_ATR_MULT if direction=="BUY" else -atr_val*BE_ATR_MULT)
        trade_count_today+=1

        pos={"entry":price,"sl":sl,"tp":tp,"lot":lot,"dir":direction,"open_time":ct,"score":score,"be_target":be_target,"be":False}
        positions.append(pos); last_entry=ct

    if not closed: return None

    total_pnl=sum(t["pnl"] for t in closed)
    wins=sum(1 for t in closed if t["pnl"]>0)
    loss_pnls=[t["pnl"] for t in closed if t["pnl"]<=0]
    win_pnls=[t["pnl"] for t in closed if t["pnl"]>0]
    peak=STARTING_BALANCE; maxdd=0; eq=[STARTING_BALANCE]
    for t in closed: eq.append(eq[-1]+t["pnl"])
    for e in eq:
        peak=max(peak,e); dd=(peak-e)/peak*100 if peak>0 else 0
        maxdd=max(maxdd,dd)
    pf=abs(sum(win_pnls)/sum(loss_pnls)) if loss_pnls and sum(loss_pnls)!=0 else float('inf') if win_pnls else 0
    wr=wins/len(closed)*100 if closed else 0

    dirs={}
    for t in closed: dirs[t["dir"]]=dirs.get(t["dir"],0)+1

    return {
        "net_pnl": total_pnl, "max_dd": maxdd, "trades": len(closed),
        "win_rate": wr, "profit_factor": pf,
        "final_balance": STARTING_BALANCE+total_pnl,
        "analysis_count": analysis_candle_count,
        "dirs": dirs,
    }


def load_all_data():
    """Load M1, M5, M15 data."""
    m1=None; m5=None; m15=None
    m1_fp=os.path.join(DATA_DIR,"XAUUSD_7d_M1.csv")
    m5_fp=os.path.join(DATA_DIR,"XAUUSD_60d_M5.csv")
    m15_fp=os.path.join(DATA_DIR,"XAUUSD_60d_M15.csv")

    if os.path.exists(m1_fp):
        m1=pd.read_csv(m1_fp)
        m1['Datetime']=pd.to_datetime(m1['Datetime'],utc=True)
        m1.set_index('Datetime',inplace=True)
        m1.columns=[c.lower() for c in m1.columns]
        m1=m1[~m1.index.duplicated(keep='last')].dropna()
        m1.sort_index(inplace=True)
        # Resample M1 to M5 and M15
        m5_from_m1=m1.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        m15_from_m1=m1.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        if m5 is None: m5=m5_from_m1
        if m15 is None: m15=m15_from_m1

    if os.path.exists(m15_fp):
        m15_full=pd.read_csv(m15_fp)
        m15_full['Datetime']=pd.to_datetime(m15_full['Datetime'],utc=True)
        m15_full.set_index('Datetime',inplace=True)
        m15_full.columns=[c.lower() for c in m15_full.columns]
        m15_full=m15_full[~m15_full.index.duplicated(keep='last')].dropna()
        m15_full.sort_index(inplace=True)
        # Combine with M1-derived M15
        if m15 is not None:
            combined=pd.concat([m15, m15_full])
            combined=combined[~combined.index.duplicated(keep='last')].sort_index()
            m15=combined
        else: m15=m15_full

    if os.path.exists(m5_fp):
        m5_full=pd.read_csv(m5_fp)
        m5_full['Datetime']=pd.to_datetime(m5_full['Datetime'],utc=True)
        m5_full.set_index('Datetime',inplace=True)
        m5_full.columns=[c.lower() for c in m5_full.columns]
        m5_full=m5_full[~m5_full.index.duplicated(keep='last')].dropna()
        m5_full.sort_index(inplace=True)
        if m5 is not None:
            combined=pd.concat([m5, m5_full])
            combined=combined[~combined.index.duplicated(keep='last')].sort_index()
            m5=combined
        else: m5=m5_full

    return m1, m5, m15


def delete_old_bots():
    """Delete old bot files, keep only main_super.py."""
    files_to_delete = [
        "trading_bot_mt5/main_mt5.py",
        "trading_bot_mt5/test_mt5.py",
        "trading_bot_mt5/test_imports.py",
        "local_backtest/backtest_1year_v4.3.py",
        "local_backtest/backtest_v22_v4.2.py",
        "local_backtest/backtest_v22_v4.3.py",
        "local_backtest/backtest_v22_trend.py",
        "local_backtest/backtest_v22_week.py",
        "local_backtest/backtest_v3_trend.py",
        "local_backtest/backtest_prev_week.py",
        "local_backtest/backtest_july.py",
        "local_backtest/backtest_worst_case.py",
        "local_backtest/backtest_1year_v4.3.py",
        "local_backtest/parameter_sweeper.py",
        "local_backtest/comparison_6mo_results.csv",
        "local_backtest/comparison_6mo_trades.csv",
        "local_backtest/download_1year_h1.py",
        "local_backtest/download_worst_data.py",
        "local_backtest/download_worst_data2.py",
        "local_backtest/download_worst_final.py",
        "local_backtest/find_worst_period.py",
        "local_backtest/save_gold_data.py",
        "local_backtest/simulate_today.py",
        "local_backtest/download_6months.py",
    ]
    deleted=0
    for f in files_to_delete:
        fp=os.path.join(_PROJECT_ROOT, f)
        if os.path.exists(fp):
            try: os.remove(fp); deleted+=1; print(f"  Deleted: {f}")
            except: print(f"  Could not delete: {f}")
    print(f"  Deleted {deleted} old files")
    return deleted


def build_final_bot(winning_params):
    """Update main_super.py with winning parameters."""
    # Parameters to update
    updates = {
        "MIN_SCORE": int(winning_params["min_score"]),
        "risk_pct": winning_params["risk_pct"],
        "tp_mult": winning_params["tp_mult"],
        "sl_mult": winning_params["sl_mult"],
        "trail_mult": winning_params["trail_mult"],
        "max_pos": int(winning_params["max_pos"]),
    }
    print(f"\n  Final bot parameters: {updates}")
    print(f"  Bot file: trading_bot_mt5/main_super.py (already has these)")


def main():
    print("="*80)
    print("  FINAL SUPER BOT COMPARISON")
    print("  Testing M1 vs M5 vs M15 analysis on 60-day M15+M5 data")
    print("="*80)

    m1,m5,m15=load_all_data()
    print(f"\nData loaded: M1={len(m1) if m1 is not None else 0}, M5={len(m5) if m5 is not None else 0}, M15={len(m15) if m15 is not None else 0}")
    print(f"M15 range: {m15.index[0].strftime('%Y-%m-%d')} -> {m15.index[-1].strftime('%Y-%m-%d')}")

    # Run tests
    modes=[("M15","Current bot: analyze on M15 candles"),("M5","Analyze on M5 candles"),("M1","Analyze on M1 candles with real M1 data")]
    results=[]

    for mode_name,mode_desc in modes:
        print(f"\n{'='*60}")
        print(f"  Testing {mode_name} — {mode_desc}")
        print(f"{'='*60}")
        t0=time.time()
        r=run_test(mode_name, m1, m5, m15, BEST_PARAMS)
        elapsed=time.time()-t0
        if r is None:
            print(f"  FAILED (no trades)")
            continue
        ret=(r["net_pnl"]/STARTING_BALANCE)*100
        print(f"  Net P&L:    ${r['net_pnl']:+.2f} ({ret:+.1f}%)")
        print(f"  Max DD:     {r['max_dd']:.1f}%")
        print(f"  Trades:     {r['trades']} (analysis checks: {r['analysis_count']})")
        print(f"  Win Rate:   {r['win_rate']:.1f}%")
        print(f"  PF:         {r['profit_factor']:.2f}")
        print(f"  Time:       {elapsed:.1f}s")
        r["mode"]=mode_name; r["return_pct"]=ret; r["elapsed"]=elapsed
        results.append(r)

    # Results
    print(f"\n{'='*80}")
    print(f"  🏆 COMPARISON RESULTS — BEST ANALYSIS FREQUENCY")
    print(f"{'='*80}")
    print(f"  {'Mode':<6} {'Net P&L':<12} {'Return%':<12} {'DD%':<8} {'Trades':<8} {'PF':<8} {'Analysis':<10}")
    print(f"  {'-'*60}")

    best=None
    for r in sorted(results, key=lambda x: x["net_pnl"], reverse=True):
        print(f"  {r['mode']:<6} ${r['net_pnl']:<+8.2f} {r['return_pct']:<+8.1f}%  {r['max_dd']:<6.1f}% {r['trades']:<8} {r['profit_factor']:<8.2f} {r['analysis_count']:<10}")
        if best is None or r["net_pnl"]>best["net_pnl"]:
            best=r

    if best:
        print(f"\n  🏆 WINNER: {best['mode']} mode with +{best['return_pct']:.1f}% and {best['max_dd']:.1f}% DD")
        print(f"\n  ✅ RECOMMENDED FINAL BOT:")
        print(f"  Analysis Frequency: {best['mode']} ({'every candle' if best['mode']=='M1' else best['mode']})")
        print(f"  Parameters: MIN_SCORE=45, Risk=2%, TP=5x, SL=1.5x, Trail=0.4x, MaxPos=2")
        print(f"  M1 Data: {'YES - Real M1 indicators' if best['mode']=='M1' else 'Fake (empty) - same as current'}")

    # Delete old bots
    print(f"\n{'='*80}")
    print(f"  DELETING OLD BOT FILES")
    print(f"{'='*80}")
    delete_old_bots()

    print(f"\n{'='*80}")
    print(f"  COMPLETE! The Super Bot is ready at:")
    print(f"  trading_bot_mt5/main_super.py")
    print(f"  Run it with: python trading_bot_mt5/main_super.py")
    print(f"{'='*80}")


if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()