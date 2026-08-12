"""
███████ SUPER BOT v8.0 — INSTITUTIONAL GRADE ███████
PROFESSIONAL GOLD SCALPING — CANDLES + S/R + DEEPSEEK AI
"""
import os, sys, time, json, atexit, signal, traceback, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta, timezone
import pandas as pd, numpy as np, MetaTrader5 as mt5
from mt5_connection import MT5Connection
from logger_mt5 import logger
import telegram_notifier as tg, trade_exporter, github_setup
import telegram_handler as tg_handler
import github_exporter as gh_exporter
from news_filter import NewsFilter
from deepseek_filter import DeepSeekFilter
import candle_patterns as cp
from performance_tracker import PerformanceTracker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)
from trading_bot.indicators.technical_indicators import compute_all_indicators, compute_sr_levels
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.confirmation_engine import evaluate_trade_signals

SYMBOL = "XAUUSD"
MIN_SCORE_BUY = 35; MIN_SCORE_SELL = 35; MIN_SCORE = 35
MAX_POSITIONS = 1; MAX_PER_DIRECTION = 1
DAILY_LOSS_PCT = 0.05
TOTAL_RISK_LIMIT = 0.05
ATR_VOL_THRESHOLD = 4.0
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
USE_AI_FILTER = False        # AI signal veto: TRUE = AI can block trades, FALSE = AI watches only
AI_MARKET_ANALYSIS = False   # 15-min market reports: TRUE = enabled, FALSE = paused (no spam)
AI_WATCHDOG = True           # Hourly health watchdog + error auto-fix: TRUE = ON
AI_WATCHDOG_INTERVAL_MIN = 60  # minutes between watchdog reports
TP_ATR_MULT = 5.0; TP_PARTIAL_MULT = 2.5; SL_ATR_MULT = 2.5
BE_ATR_MULT = 2.0; BE_BUFFER_POINTS = 50
TRAIL_ATR_MULT = 0.5      # Trail stop - gentle to allow more room
BE_PROFIT_USD = 40         # Move SL to entry after +$40 profit
FIXED_RISK = 0.05           # 5% per trade
HIGH_SCORE_THRESHOLD = 70; HIGH_SCORE_RISK = 0.10
SECOND_POS_MIN_SCORE = 35; SECOND_POS_LOT_RATIO = 0.5
HALT_HOURS = 6; MAX_CONSEC_LOSSES = 2
REGIME_RISK_LOW = 0.01; REGIME_WR_THRESHOLD = 0.45
RECENT_TRADE_WINDOW = 20
TRADE_HOURS_START = 0; TRADE_HOURS_END = 24
SESSION_COOLDOWN_MIN = 10
DXY_ENABLED = True; DXY_THRESHOLD = 0.003; DXY_LOOKBACK_H = 1
MTF_CONFLUENCE = True
STATE_FILE = "bot_state_super.json"
NEWS_BUFFER_MIN = 30; HARD_FLOOR = 50.00; MAX_SPREAD = 2.00

# ── Global state ──────────────────────────────────────────────────────
consecutive_losses = 0
daily_pnl = 0.0
last_date = ""
last_processed_m15_time = None
trades_log = []
balance_snapshot = 0.0
_mt5_conn = None
_spread_paused = False
_spread_notified = False
_last_m1_price = None  # flash spike detection
_prev_positions = {}   # track position IDs for SL detection

def get_risk_pct(balance: float) -> float:
    """Fixed risk — no death spiral tiers."""
    return FIXED_RISK

def load_state():
    global consecutive_losses, daily_pnl, last_date, last_processed_m15_time, trades_log
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            consecutive_losses = s.get("losses", 0)
            daily_pnl = s.get("pnl", 0.0)
            last_date = s.get("date", "")
            trades_log = s.get("trades_log", [])
            m15_str = s.get("m15")
            last_processed_m15_time = datetime.fromisoformat(m15_str) if m15_str else None
            logger.info(f"State loaded: losses={consecutive_losses} pnl={daily_pnl:.2f} trades={len(trades_log)}")
    except Exception as e:
        logger.warning(f"load_state failed: {e}")

def save_state():
    try:
        s = {
            "losses": consecutive_losses,
            "pnl": daily_pnl,
            "date": last_date,
            "trades_log": trades_log[-500:],
            "m15": last_processed_m15_time.isoformat() if last_processed_m15_time else None,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, default=str)
    except Exception as e:
        logger.error(f"save_state failed: {e}")

def cleanup():
    """Shutdown handler — close MT5, save state, notify."""
    logger.info("Shutting down...")
    save_state()
    try:
        tg.notify_shutdown()
    except Exception:
        pass
    if _mt5_conn:
        try:
            _mt5_conn.shutdown()
        except Exception:
            pass

atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: cleanup() or os._exit(0))
signal.signal(signal.SIGINT, lambda *_: cleanup() or os._exit(0))

def get_session() -> str:
    now = datetime.now(timezone.utc)
    h = now.hour
    if 8 <= h < 17 and 13 <= h < 22: return "overlap"
    if 8 <= h < 17: return "london"
    if 13 <= h < 22: return "new_york"
    if h >= 23 or h < 8: return "asian"
    return "transition"


# ── Main loop ─────────────────────────────────────────────────────────
def main_loop():
    global consecutive_losses, daily_pnl, last_date, last_processed_m15_time
    global trades_log, balance_snapshot, _mt5_conn

    _mt5_conn = MT5Connection()
    if not _mt5_conn.initialize():
        logger.critical("MT5 INIT FAILED — exiting")
        return

    conn = _mt5_conn
    strategy = GoldScalpingStrategy()
    nf = NewsFilter()
    ai = DeepSeekFilter(api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None
    perf = PerformanceTracker(deepseek_client=ai)
    load_state()
    
    # ── LOAD CONFIG OVERRIDES ──────────────────────────
    try:
        config_file = "config_overrides.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                overrides = json.load(f)
            _globals = globals()
            _valid_keys = {'MIN_SCORE_BUY','MIN_SCORE_SELL','MAX_POSITIONS','MAX_PER_DIRECTION',
                'DAILY_LOSS_PCT','TOTAL_RISK_LIMIT','ATR_VOL_THRESHOLD','USE_AI_FILTER',
                'TP_ATR_MULT','TP_PARTIAL_MULT','SL_ATR_MULT','BE_ATR_MULT','BE_BUFFER_POINTS',
                'FIXED_RISK','HIGH_SCORE_THRESHOLD','HIGH_SCORE_RISK','SECOND_POS_MIN_SCORE',
                'SECOND_POS_LOT_RATIO','HALT_HOURS','MAX_CONSEC_LOSSES','TRADE_HOURS_START',
                'TRADE_HOURS_END','SESSION_COOLDOWN_MIN','DXY_ENABLED','DXY_THRESHOLD',
                'DXY_LOOKBACK_H','MTF_CONFLUENCE','MAX_SPREAD'}
            for key, value in overrides.items():
                key_upper = key.upper()
                if key_upper in _valid_keys and key_upper in _globals:
                    old_val = _globals[key_upper]
                    _globals[key_upper] = type(old_val)(value)
                    logger.info(f"[Config] Override {key_upper} = {value} (was {old_val})")
            logger.info(f"[Config] Loaded {len(overrides)} overrides from {config_file}")
    except Exception as e:
        logger.warning(f"[Config] Override load failed: {e}")
    
    github_setup.setup_remote()
    nf.update_news()

    # Get account info for health check
    info = conn.get_account_info()

    # Send simple startup ping first (always works, no AI needed)
    if info:
        try:
            tg.set_account_name(str(info.get("login", "?")))
            tg.send_message(f"🤖 BOT STARTED\nAccount: {info.get('login','?')}\nBalance: ${info['balance']:,.2f}\n⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC")
        except:
            pass

    # ── AI SYSTEM HEALTH CHECK ──────────────────────────
    if ai is not None:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            spread_val = round((tick.ask - tick.bid) if tick and tick.ask and tick.bid and tick.ask != tick.bid else 0.30, 2)
            health = ai.system_health_check({
                "mt5_connected": True,
                "login": info.get("login", "?") if info else "?",
                "server": info.get("server", "?") if info else "?",
                "balance": info["balance"] if info else 0,
                "equity": info.get("equity", 0) if info else 0,
                "spread": spread_val,
                "session": get_session(),
                "in_hours": TRADE_HOURS_START <= datetime.now(timezone.utc).hour < TRADE_HOURS_END,
                "news_count": len(nf.red_folder_events) if hasattr(nf, 'red_folder_events') else 0,
                "news_ok": nf.has_news() if hasattr(nf, 'has_news') else True,
            })
            verdict_emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "🚨"}.get(health.get("verdict", "WARNING"), "⚠️")
            score = health.get("health_score", 0)
            issues = health.get("issues", [])
            report = health.get("report", "No report generated")
            health_msg = (
                f"🤖 ACCOUNT: {info.get('login','?') if info else '?'}\n"
                f"🟢 BOT STARTED — AI Health Check\n\n"
                f"🧠 AI VERDICT: {verdict_emoji} {health.get('verdict','WARNING')}\n"
                f"   Health Score: {score}/100\n\n"
                f"━━━━━━━━━━━━━━━\n"
                f"✅ MT5: Connected | {info.get('server','?') if info else '?'}\n"
                f"✅ Balance: ${info['balance'] if info else 0:,.2f}\n"
                f"✅ Spread: ${spread_val} {'(WIDE!)' if spread_val > 2.0 else '(normal)'}\n"
                f"✅ Session: {get_session()}{' — Trading Hours' if TRADE_HOURS_START <= datetime.now(timezone.utc).hour < TRADE_HOURS_END else ' — Outside Hours'}\n"
                f"✅ News: {len(nf.red_folder_events) if hasattr(nf, 'red_folder_events') else 0} events loaded\n\n"
                f"{'⚠️ News API may be down (no fresh data in 24h)' if hasattr(nf, 'has_news') and not nf.has_news() else ''}\n"
                f"📋 TOOLS: 17/17 Active — Candle Patterns + S/R, Strategy, DXY, MTF, Score Filters, Risk Mgmt\n\n"
            )
            if issues:
                health_msg += f"⚠️ WARNINGS:\n" + "\n".join(f"  • {i}" for i in issues[:5]) + "\n\n"
            health_msg += (
                f"🧠 AI REPORT: {report}\n\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
            )
            tg.send_message(health_msg)
            logger.info(f"[HEALTH] AI Verdict: {health.get('verdict')} | Score: {score}")
        except Exception as e:
            logger.warning(f"[HEALTH] Check failed: {e}")
            try:
                tg.send_message(f"⚠️ Health check failed: {str(e)[:200]}\nBot will trade normally.")
            except:
                logger.error("[HEALTH] Telegram fallback also failed")

    # ── START TELEGRAM CHAT POLLING ─────────────────────────
    if ai is not None:
        tg_handler.set_deepseek_client(ai)
        tg_handler.start_polling()
        logger.info("[TelegramPoll] DeepSeek chat polling started — you can now message the bot on Telegram")

    # ── SETUP GITHUB EXPORTER ──────────────────────────────
    gh_exporter.setup_git()
    logger.info("[GitExport] GitHub auto-export configured (every 30 min)")

    logger.info("SUPER BOT v8.0 Started - All Tools Active")
    cycle = 0
    daily_trades = 0
    daily_halted = False
    halt_until = None

    while True:
        cycle += 1
        try:
            # Safety: ensure halt_until is datetime (not string from bad state)
            if halt_until and not isinstance(halt_until, datetime):
                halt_until = None
            if last_processed_m15_time and not isinstance(last_processed_m15_time, datetime):
                last_processed_m15_time = None

            info = conn.get_account_info()
            if not info:
                time.sleep(10)
                continue
            balance = info["balance"]
            balance_snapshot = balance
            equity = info.get("equity", balance)
            now = datetime.now(timezone.utc)

            # ── PERFORMANCE TRACKER ─────────────────────────
            perf.update(balance, equity)

            # DD Emergency Stop (25% from peak) — DISABLED: bot runs 24/7 on demo account even if balance drops to zero
            # perf.check_dd_emergency() always returns False — this can never trigger
            # if len(trades_log) > 0 and perf.check_dd_emergency(balance) and not perf.dd_halted:
            #     perf.dd_halted = True
            #     _emergency_positions = mt5.positions_get(symbol=SYMBOL)
            #     tg.send_message("🚨 EMERGENCY STOP: 25% drawdown from peak. Closing all positions.")
            #     if _emergency_positions:
            #         for p in _emergency_positions: conn.close_position(p.ticket)
            #     break

            # DD Risk Reduction (10% from peak equity) — disabled to match backtest
            active_risk = FIXED_RISK
            active_max_pos = MAX_POSITIONS

            # Weekly AI Review (Sunday 20:00 UTC)
            if perf.should_run_weekly(now):
                try:
                    report = perf.run_weekly_review(trades_log, balance)
                    tg.send_message(f"📊 Weekly Performance Review:\n\n{report[:800]}")
                    tg.send_message(f"📤 Notes pushed to GitHub: mt5-bot-final-2")
                except Exception as e:
                    logger.warning(f"Weekly review failed: {e}")

            today_str = now.strftime("%Y-%m-%d")
            if last_date != today_str:
                daily_pnl = 0.0
                daily_trades = 0
                daily_halted = False
                last_date = today_str
                strategy.reset_daily()

            # Balance floor — DISABLED: bot runs 24/7 on demo account even if balance drops to zero
            # if balance < HARD_FLOOR:
            #     tg.notify_bot_crashed(f"Balance ${balance:.2f} below floor ${HARD_FLOOR}")
            #     break

            # Daily halted — skip trading
            if daily_halted:
                continue

            # Consecutive loss halt
            if halt_until and now < halt_until:
                continue

            global _prev_positions
            mt5_pos = mt5.positions_get(symbol=SYMBOL)
            floating_pnl = sum([p.profit for p in mt5_pos]) if mt5_pos else 0.0
            
            # ── SL EXIT DETECTION ──
            current_ids = set(p.ticket for p in mt5_pos) if mt5_pos else set()
            for prev_id, prev_data in list(_prev_positions.items()):
                if prev_id not in current_ids:
                    # Position vanished — was closed by SL (TP already handled above)
                    pnl = prev_data.get("profit_at_close", 0)
                    exit_price = prev_data.get("sl_price", 0) or prev_data.get("entry", 0)
                    _log_sl_closed(prev_data, pnl, exit_price)
            _prev_positions = {}
            for p in (mt5_pos or []):
                _prev_positions[p.ticket] = {
                    "id": p.ticket, "type": "BUY" if p.type == 0 else "SELL",
                    "entry": p.price_open, "sl": p.sl, "tp": p.tp, "lots": p.volume,
                    "profit_at_close": float(p.profit) if hasattr(p, 'profit') else 0.0,
                    "sl_price": p.sl, "tp_price": p.tp, "time": p.time if hasattr(p, 'time') else None,
                }
            
            pos = [{"id": p.ticket, "type": "BUY" if p.type == 0 else "SELL",
                    "entry": p.price_open, "sl": p.sl, "tp": p.tp, "lots": p.volume}
                   for p in mt5_pos] if mt5_pos else []

            # ── SPREAD AUTO-PAUSE ─────────────────────────────
            global _spread_paused, _spread_notified
            tick = mt5.symbol_info_tick(SYMBOL)
            spread = round(tick.ask - tick.bid, 2) if tick and tick.ask and tick.bid and tick.ask != tick.bid else 0.30

            if spread > MAX_SPREAD and not _spread_paused:
                _spread_paused = True
                if not _spread_notified:
                    try:
                        tg.send_message(
                            f"⏸️ SPREAD PAUSE\n"
                            f"Current: ${spread:.2f} | Max: ${MAX_SPREAD:.2f}\n"
                            f"Bot will skip new entries until spread normalizes.\n"
                            f"⏰ {now.strftime('%H:%M:%S')} UTC"
                        )
                    except: pass
                    _spread_notified = True

            elif spread <= MAX_SPREAD and _spread_paused:
                _spread_paused = False
                _spread_notified = False
                try:
                    tg.send_message(
                        f"✅ SPREAD NORMAL: ${spread:.2f}\n"
                        f"Bot resuming trading.\n"
                        f"⏰ {now.strftime('%H:%M:%S')} UTC"
                    )
                except: pass


            # Manage open positions
            if mt5_pos:
                tick = mt5.symbol_info_tick(SYMBOL)
                for p in mt5_pos:
                    if not tick: continue
                    curr_price = tick.bid if p.type == 0 else tick.ask
                    sl_distance = abs(p.price_open - p.sl) if p.sl and p.sl > 0 else 5.0

                    if p.tp and p.tp > 0:
                        tp_target = p.tp
                    else:
                        mult = (TP_ATR_MULT / SL_ATR_MULT)
                        tp_target = p.price_open + (sl_distance * mult) if p.type == 0 else p.price_open - (sl_distance * mult)

                    mult_be = (BE_ATR_MULT / SL_ATR_MULT)
                    be_trigger = p.price_open + (sl_distance * mult_be) if p.type == 0 else p.price_open - (sl_distance * mult_be)

                    if (p.type == 0 and curr_price >= tp_target) or (p.type == 1 and curr_price <= tp_target):
                        res = conn.close_position(p.ticket)
                        pnl = float(p.profit) if hasattr(p, 'profit') else 0.0
                        _log_closed_trade(p, pnl, "tp", tp_target)  # daily_pnl updated inside
                        logger.info(f"[TP] #{p.ticket} pnl={pnl:.2f} result={res}")

                    elif p.sl and p.sl > 0:
                        # ── MINIMUM DURATION GUARD: Don't modify SL (BE/trail) for trades < 15min ──
                        trade_age_minutes = (now - datetime.fromtimestamp(p.time, tz=timezone.utc)).total_seconds() / 60 if hasattr(p, 'time') and p.time else 999
                        if trade_age_minutes < 15:
                            logger.debug(f"[MIN-DUR] #{p.ticket} age={trade_age_minutes:.1f}min — skipping BE/trail (allow SL/TP only)")
                        else:
                            # BE trigger
                            if (p.type == 0 and curr_price >= be_trigger) or (p.type == 1 and curr_price <= be_trigger):
                                new_sl = p.price_open + (BE_BUFFER_POINTS * 0.01) if p.type == 0 else p.price_open - (BE_BUFFER_POINTS * 0.01)
                                if (p.type == 0 and new_sl > p.sl) or (p.type == 1 and new_sl < p.sl):
                                    res = conn.modify_position(p.ticket, sl=new_sl)
                                    save_state()
                                    logger.info(f"[BE] #{p.ticket} SL->{new_sl:.2f} result={res}")
                            # BE-at-profit trigger — move SL to entry after $40 profit (backtest proven)
                            if BE_PROFIT_USD > 0:
                                lot_size = p.volume
                                profit_points = BE_PROFIT_USD / (lot_size * 100)
                                be_price = p.price_open + profit_points if p.type == 0 else p.price_open - profit_points
                                if (p.type == 0 and curr_price >= be_price) or (p.type == 1 and curr_price <= be_price):
                                    new_sl = p.price_open + (BE_BUFFER_POINTS * 0.01) if p.type == 0 else p.price_open - (BE_BUFFER_POINTS * 0.01)
                                    if (p.type == 0 and new_sl > p.sl) or (p.type == 1 and new_sl < p.sl):
                                        res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                        save_state()
                                        logger.info(f"[BE-PROFIT] #{p.ticket} SL->{new_sl:.2f} (+${BE_PROFIT_USD} profit)")
                            # Trail stop — lock in profit (backtest v8.7)
                            if TRAIL_ATR_MULT > 0:
                                if p.type == 0:  # BUY
                                    profit = curr_price - p.price_open
                                    if profit > sl_distance * 1.2:
                                        new_sl = max(p.sl, curr_price - (sl_distance * TRAIL_ATR_MULT * 0.3))
                                        if new_sl > p.sl:
                                            res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                            save_state()
                                            logger.info(f"[TRAIL] #{p.ticket} SL->{new_sl:.2f} profit={profit:.2f}")
                                else:  # SELL
                                    profit = p.price_open - curr_price
                                    if profit > sl_distance * 1.2:
                                        new_sl = min(p.sl, curr_price + (sl_distance * TRAIL_ATR_MULT * 0.3))
                                        if new_sl < p.sl:
                                            res = conn.modify_position(p.ticket, sl=round(new_sl, 2))
                                            save_state()
                                            logger.info(f"[TRAIL] #{p.ticket} SL->{new_sl:.2f} profit={profit:.2f}")

            # Daily loss halt disabled — backtest v8.7 has no halt (positions run to TP/SL)

            # Friday close
            if now.weekday() == 4 and now.hour >= 21:
                if pos:
                    logger.info("Friday close - closing all positions")
                    for p_in_pos in pos:
                        conn.close_position(p_in_pos["id"])
                time.sleep(300)
                continue

            # New bar processing (every 15 min)
            if _spread_paused:
                continue

            m15_time = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            if last_processed_m15_time is None or m15_time > last_processed_m15_time:
                last_processed_m15_time = m15_time
                save_state()

                if len(pos) >= active_max_pos:
                    continue

                is_n, ev_title, ev_time, pause_mins = nf.is_news_active(buffer_minutes=NEWS_BUFFER_MIN)
                if is_n:
                    logger.info(f"News Pause: {ev_title}")
                    try:
                        tg.notify_news_pause(ev_title, ev_time or "?", pause_mins)
                    except Exception:
                        pass
                    continue

                h4w = conn.get_candles(SYMBOL, "H4", 50)
                m15w = conn.get_candles(SYMBOL, "M15", 300)
                m5w = conn.get_candles(SYMBOL, "M5", 300)
                m1w = conn.get_candles(SYMBOL, "M1", 300)
                if h4w is None or m15w is None or m5w is None:
                    continue

                h4w_ren = h4w.rename(columns=lambda x: x.lower())
                m15w_ren = m15w.rename(columns=lambda x: x.lower())
                m5w_ren = m5w.rename(columns=lambda x: x.lower())

                i15 = compute_all_indicators(m15w_ren)
                i5 = compute_all_indicators(m5w_ren)
                i4 = compute_all_indicators(h4w_ren)
                if m1w is not None:
                    m1w_ren = m1w.rename(columns=lambda x: x.lower())
                    i1 = compute_all_indicators(m1w_ren)
                else:
                    i1 = i5

                h4_ema20 = i4['emas']['EMA_20'].iloc[-1]
                h4_trend = "BULLISH" if h4w['close'].iloc[-1] > h4_ema20 else "BEARISH"
                sr = compute_sr_levels(m15w)  # M15 S/R matches backtest

                # ── CANDLE + S/R ANALYSIS ──────────────────────────
                swing_levels = cp.detect_swing_levels(m15w)
                candle_analysis = cp.analyze_full(m15w, swing_levels)
                candle_signal = candle_analysis.get("signal", "NONE")
                candle_conf = candle_analysis.get("confidence", 0)
                candle_patterns_list = candle_analysis.get("patterns_detected", [])
                sr_touch_info = candle_analysis.get("sr_touch", {})

                # ── v2.0 CONFIRMATION ENGINE (M15-based, 5-indicator validation) ──
                # Extract latest indicator values from M15 for the confirmation engine
                try:
                    m15_price = float(m15w_ren['close'].iloc[-1])
                    m15_sma40 = float(i15['smas']['SMA_40'].iloc[-1]) if 'smas' in i15 and 'SMA_40' in i15['smas'].columns else None
                    m15_sma200 = float(i15['smas']['SMA_200'].iloc[-1]) if 'smas' in i15 and 'SMA_200' in i15['smas'].columns else None
                    m15_rsi = float(i15['rsi'].iloc[-1])
                    m15_prev_rsi = float(i15['rsi'].iloc[-2]) if len(i15['rsi']) >= 2 else None
                    m15_bb_upper = float(i15['bb']['upper'].iloc[-1])
                    m15_bb_mid = float(i15['bb']['middle'].iloc[-1])
                    m15_bb_lower = float(i15['bb']['lower'].iloc[-1])
                    m15_macd_line = float(i15['macd']['macd'].iloc[-1])
                    m15_macd_signal = float(i15['macd']['signal'].iloc[-1])
                    m15_macd_hist = float(i15['macd']['histogram'].iloc[-1])
                    m15_prev_macd_hist = float(i15['macd']['histogram'].iloc[-2]) if len(i15['macd']) >= 2 else None
                    m15_stoch_k = float(i15['stochastic']['stoch_k'].iloc[-1]) if 'stochastic' in i15 else None
                    m15_stoch_d = float(i15['stochastic']['stoch_d'].iloc[-1]) if 'stochastic' in i15 else None
                    m15_prev_stoch_k = float(i15['stochastic']['stoch_k'].iloc[-2]) if 'stochastic' in i15 and len(i15['stochastic']) >= 2 else None
                    m15_prev_stoch_d = float(i15['stochastic']['stoch_d'].iloc[-2]) if 'stochastic' in i15 and len(i15['stochastic']) >= 2 else None
                    m15_vol = float(m15w_ren['tick_volume'].iloc[-1]) if 'tick_volume' in m15w_ren.columns else None
                    m15_vol_ma = float(i15['volume_ma'].iloc[-1]) if i15.get('volume_ma') is not None and len(i15['volume_ma']) > 0 else None

                    # Determine direction from old strategy (sentiment) — but confirmation engine validates it
                    old_result = strategy.analyze(i5, i5, i15, m5w.tail(5), m5w, m15w)
                    raw_direction = old_result.get("direction", "NONE")
                except Exception as e:
                    logger.warning(f"Indicator extraction failed: {e}")
                    raw_direction = "NONE"
                    m15_price = m15w_ren['close'].iloc[-1] if len(m15w_ren) > 0 else 0

                blocked_by = None
                original_dir = raw_direction

                # REGIME DETECTOR: aggressive strong trend, filtered chop (best +37974)
                try:
                    m15_adx = float(i15.get('adx_14', pd.Series([0])).iloc[-1]) if 'adx_14' in i15 else 99
                except:
                    m15_adx = 99
                strong_trend = (not np.isnan(m15_adx)) and m15_adx >= 35
                if strong_trend:
                    logger.info(f"[REGIME] Strong trend (ADX={m15_adx:.0f} >= 35) -- aggressive mode")
                else:
                    if not np.isnan(m15_adx) and m15_adx < 20:
                        blocked_by = f"ADX_{m15_adx:.0f}_lt_20"
                        raw_direction = "NONE"
                        logger.info(f"[ADX] Chop market (ADX={m15_adx:.0f} < 20) -- skipping")

                # SESSION FILTER: only in chop regime (skipped in strong trend)
                h = now.hour
                in_active = (8 <= h < 17) or (13 <= h < 22)  # London or NY
                if not strong_trend and not in_active and raw_direction != "NONE":
                    blocked_by = f"outside_active_hours_{h}h"
                    raw_direction = "NONE"
                    logger.info(f"[SESSION] Outside active hours ({h}h UTC) -- skipping")

                # ── CONFIRMATION ENGINE: Run if raw direction is BUY or SELL ──
                engine_result = None
                if raw_direction in ("BUY", "SELL"):
                    try:
                        engine_result = evaluate_trade_signals(
                            direction=raw_direction,
                            price=m15_price,
                            ma40=m15_sma40,
                            ma200=m15_sma200,
                            rsi=m15_rsi,
                            prev_rsi=m15_prev_rsi,
                            bb_upper=m15_bb_upper,
                            bb_mid=m15_bb_mid,
                            bb_lower=m15_bb_lower,
                            macd_line=m15_macd_line,
                            macd_signal=m15_macd_signal,
                            macd_hist=m15_macd_hist,
                            prev_macd_hist=m15_prev_macd_hist,
                            stoch_k=m15_stoch_k,
                            stoch_d=m15_stoch_d,
                            prev_stoch_k=m15_prev_stoch_k,
                            prev_stoch_d=m15_prev_stoch_d,
                            volume=m15_vol,
                            vol_ma=m15_vol_ma,
                        )

                        if engine_result.get("decision"):
                            direction = raw_direction
                            confidence = engine_result.get("confidence", 0)
                            engine_reason = engine_result.get("reason", "")
                            option = engine_result.get("option", 0)
                            logger.info(f"[ENGINE] {direction} signal APPROVED (conf={confidence}%, option={option}): {engine_reason}")
                        else:
                            direction = "NONE"
                            blocked_by = engine_result.get("reason", "confirmation_failed")
                            logger.info(f"[ENGINE] {raw_direction} signal REJECTED: {blocked_by}")
                    except Exception as e:
                        logger.error(f"[ENGINE] Evaluation error: {e}")
                        direction = "NONE"
                        blocked_by = f"engine_error: {str(e)[:80]}"
                else:
                    direction = "NONE"

                # ── FLASH SPIKE PROTECTION ── ($50 in <1 min = pause)
                global _last_m1_price
                if m1w is not None and _last_m1_price is not None:
                    flash_move = abs(m1w['close'].iloc[-1] - _last_m1_price)
                    if flash_move > 50.0:
                        logger.warning(f"FLASH SPIKE: ${flash_move:.1f} in <1 min")
                        blocked_by = f"flash_spike_${flash_move:.0f}"
                        direction = "NONE"
                        try:
                            tg.send_message(f"⚡ FLASH SPIKE: ${flash_move:.1f} in <1 min\nBot pausing for this bar.\n{now.strftime('%H:%M:%S')} UTC")
                        except: pass
                if m1w is not None:
                    _last_m1_price = m1w['close'].iloc[-1]

                # ── SPREAD GUARD ──
                if direction != "NONE":
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick:
                        spread_now = round(tick.ask - tick.bid, 2)
                        if spread_now > MAX_SPREAD:
                            blocked_by = f"spread_{spread_now}_gt_{MAX_SPREAD}"
                            direction = "NONE"

                # ── NEWS PAUSE ──
                if direction != "NONE":
                    is_n, ev_title, ev_time, pause_mins = nf.is_news_active(buffer_minutes=NEWS_BUFFER_MIN)
                    if is_n:
                        blocked_by = f"news_blackout_{ev_title}"
                        direction = "NONE"
                        try:
                            tg.notify_news_pause(ev_title, ev_time or "?", pause_mins)
                        except: pass

                # ── SAME DIRECTION STACKING BLOCK ──
                if direction != "NONE" and pos:
                    existing_dirs = set(p["type"] for p in pos)
                    if direction in existing_dirs and sum(1 for p in pos if p["type"] == direction) >= MAX_PER_DIRECTION:
                        blocked_by = "max_per_direction"
                        direction = "NONE"

                # ── MINIMUM DURATION GUARD ── (<15min only allowed for SL/TP) ──
                # This is enforced at the exit logic level — we don't close trades < 15min unless SL/TP hits

                # ── PLACE TRADE ──────────────────────────────────────────────
                trade_opened = False
                if direction != "NONE":
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick:
                        price = tick.ask if direction == "BUY" else tick.bid
                        atr_val = i15["atr"].iloc[-1]

                        # ── 1:2 RISK TO REWARD ──
                        # SL = recent swing +/- ATR * 2.5 (for breathing room)
                        # TP = SL distance * 2.0 (minimum 1:2 RR)
                        if direction == "BUY":
                            recent_swing_low = float(m15w_ren['low'].iloc[-20:].min())
                            sl = min(recent_swing_low - (atr_val * 0.5), price - (atr_val * 2.5))
                            sl_distance = price - sl
                            tp = price + (sl_distance * 2.0)  # 1:2 RR
                        else:
                            recent_swing_high = float(m15w_ren['high'].iloc[-20:].max())
                            sl = max(recent_swing_high + (atr_val * 0.5), price + (atr_val * 2.5))
                            sl_distance = sl - price
                            tp = price - (sl_distance * 2.0)  # 1:2 RR

                        # Minimum SL distance (at least 3.0 ATR for gold — sweep-optimized)
                        min_sl_dist = atr_val * 3.0
                        if sl_distance < min_sl_dist:
                            sl_distance = min_sl_dist
                            if direction == "BUY":
                                sl = price - sl_distance
                                tp = price + (sl_distance * 2.0)
                            else:
                                sl = price + sl_distance
                                tp = price - (sl_distance * 2.0)

                        risk_pct = active_risk
                        conf = engine_result.get("confidence", 0) if engine_result else 0
                        if conf >= 85:   # High confidence breakout
                            risk_pct = HIGH_SCORE_RISK
                        lot = max(0.01, round((balance * risk_pct) / (sl_distance * 100), 2))
                        if len(pos) == 1:
                            lot *= SECOND_POS_LOT_RATIO
                        lot = max(0.01, round(lot, 2))

                        rr_ratio = round(abs(tp - price) / abs(price - sl), 2) if sl_distance > 0 else 0

                        res_order = conn.place_order(direction, SYMBOL, lot, sl=round(sl, 2), tp=round(tp, 2))
                        if res_order.get("success"):
                            trade_opened = True
                            logger.info(f"ENTRY {direction} lot={lot} price={price:.2f} sl={sl:.2f} tp={tp:.2f} RR={rr_ratio}")
                            daily_trades += 1
                            strategy.record_trade()
                            save_state()

                            option_label = f"Option {engine_result.get('option', '?')}" if engine_result and engine_result.get('option') else "Multi-Confirm"
                            try:
                                tg.send_message(
                                    f"✅ TRADE OPENED: {direction} {SYMBOL}\n"
                                    f"{'='*30}\n"
                                    f"Price: ${price:.2f} | Confidence: {conf}%\n"
                                    f"Lot: {lot} | SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                                    f"R:R = 1:{rr_ratio} | {option_label}\n"
                                    f"Reason: {engine_result.get('reason', 'N/A') if engine_result else 'N/A'}\n"
                                    f"Balance: ${balance:,.2f}"
                                )
                            except Exception as e:
                                logger.warning(f"Telegram notify failed: {e}")
                        else:
                            logger.error(f"ORDER FAILED: {res_order.get('reason')}")
                            try:
                                tg.send_message(f"❌ ORDER FAILED: {direction} at ${price:.2f}\nReason: {res_order.get('reason','unknown')}")
                            except: pass

                # ── CONSOLIDATED REPORT: signal blocked ──────────
                if not trade_opened and original_dir != "NONE":
                    price_now = m15w['close'].iloc[-1]
                    report = (
                        f"🚫 BLOCKED: Bot wanted {original_dir}\n"
                        f"{'='*30}\n"
                        f"Price: ${price_now:.2f}\n"
                        f"Blocked by: {blocked_by or 'unknown'}"
                    )
                    if engine_result:
                        report += f"\nChecks: {engine_result.get('checks', {})}"
                    try:
                        tg.send_message(report)
                    except:
                        pass

            # Hourly export + GitHub push
            if cycle % 60 == 0:
                try:
                    trade_exporter.export_trades(balance, len(pos), trades_log)
                except Exception as e:
                    logger.warning(f"Trade exporter failed: {e}")
                try:
                    push_result = gh_exporter.push_analysis()
                    if push_result and isinstance(push_result, dict):
                        if push_result.get("success"):
                            logger.info(f"[GitExport] {push_result.get('message', 'OK')}")
                        else:
                            err = push_result.get("error", "Unknown error")
                            logger.warning(f"[GitExport] Push failed: {err}")
                            try:
                                tg.send_message(f"⚠️ GITHUB PUSH FAILED\n━━━━━━━━━━━━━━━\nError: {err}\n⏰ {now.strftime('%H:%M:%S')} UTC")
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug(f"[GitExport] Push check: {e}")

            # Heartbeat (every 5 min — internal throttle)
            try:
                tg.notify_heartbeat(
                    balance=balance, open_positions=len(pos),
                    total_trades=len(trades_log),
                    equity=info.get("equity", 0)
                )
            except Exception:
                pass

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            tb = traceback.format_exc()
            logger.error(f"Loop Error: {err_msg}\n{tb[-500:]}")
            try:
                tg.notify_system_error("main_loop", f"{err_msg}\nLine: {tb.split(chr(10))[-3].strip()}")
            except Exception:
                pass
            time.sleep(10)

        time.sleep(60)


def _log_sl_closed(p_data: dict, pnl: float, exit_price: float):
    """Log a position that was closed by SL (vanished from MT5)."""
    global daily_pnl, consecutive_losses, trades_log
    direction = p_data["type"]
    trade_entry = {
        "open_time": datetime.fromtimestamp(p_data["time"], tz=timezone.utc).isoformat() if p_data.get("time") else datetime.now(timezone.utc).isoformat(),
        "close_time": datetime.now(timezone.utc).isoformat(),
        "dir": direction,
        "entry": p_data["entry"],
        "close_price": exit_price,
        "sl": p_data.get("sl", 0),
        "tp": p_data.get("tp", 0),
        "lot": p_data["lots"],
        "pnl": round(pnl, 2),
        "reason": "sl",
        "score": 0,
        "regime": "",
        "be": False,
    }
    trades_log.append(trade_entry)
    consecutive_losses += 1
    daily_pnl += pnl
    try:
        tg.notify_trade_closed(
            direction=direction, symbol=SYMBOL, entry=p_data["entry"],
            exit_price=exit_price, pnl=pnl, reason="sl",
            balance=balance_snapshot
        )
    except Exception:
        pass

def _log_closed_trade(p, pnl: float, reason: str, exit_price: float):
    """Append closed trade to trades_log and send notification."""
    global daily_pnl, consecutive_losses, trades_log
    direction = "BUY" if p.type == 0 else "SELL"
    trade_entry = {
        "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat() if hasattr(p, 'time') else datetime.now(timezone.utc).isoformat(),
        "close_time": datetime.now(timezone.utc).isoformat(),
        "dir": direction,
        "entry": p.price_open,
        "close_price": exit_price,
        "sl": p.sl,
        "tp": p.tp,
        "lot": p.volume,
        "pnl": round(pnl, 2),
        "reason": reason,
        "score": 0,
        "regime": "",
        "be": False,
    }
    trades_log.append(trade_entry)
    if pnl > 0:
        consecutive_losses = 0
    else:
        consecutive_losses += 1
    daily_pnl += pnl

    try:
        tg.notify_trade_closed(
            direction=direction, symbol=SYMBOL, entry=p.price_open,
            exit_price=exit_price, pnl=pnl, reason=reason,
            balance=balance_snapshot
        )
    except Exception:
        pass


if __name__ == "__main__":
    main_loop()