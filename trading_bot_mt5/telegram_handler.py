"""
Telegram Chat Handler — DeepSeek V4 Flash Chat Bot
====================================================
Poll for incoming Telegram messages, forward them to DeepSeek AI,
handle command execution (change settings, fix errors, edit files), and reply.

Permissions: Only the authorized CHAT_ID can issue commands.
"""

import requests
import json
import time
import os
import sys
import threading
import traceback
import subprocess
from datetime import datetime, timezone
from logger_mt5 import logger

# Reuse same token/chat_id from telegram_notifier
from telegram_notifier import TOKEN, CHAT_ID, send_message

# DeepSeek client (lazy import to avoid circular)
_deepseek_client = None
_config_module = None

# In-memory conversation history
_conversation_history = []
_MAX_HISTORY = 20

# Polling
_polling_active = False
_last_update_id = 0
_POLL_INTERVAL = 3

# Track last restart to prevent restart loops
_last_restart_time = 0
_MIN_RESTART_INTERVAL = 300  # 5 min between restarts

# Project root directory
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else '.')))


def set_deepseek_client(client):
    """Inject the DeepSeekFilter instance so we can use it for chat."""
    global _deepseek_client
    _deepseek_client = client


def get_pending_updates():
    """Fetch pending messages from Telegram."""
    global _last_update_id
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        params = {"offset": _last_update_id + 1, "timeout": 5}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("ok") and data.get("result"):
            return data["result"]
    except Exception as e:
        logger.debug(f"[TelegramPoll] Poll error: {e}")
    return []


def process_update(update):
    """Process a single incoming Telegram message."""
    global _last_update_id, _conversation_history
    _last_update_id = update.get("update_id", 0)

    if "message" not in update:
        return
    msg = update["message"]
    chat_id = msg.get("chat", {}).get("id")
    text = msg.get("text", "")

    if chat_id != CHAT_ID:
        logger.info(f"[TelegramPoll] Ignored message from chat {chat_id}")
        return

    if not text or text.startswith("/"):
        return

    logger.info(f"[TelegramPoll] Message: {text[:80]}")

    _conversation_history.append({"role": "user", "content": text})
    if len(_conversation_history) > _MAX_HISTORY:
        _conversation_history = _conversation_history[-_MAX_HISTORY:]

    threading.Thread(target=_handle_message, args=(text,), daemon=True).start()


def _handle_message(user_text: str):
    """Send user message to DeepSeek and reply via Telegram."""
    global _conversation_history, _config_module
    try:
        if _deepseek_client is None or _deepseek_client.client is None:
            send_message("🤖 AI Assistant not available (no DeepSeek API key).")
            return

        # Gather live bot state data for the AI to use
        bot_state = _get_bot_state_data()
        
        system_prompt = f"""You are DeepSeek V4 Flash, the AI brain of a live XAUUSD Gold Scalping Bot running on MT5.

## LIVE BOT STATE (refreshed every poll):
{json.dumps(bot_state, indent=2, default=str)}

You have the following CAPABILITIES:

## 1. CHAT MODE - FULL DATA ACCESS
You have FULL access to the bot's live data above. Answer questions about:
- Current balance, equity, PnL, win rate
- Open positions and trade history
- Blocked trades and why they were blocked
- Recent AI market analyses
- Performance metrics and DD status
- Give improvement suggestions based on REAL data, not guesses

## 2. SETTINGS CHANGES
When user asks to change a setting, respond with a JSON block at the END:
```config
{{"SETTING_NAME": new_value}}
```
Available: MIN_SCORE_BUY, MIN_SCORE_SELL, MAX_POSITIONS, DAILY_LOSS_PCT, FIXED_RISK, HIGH_SCORE_THRESHOLD, HIGH_SCORE_RISK, MAX_CONSEC_LOSSES, HALT_HOURS, TP_ATR_MULT, SL_ATR_MULT, TRADE_HOURS_START, TRADE_HOURS_END, MAX_SPREAD, SESSION_COOLDOWN_MIN, ATR_VOL_THRESHOLD, USE_AI_FILTER, AI_MARKET_ANALYSIS

## 3. EDIT BOT FILES
When user asks to edit a bot file, respond with:
```file
{{"path": "trading_bot_mt5/main_super.py", "search": "EXACT LINE TO FIND", "replace": "NEW LINE CONTENT"}}
```

## 4. RESTART BOT
When user asks to restart or the bot needs restart after a fix, include:
```restart
{{"action": "restart"}}
```

Bot: XAUUSD Gold Scalping v8, MT5 local, DeepSeek AI active.
IMPORTANT: When user asks for status, ALWAYS reference the LIVE data provided. Never say "I don't have access."
"""

        messages = [{"role": "system", "content": system_prompt}]
        for h in _conversation_history[-10:]:
            messages.append(h)

        response = _deepseek_client.client.chat.completions.create(
            model=_deepseek_client.model,
            messages=messages,
            max_tokens=800,
            timeout=20
        )

        reply = response.choices[0].message.content

        # Check for config changes
        config_changes = _extract_config_json(reply)
        if config_changes:
            try:
                result = _apply_config_changes(config_changes)
                reply += f"\n\n✅ Settings applied: {result}"
                _save_config_overrides(config_changes)
            except Exception as e:
                reply += f"\n\n❌ Failed: {str(e)[:100]}"

        # Check for file edit commands
        file_edit = _extract_file_edit(reply)
        if file_edit:
            try:
                result = _apply_file_edit(file_edit)
                reply += f"\n\n📝 File edit: {result}"
            except Exception as e:
                reply += f"\n\n❌ File edit failed: {str(e)[:100]}"

        # Check for file write commands
        file_write = _extract_file_write(reply)
        if file_write:
            try:
                result = _apply_file_write(file_write)
                reply += f"\n\n📄 File write: {result}"
            except Exception as e:
                reply += f"\n\n❌ File write failed: {str(e)[:100]}"

        # Check for restart command
        if '```restart' in reply:
            _handle_restart()

        _conversation_history.append({"role": "assistant", "content": reply})
        if len(_conversation_history) > _MAX_HISTORY:
            _conversation_history = _conversation_history[-_MAX_HISTORY:]

        if len(reply) > 4000:
            for i in range(0, len(reply), 3500):
                send_message(reply[i:i+3500])
        else:
            send_message(reply)

        _log_conversation(user_text, reply)

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:100]}"
        logger.error(f"[TelegramAI] Chat failed: {err}")
        send_message(f"⚠️ AI Error: {err}")


def _extract_config_json(text: str) -> dict:
    """Extract JSON config block from AI response."""
    import re
    patterns = [
        r'```config\s*\n?(.*?)\n?```',
        r'```json\s*\n?(.*?)\n?```',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, IndexError):
                continue
    return None


def _extract_file_edit(text: str) -> dict:
    """Extract file edit block: ```file {...} ```"""
    import re
    match = re.search(r'```file\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None


def _extract_file_write(text: str) -> dict:
    """Extract file write block: ```file_write {...} ```"""
    import re
    match = re.search(r'```file_write\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None


def _apply_file_edit(edit: dict) -> str:
    """Apply a file edit: replace 'search' with 'replace' in 'path'."""
    path = edit.get("path", "")
    search = edit.get("search", "")
    replace = edit.get("replace", "")
    if not path or not search:
        return "Missing path or search"
    
    # Resolve path relative to project
    full_path = os.path.join(_PROJECT_DIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        return f"File not found: {path}"
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if search not in content:
        return f"Search text not found in {path}"
    
    content = content.replace(search, replace, 1)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"[FileEdit] Modified {path}")
    return f"✅ Edited {path}"


def _apply_file_write(file_write: dict) -> str:
    """Write full content to a file."""
    path = file_write.get("path", "")
    content = file_write.get("content", "")
    if not path or not content:
        return "Missing path or content"
    
    full_path = os.path.join(_PROJECT_DIR, path) if not os.path.isabs(path) else path
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"[FileWrite] Wrote {path} ({len(content)} bytes)")
    return f"✅ Wrote {path} ({len(content)} bytes)"


def _handle_restart():
    """Restart the bot process."""
    global _last_restart_time
    now = time.time()
    if now - _last_restart_time < _MIN_RESTART_INTERVAL:
        send_message("⏳ Please wait before restarting (5 min cooldown)")
        return
    
    _last_restart_time = now
    send_message("🔄 Restarting bot...")
    logger.info("[Restart] Bot restart requested via Telegram")
    
    try:
        # Spawn new process
        python = sys.executable
        script = os.path.join(os.path.dirname(__file__), "main_super.py")
        subprocess.Popen([python, script], cwd=os.path.dirname(script))
        # Exit current process
        os._exit(0)
    except Exception as e:
        send_message(f"❌ Restart failed: {str(e)[:100]}")


def _apply_config_changes(changes: dict) -> str:
    """Apply config changes to the running bot's globals."""
    global _config_module
    if _config_module is None:
        import main_super as ms
        _config_module = ms

    applied = []
    for key, value in changes.items():
        key_upper = key.upper()
        mod = _config_module
        if hasattr(mod, key_upper):
            old_val = getattr(mod, key_upper)
            setattr(mod, key_upper, type(old_val)(value))
            applied.append(f"{key_upper}: {old_val} → {value}")
        elif hasattr(mod, key):
            old_val = getattr(mod, key)
            setattr(mod, key, type(old_val)(value))
            applied.append(f"{key}: {old_val} → {value}")
        else:
            applied.append(f"{key}: NOT FOUND")
    return "; ".join(applied) if applied else "No valid settings"


def _save_config_overrides(changes: dict):
    """Persist config changes to a JSON file so they survive bot restart."""
    try:
        config_file = os.path.join(os.path.dirname(__file__), "config_overrides.json")
        overrides = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                overrides = json.load(f)
        overrides.update(changes)
        with open(config_file, "w") as f:
            json.dump(overrides, f, indent=2)
        logger.info(f"[ConfigOverride] Saved: {changes}")
    except Exception as e:
        logger.warning(f"[ConfigOverride] Save failed: {e}")


def _log_conversation(user_msg: str, ai_reply: str):
    """Log DeepSeek conversation to a file for GitHub upload."""
    try:
        log_file = os.path.join(os.path.dirname(__file__), "deepseek_chat_log.jsonl")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_msg,
            "assistant": ai_reply,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"[ChatLog] Write failed: {e}")


def _get_bot_state_data() -> dict:
    """Gather live bot state for AI context. Reads from main_super globals."""
    try:
        import main_super as ms
    except:
        return {"error": "Cannot import main_super"}
    
    try:
        # Trades summary
        trades_list = getattr(ms, 'trades_log', [])
        wins = [t for t in trades_list if t.get('pnl', 0) > 0] if trades_list else []
        losses = [t for t in trades_list if t.get('pnl', 0) <= 0] if trades_list else []
        total_trades = len(trades_list)
        win_rate = round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0
        total_pnl = round(sum(t.get('pnl', 0) for t in trades_list), 2) if trades_list else 0
        recent_trades = trades_list[-10:] if trades_list else []
        
        # Read latest AI analysis
        latest_analyses = []
        try:
            log_file = os.path.join(os.path.dirname(__file__), "deepseek_analysis_log.jsonl")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                for line in lines[-5:]:
                    try:
                        latest_analyses.append(json.loads(line))
                    except:
                        pass
        except:
            pass
        
        # Read performance state
        perf_data = {}
        try:
            perf_file = os.path.join(os.path.dirname(__file__), "performance_state.json")
            if os.path.exists(perf_file):
                with open(perf_file) as f:
                    perf_data = json.load(f)
        except:
            pass
        
        return {
            "account": {
                "balance": getattr(ms, 'balance_snapshot', 0),
                "daily_pnl": round(getattr(ms, 'daily_pnl', 0), 2),
                "open_positions_count": 0,  # filled below if available
                "consecutive_losses": getattr(ms, 'consecutive_losses', 0),
                "daily_halted": getattr(ms, 'daily_halted', False) if hasattr(ms, 'daily_halted') else False,
            },
            "performance": {
                "total_trades": total_trades,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": win_rate,
                "total_pnl": total_pnl,
                "avg_win": round(sum(t.get('pnl',0) for t in wins)/len(wins), 2) if wins else 0,
                "avg_loss": round(sum(t.get('pnl',0) for t in losses)/len(losses), 2) if losses else 0,
                "peak_balance": perf_data.get("peak_balance", 0),
                "peak_equity": perf_data.get("peak_equity", 0),
                "dd_halted": getattr(ms, 'perf', None) and getattr(getattr(ms, 'perf', None), 'dd_halted', False) if hasattr(ms, 'perf') else False,
            },
            "recent_trades": [{
                "dir": t.get("dir","?"),
                "entry": t.get("entry",0),
                "pnl": t.get("pnl",0),
                "reason": t.get("reason","?"),
                "time": t.get("close_time","?")[:19] if t.get("close_time") else "?",
            } for t in recent_trades],
            "latest_ai_analyses": [{
                "time": a.get("timestamp","?")[:19],
                "price": a.get("price",0),
                "bias": a.get("bias","?"),
                "risk": a.get("risk","?"),
                "confidence": a.get("confidence",0),
            } for a in latest_analyses],
            "config": {
                "MIN_SCORE_BUY": getattr(ms, 'MIN_SCORE_BUY', 20),
                "MIN_SCORE_SELL": getattr(ms, 'MIN_SCORE_SELL', 20),
                "MAX_POSITIONS": getattr(ms, 'MAX_POSITIONS', 2),
                "FIXED_RISK": getattr(ms, 'FIXED_RISK', 0.02),
                "TRADE_HOURS": f"{getattr(ms, 'TRADE_HOURS_START', 8)}-{getattr(ms, 'TRADE_HOURS_END', 17)} UTC",
                "SESSION": getattr(ms, 'get_session', lambda: "?")() if callable(getattr(ms, 'get_session', None)) else "?",
                "USE_AI_FILTER": getattr(ms, 'USE_AI_FILTER', False),
                "AI_MARKET_ANALYSIS": getattr(ms, 'AI_MARKET_ANALYSIS', False),
            },
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def start_polling():
    """Start polling for Telegram messages in a background thread."""
    global _polling_active
    if _polling_active:
        return
    _polling_active = True
    t = threading.Thread(target=_polling_loop, daemon=True, name="TelegramPoll")
    t.start()
    logger.info("[TelegramPoll] Started polling for chat messages")


def stop_polling():
    """Stop polling."""
    global _polling_active
    _polling_active = False


def _polling_loop():
    """Background loop polling Telegram for new messages."""
    global _polling_active
    while _polling_active:
        try:
            updates = get_pending_updates()
            for update in updates:
                process_update(update)
        except Exception as e:
            logger.debug(f"[TelegramPoll] Loop error: {e}")
        time.sleep(_POLL_INTERVAL)