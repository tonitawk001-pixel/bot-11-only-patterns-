"""
DEEPSEEK AI FILTER
==================
Uses DeepSeek V4 Flash to provide a final 'Go/No-Go' decision on trade signals.
Analyzes market context, technical indicators, and news sentiment.
"""

from openai import OpenAI
from logger_mt5 import logger
import json

class DeepSeekFilter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key or api_key.startswith("sk-"):
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = None
            logger.warning("[DEEPSEEK] No valid API key provided — AI filter disabled")
        self.model = "deepseek-chat"

    def analyze_signal(self, direction, price, context):
        """
        Send technical context to DeepSeek for analysis.
        Returns (bool: allowed, str: reason).
        """
        if self.client is None:
            logger.info("[DEEPSEEK] Client not available — skipping AI filter")
            return True, "AI disabled (no valid key)"

        prompt = f"""
        ACT AS AN EXPERT GOLD TRADER. 
        Your task is to validate a technical trading signal for XAUUSD.
        ALSO verify that the indicators and data feeding the signal are healthy.
        
        SIGNAL: {direction} @ {price}
        
        INDICATOR HEALTH CHECK (verify values are valid):
        - RSI (M15): {context.get('rsi')} — should be 0-100. If NaN, 0, or >100: INDICATOR BROKEN
        - ATR: {context.get('atr_status')} — should be >$0 and <$50 for gold
        - MACD: {context.get('macd_status','?')} — should not be 0 if market is moving
        - Price: ${context.get('price', price)} — should match current XAUUSD market (~$2500-4500)
        - Candle Patterns Detected: {context.get('candle_patterns','none')}
        - If ANY indicator is NaN, 0 for extended period, or wildly off: REJECT with warning
        
        MARKET CONTEXT:
        - M15 Bias: {context.get('m15_bias')}
        - H4 Trend: {context.get('h4_trend')}
        - Support/Resistance: {context.get('sr_levels')}
        - Setup Score: {context.get('score')}
        - Strategy Reason: {context.get('reason')}
        - Open Positions: {context.get('open_positions')}
        - Daily PnL: {context.get('daily_pnl')}
        - Upcoming News: {context.get('news')}

        NEWS-DRIVEN DECISION RULES:
        6. If a recent major event (FOMC, NFP, CPI) was hawkish → FAVOR SELL, REJECT BUY.
        7. If a recent major event was dovish → FAVOR BUY, REJECT SELL.
        8. If a major event is within 90 minutes → REDUCE confidence, only accept high-score signals (>60).
        9. If no major news today → use technical analysis only.
        
        RULES:
        1. Reject BUY if Distance to Resistance is less than $2.50.
        2. Reject SELL if Distance to Support is less than $2.50.
        3. Reject if high-impact news is within 60 mins.
        4. Reject if the signal is counter to the H4 trend unless extreme RSI exhaustion (>80 or <20).
        5. REJECT if ANY indicator value is suspicious (NaN, 0 for RSI, negative ATR).
        
        OUTPUT FORMAT (Strict JSON only):
        {{
            "decision": "GO" or "NO_GO",
            "confidence": 0-100,
            "reason": "short explanation. Include WARNING: if indicators seem off"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Fast Gold Scalping Filter. Respond strictly in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                timeout=5
            )
            
            result = json.loads(response.choices[0].message.content)
            decision = result.get("decision")
            reason = result.get("reason", "No reason provided")
            confidence = result.get("confidence", 0)
            
            logger.info(f"[DEEPSEEK] Decision: {decision} | Conf: {confidence}% | Reason: {reason}")
            
            return decision == "GO", reason
            
        except Exception as e:
            logger.error(f"[DEEPSEEK] Analysis failed: {e}")
            # SAFETY: AI failure means we BLOCK the trade (fail-closed)
            # Only allow if the error was a simple timeout and score is exceptional
            return False, f"AI Fallback: Error - {str(e)[:80]}"


    def analyze_market(self, context: dict) -> dict:
        """
        Send full market picture to DeepSeek for comprehensive analysis.
        Returns dict with bias, risk_level, notes.
        Called every 15-min bar close.
        """
        if self.client is None:
            return {"bias": "unknown", "risk": "unknown", "notes": "AI disabled"}

        prompt = f"""
        ACT AS AN EXPERT GOLD TRADER ANALYZING THE MARKET.
        Give a quick professional read of current XAUUSD conditions.

        CURRENT STATE:
        - Price: ${context.get('price', '?')}
        - Session: {context.get('session', '?')}
        - M15 Bias: {context.get('m15_bias', '?')}
        - H4 Trend: {context.get('h4_trend', '?')}
        - RSI (M15): {context.get('rsi', '?')}
        - MACD: {context.get('macd_status', '?')}
        - ATR: {context.get('atr_status', '?')}
        - Nearest Support: ${context.get('nearest_support', '?')}
        - Nearest Resistance: ${context.get('nearest_resistance', '?')}
        - Candle Patterns: {context.get('candle_patterns', 'none')}
        - S/R Touch: {context.get('sr_touch_info', 'none')}
        - Upcoming News: {context.get('news', 'none')}

        OUTPUT FORMAT (Strict JSON):
        {{
            "bias": "bullish" or "bearish" or "neutral",
            "risk_level": "low" or "medium" or "high",
            "confidence": 0-100,
            "notes": "One line professional market summary (max 150 chars)"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Expert gold analyst. Respond strictly in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                timeout=8
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"[DEEPSEEK-MARKET] Bias: {result.get('bias')} | Risk: {result.get('risk_level')} | Conf: {result.get('confidence')}")
            return result
        except Exception as e:
            logger.error(f"[DEEPSEEK-MARKET] Analysis failed: {e}")
            return {"bias": "unknown", "risk": "unknown", "notes": f"Error: {str(e)[:80]}"}

    def system_health_check(self, system_state: dict) -> dict:
        """
        DeepSeek verifies all systems are operational at startup.
        Returns {"verdict": "OK"|"WARNING"|"ERROR", "health_score": 0-100,
                 "issues": [], "report": "summary"}
        """
        if self.client is None:
            return {"verdict": "WARNING", "health_score": 80,
                    "issues": ["AI client not initialized"], "report": "AI filter disabled - no valid key"}

        prompt = f"""
        ACT AS A SYSTEM AUDITOR FOR A GOLD TRADING BOT.
        Verify all systems and tools are operational. Report any anomalies.

        SYSTEM STATE:
        - MT5 Connection: {system_state.get('mt5_connected','?')}
        - Account: {system_state.get('login','?')} | Server: {system_state.get('server','?')}
        - Balance: ${system_state.get('balance',0):.2f} | Equity: ${system_state.get('equity',0):.2f}
        - Spread: ${system_state.get('spread','?')} | Session: {system_state.get('session','?')}
        - In Trading Hours: {system_state.get('in_hours','?')}
        - News Events Loaded: {system_state.get('news_count',0)}
        - Candle Patterns: Loaded (9 patterns + S/R engine)
        - Strategy: Active (RSI/MACD/EMA/ATR/BB)
        - All 17 Tools: Should be active

        VALIDATION RULES:
        - Balance must be > $50 (bot floor) — if below, CRITICAL ERROR
        - Spread > $2.00 on gold = WARNING (wide market)
        - If outside trading hours (8-17 UTC), note it
        - 0 news events loaded = WARNING (API might be down)
        - Missing tools = WARNING

        OUTPUT JSON:
        {{
            "verdict": "OK" or "WARNING" or "ERROR",
            "health_score": 0-100,
            "issues": ["list of problems found, empty if none"],
            "report": "2-3 sentence professional summary of system state"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "Expert system auditor. Respond strictly in JSON."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200, timeout=8
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"[DEEPSEEK-HEALTH] Verdict: {result.get('verdict')} | Score: {result.get('health_score')}")
            return result
        except Exception as e:
            logger.error(f"[DEEPSEEK-HEALTH] Check failed: {e}")
            return {"verdict": "WARNING", "health_score": 70,
                    "issues": [f"Health check failed: {str(e)[:80]}"],
                    "report": "Could not complete AI health check. Bot may still trade normally."}

