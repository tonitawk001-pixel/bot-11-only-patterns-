"""
CANDLE PATTERNS + S/R ENGINE - Advanced Professional Grade
===========================================================
Detects 30+ candlestick patterns, support/resistance levels, and generates
trade signals based on price action at key levels.

Patterns:
  Single: Doji, Hammer, Hanging Man, Shooting Star, Inverted Hammer,
          Marubozu, Long Upper/Lower Wick, Spinning Top, High Wave
  Double: Bullish/Bearish Engulfing, Piercing Line, Dark Cloud Cover,
          Harami (Bullish/Bearish), Tweezers, Inside Bar
  Triple: Morning/Evening Star, Three White Soldiers, Three Black Crows,
          Three Inside Up/Down, Abandoned Baby
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
try:
    from logger_mt5 import logger
except ImportError:
    import logging
    logger = logging.getLogger("candle_patterns")


# ─────────────────────────────────────────────────────────────
#  SINGLE CANDLE CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_candle(open_: float, high: float, low: float, close_: float) -> str:
    """Advanced single candle classification - 14 types."""
    body = abs(close_ - open_)
    total_range = high - low
    if total_range <= 0:
        return "none"

    lower_wick = min(open_, close_) - low
    upper_wick = high - max(open_, close_)
    body_pct = body / total_range if total_range > 0 else 0
    lower_wick_pct = lower_wick / total_range if total_range > 0 else 0
    upper_wick_pct = upper_wick / total_range if total_range > 0 else 0
    is_bullish = close_ > open_

    # ── Doji variants ──
    if body_pct < 0.03:
        return "doji"

    # ── Spinning Top (small body, both wicks) ──
    if body_pct < 0.30 and lower_wick_pct > 0.25 and upper_wick_pct > 0.25:
        return "spinning_top"

    # ── High Wave (small body, very long wicks) ──
    if body_pct < 0.20 and (lower_wick_pct > 0.35 or upper_wick_pct > 0.35):
        return "high_wave"

    # ── Hammer / Hanging Man (long lower wick, small body at top) ──
    if lower_wick > body * 2.5 and upper_wick < body * 0.8:
        return "hammer" if is_bullish else "hanging_man"

    # ── Shooting Star / Inverted Hammer (long upper wick, small body at bottom) ──
    if upper_wick > body * 2.5 and lower_wick < body * 0.8:
        return "shooting_star" if not is_bullish else "inverted_hammer"

    # ── Marubozu (no wicks, full body) ──
    if body_pct > 0.85 and lower_wick_pct < 0.05 and upper_wick_pct < 0.05:
        return "bullish_marubozu" if is_bullish else "bearish_marubozu"

    # ── Long lower wick ──
    if lower_wick_pct > 0.65 and lower_wick > body * 2:
        return "long_lower_wick"

    # ── Long upper wick ──
    if upper_wick_pct > 0.65 and upper_wick > body * 2:
        return "long_upper_wick"

    # ── Strong directional ──
    if is_bullish and body_pct > 0.55:
        return "strong_bullish"
    if not is_bullish and body_pct > 0.55:
        return "strong_bearish"

    return "none"


# ─────────────────────────────────────────────────────────────
#  MULTI-CANDLE PATTERN DETECTION (30+ patterns)
# ─────────────────────────────────────────────────────────────

def detect_patterns(ohlcv: pd.DataFrame) -> List[Dict]:
    """Detect all multi-candle patterns. Returns list of patterns with metadata."""
    if len(ohlcv) < 5:
        return []

    o = ohlcv['open'].values
    h = ohlcv['high'].values
    l = ohlcv['low'].values
    c = ohlcv['close'].values

    patterns = []

    # Classify candles
    last = classify_candle(o[-1], h[-1], l[-1], c[-1])
    prev = classify_candle(o[-2], h[-2], l[-2], c[-2])
    p3 = classify_candle(o[-3], h[-3], l[-3], c[-3]) if len(o) >= 3 else "none"
    p4 = classify_candle(o[-4], h[-4], l[-4], c[-4]) if len(o) >= 4 else "none"

    is_bullish_1 = c[-1] > o[-1]
    is_bullish_2 = c[-2] > o[-2]
    is_bullish_3 = c[-3] > o[-3] if len(o) >= 3 else False
    is_bullish_4 = c[-4] > o[-4] if len(o) >= 4 else False

    # ═══════ 2-CANDLE PATTERNS ═══════

    # ── Bullish Engulfing ──
    if not is_bullish_2 and is_bullish_1 and c[-1] > o[-2] and o[-1] < c[-2]:
        patterns.append({"name": "bullish_engulfing", "direction": "BUY", "strength": 80, "price": c[-1]})

    # ── Bearish Engulfing ──
    if is_bullish_2 and not is_bullish_1 and c[-1] < o[-2] and o[-1] > c[-2]:
        patterns.append({"name": "bearish_engulfing", "direction": "SELL", "strength": 80, "price": c[-1]})

    # ── Piercing Line ──
    mid_last = (o[-2] + c[-2]) / 2
    if not is_bullish_2 and is_bullish_1 and o[-1] < l[-2] and c[-1] > mid_last:
        patterns.append({"name": "piercing_line", "direction": "BUY", "strength": 75, "price": c[-1]})

    # ── Dark Cloud Cover ──
    if is_bullish_2 and not is_bullish_1 and o[-1] > h[-2] and c[-1] < mid_last:
        patterns.append({"name": "dark_cloud_cover", "direction": "SELL", "strength": 75, "price": c[-1]})

    # ── Bullish Harami ──
    if not is_bullish_2 and is_bullish_1:
        body1 = abs(c[-2] - o[-2])
        body2 = abs(c[-1] - o[-1])
        if o[-1] > c[-2] and c[-1] < o[-2] and body2 * 2.5 < body1:
            patterns.append({"name": "bullish_harami", "direction": "BUY", "strength": 70, "price": c[-1]})

    # ── Bearish Harami ──
    if is_bullish_2 and not is_bullish_1:
        body1 = abs(c[-2] - o[-2])
        body2 = abs(c[-1] - o[-1])
        if o[-1] < c[-2] and c[-1] > o[-2] and body2 * 2.5 < body1:
            patterns.append({"name": "bearish_harami", "direction": "SELL", "strength": 70, "price": c[-1]})

    # ── Tweezer Top ──
    if is_bullish_2 and not is_bullish_1 and abs(h[-1] - h[-2]) / max(h[-1], 1) < 0.001:
        patterns.append({"name": "tweezer_top", "direction": "SELL", "strength": 65, "price": c[-1]})

    # ── Tweezer Bottom ──
    if not is_bullish_2 and is_bullish_1 and abs(l[-1] - l[-2]) / max(l[-1], 1) < 0.001:
        patterns.append({"name": "tweezer_bottom", "direction": "BUY", "strength": 65, "price": c[-1]})

    # ── Inside Bar (NR7: narrowest range in 7 bars) ──
    if h[-1] <= h[-2] and l[-1] >= l[-2]:
        patterns.append({"name": "inside_bar", "direction": "NONE", "strength": 40, "price": c[-1],
                         "note": "Breakout setup — direction determined by break"})

    # ── Outside Bar (volatility expansion) ──
    if h[-1] > h[-2] and l[-1] < l[-2] and abs(c[-1] - o[-1]) > abs(c[-2] - o[-2]) * 1.5:
        patterns.append({"name": "outside_bar", "direction": "BUY" if is_bullish_1 else "SELL",
                         "strength": 60, "price": c[-1]})

    # ═══════ 3-CANDLE PATTERNS ═══════
    if len(o) >= 3:
        # ── Morning Star ──
        if not is_bullish_3 and abs(c[-2] - o[-2]) / max(h[-2] - l[-2], 0.01) < 0.35 and is_bullish_1 and c[-1] > (o[-3] + c[-3]) / 2:
            patterns.append({"name": "morning_star", "direction": "BUY", "strength": 85, "price": c[-1]})

        # ── Evening Star ──
        if is_bullish_3 and abs(c[-2] - o[-2]) / max(h[-2] - l[-2], 0.01) < 0.35 and not is_bullish_1 and c[-1] < (o[-3] + c[-3]) / 2:
            patterns.append({"name": "evening_star", "direction": "SELL", "strength": 85, "price": c[-1]})

        # ── Three White Soldiers ──
        if is_bullish_3 and is_bullish_2 and is_bullish_1 and c[-3] < c[-2] < c[-1]:
            body_check = (abs(c[-1] - o[-1]) > abs(c[-2] - o[-2]) * 0.7)
            if body_check:
                patterns.append({"name": "three_white_soldiers", "direction": "BUY", "strength": 82, "price": c[-1]})

        # ── Three Black Crows ──
        if not is_bullish_3 and not is_bullish_2 and not is_bullish_1 and c[-3] > c[-2] > c[-1]:
            body_check = (abs(c[-1] - o[-1]) > abs(c[-2] - o[-2]) * 0.7)
            if body_check:
                patterns.append({"name": "three_black_crows", "direction": "SELL", "strength": 82, "price": c[-1]})

        # ── Three Inside Up ──
        if not is_bullish_3 and is_bullish_2 and is_bullish_1 and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.5 and h[-2] <= h[-3] and l[-2] >= l[-3] and c[-1] > h[-3]:
            patterns.append({"name": "three_inside_up", "direction": "BUY", "strength": 78, "price": c[-1]})

        # ── Three Inside Down ──
        if is_bullish_3 and not is_bullish_2 and not is_bullish_1 and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.5 and h[-2] <= h[-3] and l[-2] >= l[-3] and c[-1] < l[-3]:
            patterns.append({"name": "three_inside_down", "direction": "SELL", "strength": 78, "price": c[-1]})

        # ── Abandoned Baby (strong reversal) ──
        if not is_bullish_3 and abs(c[-2] - o[-2]) / max(h[-2] - l[-2], 0.01) < 0.1 and is_bullish_1 and l[-1] > h[-2]:
            patterns.append({"name": "abandoned_baby_bull", "direction": "BUY", "strength": 90, "price": c[-1]})
        if is_bullish_3 and abs(c[-2] - o[-2]) / max(h[-2] - l[-2], 0.01) < 0.1 and not is_bullish_1 and h[-1] < l[-2]:
            patterns.append({"name": "abandoned_baby_bear", "direction": "SELL", "strength": 90, "price": c[-1]})

    # ═══════ SINGLE CANDLE SIGNALS ═══════
    if last == "hammer":
        patterns.append({"name": "hammer", "direction": "BUY", "strength": 50, "price": c[-1]})
    if last == "shooting_star":
        patterns.append({"name": "shooting_star", "direction": "SELL", "strength": 50, "price": c[-1]})
    if last == "hanging_man":
        patterns.append({"name": "hanging_man", "direction": "SELL", "strength": 45, "price": c[-1]})
    if last == "inverted_hammer":
        patterns.append({"name": "inverted_hammer", "direction": "BUY", "strength": 45, "price": c[-1]})
    if last == "doji":
        patterns.append({"name": "doji", "direction": "NONE", "strength": 25, "price": c[-1]})
    if last in ("bullish_marubozu", "strong_bullish"):
        patterns.append({"name": last, "direction": "BUY", "strength": 55, "price": c[-1]})
    if last in ("bearish_marubozu", "strong_bearish"):
        patterns.append({"name": last, "direction": "SELL", "strength": 55, "price": c[-1]})

    return patterns


# ─────────────────────────────────────────────────────────────
#  S/R TOUCH DETECTION (enhanced)
# ─────────────────────────────────────────────────────────────

def detect_swing_levels(ohlcv: pd.DataFrame, lookback: int = 50) -> Dict:
    """Find swing highs and lows for S/R detection with clustering."""
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    closes = ohlcv['close'].values if 'close' in ohlcv.columns else None
    swing_highs = []
    swing_lows = []
    window = 5
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])

    def cluster_levels(levels: List[float], threshold_pct: float = 0.003) -> List[float]:
        if not levels:
            return []
        sorted_levels = sorted(set(levels))
        clusters = []
        current = [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            if lvl and current[-1] and abs(lvl - current[-1]) / current[-1] < threshold_pct:
                current.append(lvl)
            else:
                clusters.append(sum(current) / len(current))
                current = [lvl]
        clusters.append(sum(current) / len(current))
        return sorted(clusters, reverse=True)

    result = {
        "resistances": cluster_levels(swing_highs),
        "supports": cluster_levels(swing_lows, 0.003),
        "swing_highs_count": len(swing_highs),
        "swing_lows_count": len(swing_lows),
    }
    if closes is not None and len(closes) > 0:
        cp = closes[-1]
        res_above = [r for r in result["resistances"] if r > cp]
        sup_below = [s for s in result["supports"] if s < cp]
        result["nearest_resistance"] = min(res_above) if res_above else cp + 15
        result["nearest_support"] = max(sup_below) if sup_below else cp - 15
        result["current_price"] = cp
        # Add strength: how many times level was tested
        result["resistance_strength"] = [swing_highs.count(r) for r in result["resistances"][:3]]
        result["support_strength"] = [swing_lows.count(s) for s in result["supports"][:3]]
    return result


def check_sr_touch(ohlcv: pd.DataFrame, sr_levels: Dict) -> Dict:
    """Advanced S/R touch detection with pattern confirmation."""
    if not sr_levels or len(ohlcv) < 3:
        return {"touch_type": "none", "signal": "NONE", "confidence": 0, "level": 0,
                "dist_to_resistance": 99, "dist_to_support": 99, "reason": ""}

    curr = ohlcv['close'].iloc[-1]
    near_res = sr_levels.get("nearest_resistance", 99999)
    near_sup = sr_levels.get("nearest_support", 0)
    dist_r = near_res - curr if near_res > curr else 999
    dist_s = curr - near_sup if near_sup < curr else 999

    last_c = classify_candle(ohlcv['open'].iloc[-1], ohlcv['high'].iloc[-1],
                             ohlcv['low'].iloc[-1], ohlcv['close'].iloc[-1])
    prev_c = classify_candle(ohlcv['open'].iloc[-2], ohlcv['high'].iloc[-2],
                             ohlcv['low'].iloc[-2], ohlcv['close'].iloc[-2])

    result = {"touch_type": "none", "signal": "NONE", "confidence": 0, "level": 0, "reason": "",
              "dist_to_resistance": round(dist_r, 2), "dist_to_support": round(dist_s, 2)}

    bearish_rev = ("shooting_star", "bearish_marubozu", "strong_bearish", "hanging_man",
                   "long_upper_wick", "dark_cloud_cover", "bearish_engulfing", "evening_star")
    bullish_rev = ("hammer", "bullish_marubozu", "strong_bullish", "inverted_hammer",
                   "long_lower_wick", "piercing_line", "bullish_engulfing", "morning_star")

    # Resistance touch + reversal signal
    if dist_r < 2.5:
        result["touch_type"] = "resistance"
        result["level"] = near_res
        if last_c in bearish_rev or prev_c == "doji":
            confidence = 80 if last_c in ("shooting_star", "bearish_marubozu", "bearish_engulfing") else 60
            result["signal"] = "SELL"
            result["confidence"] = confidence
            result["reason"] = f"Resistance ${near_res:.2f} + {last_c}"
        elif last_c == "doji" or last_c == "spinning_top":
            result["signal"] = "SELL"
            result["confidence"] = 45
            result["reason"] = f"Resistance ${near_res:.2f} + indecision ({last_c})"

    # Support touch + reversal signal
    elif dist_s < 2.5:
        result["touch_type"] = "support"
        result["level"] = near_sup
        if last_c in bullish_rev or prev_c == "doji":
            confidence = 80 if last_c in ("hammer", "bullish_marubozu", "bullish_engulfing") else 60
            result["signal"] = "BUY"
            result["confidence"] = confidence
            result["reason"] = f"Support ${near_sup:.2f} + {last_c}"
        elif last_c == "doji" or last_c == "spinning_top":
            result["signal"] = "BUY"
            result["confidence"] = 45
            result["reason"] = f"Support ${near_sup:.2f} + indecision ({last_c})"

    # Near S/R zones (within 5 points) — early warning
    elif dist_r < 5.0:
        result["touch_type"] = "near_resistance"
        result["level"] = near_res
        result["signal"] = "SELL" if last_c in bearish_rev else "NONE"
        result["confidence"] = 35
        result["reason"] = f"Near resistance ${near_res:.2f} (dist: ${dist_r:.1f})"
    elif dist_s < 5.0:
        result["touch_type"] = "near_support"
        result["level"] = near_sup
        result["signal"] = "BUY" if last_c in bullish_rev else "NONE"
        result["confidence"] = 35
        result["reason"] = f"Near support ${near_sup:.2f} (dist: ${dist_s:.1f})"

    return result


# ─────────────────────────────────────────────────────────────
#  FULL ANALYSIS (patterns + S/R)
# ─────────────────────────────────────────────────────────────

def analyze_full(ohlcv: pd.DataFrame, sr_levels: Dict = None) -> Dict:
    """Full combined analysis: 30+ patterns + S/R touch."""
    if sr_levels is None:
        sr_levels = detect_swing_levels(ohlcv)

    patterns = detect_patterns(ohlcv)
    sr_touch = check_sr_touch(ohlcv, sr_levels)

    # Find strongest directional pattern
    strongest = None
    strongest_buy = None
    strongest_sell = None
    pattern_agreement = 0  # how many patterns agree

    for p in patterns:
        if strongest is None or p["strength"] > strongest["strength"]:
            strongest = p
        if p.get("direction") == "BUY":
            if strongest_buy is None or p["strength"] > strongest_buy["strength"]:
                strongest_buy = p
        elif p.get("direction") == "SELL":
            if strongest_sell is None or p["strength"] > strongest_sell["strength"]:
                strongest_sell = p

    # Count directional agreement
    buy_patterns = [p for p in patterns if p.get("direction") == "BUY"]
    sell_patterns = [p for p in patterns if p.get("direction") == "SELL"]
    if len(buy_patterns) > len(sell_patterns):
        pattern_agreement = len(buy_patterns)
    elif len(sell_patterns) > len(buy_patterns):
        pattern_agreement = -len(sell_patterns)

    # Assemble signal
    signal = "NONE"
    confidence = 0
    parts = []

    # Priority 1: S/R touch with pattern
    if sr_touch.get("signal") != "NONE":
        signal = sr_touch["signal"]
        confidence = sr_touch["confidence"]
        parts.append(sr_touch.get("reason", ""))

    # Priority 2: Strongest pattern
    if strongest and strongest.get("direction") != "NONE":
        if signal == "NONE":
            signal = strongest["direction"]
            confidence = strongest["strength"]
            parts.append(f"Pattern: {strongest['name']}")
        elif signal == strongest["direction"]:
            # Agreement bonus
            confidence = min(confidence + 18, 98)
            parts.append(f"+Pattern: {strongest['name']}")
        else:
            # Conflict — reduce but don't override S/R
            confidence = max(confidence - 25, 20)
            parts.append(f"conflict: {strongest['name']}")

    # Priority 3: Pattern agreement boost
    if abs(pattern_agreement) >= 3 and signal != "NONE":
        if (pattern_agreement > 0 and signal == "BUY") or (pattern_agreement < 0 and signal == "SELL"):
            confidence = min(confidence + 10, 100)
            parts.append(f"agreement: {abs(pattern_agreement)} patterns")

    if confidence <= 0 and strongest:
        confidence = 20
        parts.append("weak_pattern_only")

    return {
        "signal": signal,
        "confidence": min(confidence, 100),
        "reason": "; ".join(parts) if parts else "no pattern",
        "sr_touch": sr_touch,
        "patterns_detected": [p["name"] for p in patterns],
        "patterns_full": patterns,
        "pattern_agreement": pattern_agreement,
        "swing_levels": sr_levels,
        "nearest_resistance": sr_levels.get("nearest_resistance", 9999),
        "nearest_support": sr_levels.get("nearest_support", 0),
        "current_price": sr_levels.get("current_price", 0),
    }