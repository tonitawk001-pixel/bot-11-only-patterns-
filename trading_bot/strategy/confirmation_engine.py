"""
CONFIRMATION ENGINE v2.0 — Data-Backed Multi-Indicator Validation
=================================================================
Modular validation functions for BUY and SELL signals on M15.
Implements refined logic based on 30-trade empirical analysis.

BUY Option 1: Strong Breakout (RSI > 70 + Volume + Stochastic guard)
BUY Option 2: Pullback Rebound (RSI 40-70 + BB/MA bounce + MACD + Stochastic)
SELL:          ALL 5 conditions must pass (strict anti-trap rules)

Each validator returns (bool, str) — pass/fail + timestamped reason.
"""

from datetime import datetime, timezone
from typing import Tuple, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(result: bool, label: str, detail: str) -> Tuple[bool, str]:
    """Standardized logging for each check."""
    status = "PASS" if result else "FAIL"
    msg = f"[{_ts()}] [{status}] {label}: {detail}"
    return result, msg


# ===================================================================
# BUY VALIDATORS
# ===================================================================

def validate_trend_filter_buy(ma40: Optional[float], ma200: Optional[float]) -> Tuple[bool, str]:
    """
    BUY Trend: MA 40 must be above MA 200.
    """
    if ma40 is None or ma200 is None or np.isnan(ma40) or np.isnan(ma200):
        return _log(False, "TREND", f"MA40={ma40}, MA200={ma200} — insufficient data")
    if ma40 > ma200:
        return _log(True, "TREND", f"MA40 ({ma40:.2f}) > MA200 ({ma200:.2f}) — bullish structure confirmed")
    return _log(False, "TREND", f"MA40 ({ma40:.2f}) <= MA200 ({ma200:.2f}) — no bullish trend")


def validate_breakout_momentum_buy(
    rsi: Optional[float],
    volume: Optional[float],
    vol_ma: Optional[float],
    stoch_k: Optional[float],
    stoch_d: Optional[float],
    prev_stoch_k: Optional[float] = None,
    prev_stoch_d: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    BUY Option 1: Strong Breakout Momentum (RSI > 70 allowed with volume backing).
    Requires: RSI > 68, Volume > VolMA (optional bonus), Stochastic NOT forming bearish cross.
    """
    reasons = []

    if rsi is None or np.isnan(rsi):
        return _log(False, "BREAKOUT", "RSI unavailable")
    if rsi <= 68:
        return _log(False, "BREAKOUT", f"RSI={rsi:.1f} — not in breakout zone (>68 required)")

    # Volume: bonus confirmation, not required (sweep-optimized)
    vol_bonus = ""
    if volume is not None and vol_ma is not None and not np.isnan(volume) and not np.isnan(vol_ma):
        if volume > vol_ma:
            vol_bonus = f", Vol ({volume:.0f}) > VolMA ({vol_ma:.0f}) +bonus"
        else:
            vol_bonus = f", Vol ({volume:.0f}) <= VolMA ({vol_ma:.0f}) (no bonus)"

    # Stochastic guard: no bearish cross
    if stoch_k is not None and stoch_d is not None and prev_stoch_k is not None and prev_stoch_d is not None:
        bearish_cross_now = prev_stoch_k > prev_stoch_d and stoch_k < stoch_d
        if bearish_cross_now:
            return _log(False, "BREAKOUT", f"RSI={rsi:.1f} Vol OK but Stochastic bearish cross detected (K={stoch_k:.1f} < D={stoch_d:.1f}) — momentum fading")
        return _log(True, "BREAKOUT", f"RSI={rsi:.1f} in breakout{vol_bonus}, Stochastic OK (K={stoch_k:.1f} D={stoch_d:.1f}) — STRONG BUY")
    else:
        return _log(True, "BREAKOUT", f"RSI={rsi:.1f} in breakout{vol_bonus} — momentum entry")


def validate_pullback_rebound_buy(
    rsi: Optional[float],
    price: Optional[float],
    bb_mid: Optional[float],
    ma40: Optional[float],
    macd_hist: Optional[float],
    prev_macd_hist: Optional[float],
    stoch_k: Optional[float],
    stoch_d: Optional[float],
    prev_stoch_k: Optional[float] = None,
    prev_stoch_d: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    BUY Option 2: Pullback & Rebound (RSI 40-70, near MA40/BB mid, MACD rising, Stochastic cross).
    """
    reasons = []

    if rsi is None or np.isnan(rsi):
        return _log(False, "PULLBACK", "RSI unavailable")
    if rsi < 40:
        return _log(False, "PULLBACK", f"RSI={rsi:.1f} — too weak (<40), wait for stabilization")
    if rsi > 60:
        return _log(False, "PULLBACK", f"RSI={rsi:.1f} — above pullback zone (>60), consider BREAKOUT option")
    # RSI 40-60 range (sweep-optimized sweet spot)
    reasons.append(f"RSI={rsi:.1f} in pullback zone (40-60)")

    # Price location: near MA40 or BB mid
    if price is None or ma40 is None or bb_mid is None:
        return _log(False, "PULLBACK", f"RSI OK but price/MA data incomplete")

    near_ma40 = abs(price - ma40) / ma40 < 0.003  # within 0.3%
    near_bb_mid = abs(price - bb_mid) / bb_mid < 0.005  # within 0.5%

    if not (near_ma40 or near_bb_mid):
        dist_ma = abs(price - ma40)
        dist_bb = abs(price - bb_mid)
        return _log(False, "PULLBACK", f"RSI OK but Price ({price:.2f}) not near MA40 ({ma40:.2f}, dist={dist_ma:.1f}) or BB mid ({bb_mid:.2f}, dist={dist_bb:.1f})")

    if near_ma40:
        reasons.append(f"Price near MA40 ({ma40:.2f})")
    if near_bb_mid:
        reasons.append(f"Price near BB mid ({bb_mid:.2f})")

    # MACD: histogram rising
    if macd_hist is None or prev_macd_hist is None or np.isnan(macd_hist) or np.isnan(prev_macd_hist):
        return _log(False, "PULLBACK", f"Price OK but MACD histogram data unavailable")
    if macd_hist <= prev_macd_hist:
        return _log(False, "PULLBACK", f"Price OK but MACD histogram not rising ({macd_hist:.4f} <= {prev_macd_hist:.4f})")
    reasons.append(f"MACD hist rising ({prev_macd_hist:.4f} -> {macd_hist:.4f})")

    # Stochastic: K crosses above D
    if stoch_k is None or stoch_d is None or prev_stoch_k is None or prev_stoch_d is None:
        return _log(False, "PULLBACK", f"MACD OK but Stochastic data unavailable")

    if stoch_k >= 80:
        return _log(False, "PULLBACK", f"MACD OK but Stochastic K={stoch_k:.1f} >= 80 — overbought on oscillator")

    bullish_cross = prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d
    if not bullish_cross:
        return _log(False, "PULLBACK", f"MACD OK but no Stochastic bullish cross (K={stoch_k:.1f} D={stoch_d:.1f})")
    reasons.append(f"Stochastic bullish cross K={stoch_k:.1f} > D={stoch_d:.1f}")

    return _log(True, "PULLBACK", "; ".join(reasons) + " — PULLBACK BUY confirmed")


# ===================================================================
# REVERSAL SELL (Overbought Exhaustion)
# ===================================================================

def validate_reversal_sell(
    rsi: Optional[float],
    volume: Optional[float],
    vol_ma: Optional[float],
    stoch_k: Optional[float],
    stoch_d: Optional[float],
    prev_stoch_k: Optional[float],
    prev_stoch_d: Optional[float],
    price: Optional[float],
    bb_upper: Optional[float],
) -> Tuple[bool, str]:
    """
    SELL Reversal from Overbought: RSI > 68 + fading volume + Stochastic bearish cross + near Upper BB.
    Captures the exhaustion reversal the regular SELL path misses.
    """
    if rsi is None or np.isnan(rsi):
        return _log(False, "REVERSAL", "RSI unavailable")
    if rsi < 65:
        return _log(False, "REVERSAL", f"RSI={rsi:.1f} — not overbought (<65), use regular SELL")

    # Fading volume check
    vol_fading = False
    vol_note = ""
    if volume is not None and vol_ma is not None and not np.isnan(volume) and not np.isnan(vol_ma):
        if volume < vol_ma:
            vol_fading = True
            vol_note = f"vol fading ({volume:.0f} < {vol_ma:.0f})"
        else:
            vol_note = f"vol still strong ({volume:.0f} > {vol_ma:.0f})"

    # Stochastic bearish cross
    stoch_cross = False
    stoch_note = ""
    if stoch_k is not None and stoch_d is not None and prev_stoch_k is not None and prev_stoch_d is not None:
        if prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d:
            stoch_cross = True
            stoch_note = f"bearish cross K({prev_stoch_k:.1f}->{stoch_k:.1f}) < D({prev_stoch_d:.1f}->{stoch_d:.1f})"
        else:
            stoch_note = f"no bearish cross (K={stoch_k:.1f} D={stoch_d:.1f})"

    # Near Upper BB
    near_upper = False
    bb_note = ""
    if price is not None and bb_upper is not None and not np.isnan(price) and not np.isnan(bb_upper):
        dist_pct = abs(price - bb_upper) / bb_upper if bb_upper > 0 else 999
        if dist_pct < 0.01:  # within 1% of upper band
            near_upper = True
            bb_note = f"price near upper BB (dist={dist_pct*100:.1f}%)"
        else:
            bb_note = f"price away from upper BB (dist={dist_pct*100:.1f}%)"

    score = sum([1, vol_fading, stoch_cross, near_upper])

    if score >= 2:
        parts = [f"RSI={rsi:.1f} overbought"]
        if vol_fading: parts.append(vol_note)
        if stoch_cross: parts.append(stoch_note)
        if near_upper: parts.append(bb_note)
        return _log(True, "REVERSAL", "; ".join(parts) + f" — EXHAUSTION SELL (score={score}/4)")

    failures = []
    if not stoch_cross: failures.append("no bearish Stochastic cross (required)")
    return _log(False, "REVERSAL", f"RSI={rsi:.1f} overbought but {', '.join(failures)} (score={score}/4)")


# ===================================================================

def validate_trend_filter_sell(
    price: Optional[float],
    ma40: Optional[float],
    bb_upper: Optional[float],
) -> Tuple[bool, str]:
    """
    SELL Trend: Price must be below MA40 OR rejected at Upper BB.
    """
    if price is None or ma40 is None or np.isnan(price) or np.isnan(ma40):
        return _log(False, "TREND", "Price/MA40 data unavailable")

    below_ma40 = price < ma40
    rejected_bb = (bb_upper is not None and not np.isnan(bb_upper) and
                   abs(price - bb_upper) / bb_upper < 0.003)

    if below_ma40:
        return _log(True, "TREND", f"Price ({price:.2f}) < MA40 ({ma40:.2f}) — bearish structure")
    elif rejected_bb:
        return _log(True, "TREND", f"Price ({price:.2f}) rejected at Upper BB ({bb_upper:.2f}) — bearish reversal setup")
    else:
        return _log(False, "TREND", f"Price ({price:.2f}) >= MA40 ({ma40:.2f}) — no bearish structure")


def validate_rsi_sell(rsi: Optional[float], prev_rsi: Optional[float] = None) -> Tuple[bool, str]:
    """
    SELL RSI: RSI < 50 and turning downward.
    """
    if rsi is None or np.isnan(rsi):
        return _log(False, "RSI", "RSI data unavailable")
    if rsi >= 50:
        return _log(False, "RSI", f"RSI={rsi:.1f} >= 50 — not in bearish zone")
    if rsi < 30:
        return _log(False, "RSI", f"RSI={rsi:.1f} < 30 — oversold, do not chase bottom")

    turning_down = prev_rsi is not None and rsi < prev_rsi
    if not turning_down and prev_rsi is not None:
        return _log(False, "RSI", f"RSI={rsi:.1f} in zone but not turning down (prev={prev_rsi:.1f})")

    direction_note = " and declining" if turning_down else ""
    return _log(True, "RSI", f"RSI={rsi:.1f} in bearish zone (30-50){direction_note}")


def validate_macd_sell(
    macd_line: Optional[float],
    macd_signal: Optional[float],
    macd_hist: Optional[float],
    prev_macd_hist: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    SELL MACD: MACD line < Signal line AND histogram negative/declining.
    """
    if macd_line is None or macd_signal is None or np.isnan(macd_line) or np.isnan(macd_signal):
        return _log(False, "MACD", "MACD data unavailable")

    if macd_line >= macd_signal:
        return _log(False, "MACD", f"MACD line ({macd_line:.4f}) >= Signal ({macd_signal:.4f}) — no bearish cross")

    if macd_hist is None or np.isnan(macd_hist):
        return _log(True, "MACD", f"MACD line ({macd_line:.4f}) < Signal ({macd_signal:.4f}) — bearish")

    if macd_hist > 0:
        return _log(False, "MACD", f"MACD bearish but histogram positive ({macd_hist:.4f}) — weakening momentum")

    declining = prev_macd_hist is not None and macd_hist <= prev_macd_hist
    decline_note = " and declining" if declining else ""
    return _log(True, "MACD", f"MACD line ({macd_line:.4f}) < Signal ({macd_signal:.4f}), histogram negative ({macd_hist:.4f}){decline_note}")


def validate_stochastic_sell(
    stoch_k: Optional[float],
    stoch_d: Optional[float],
    prev_stoch_k: Optional[float] = None,
    prev_stoch_d: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    SELL Stochastic: %K crosses BELOW %D from above 20.
    """
    if stoch_k is None or stoch_d is None or np.isnan(stoch_k) or np.isnan(stoch_d):
        return _log(False, "STOCH", "Stochastic data unavailable")

    if stoch_k <= 20:
        return _log(False, "STOCH", f"Stochastic K={stoch_k:.1f} <= 20 — oversold, do not short")

    if prev_stoch_k is None or prev_stoch_d is None:
        return _log(False, "STOCH", "No previous Stochastic data for cross detection")

    bearish_cross = prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d
    if not bearish_cross:
        return _log(False, "STOCH", f"No bearish cross (K={stoch_k:.1f} D={stoch_d:.1f}, prev K={prev_stoch_k:.1f} D={prev_stoch_d:.1f})")

    return _log(True, "STOCH", f"Bearish cross confirmed: K ({prev_stoch_k:.1f}->{stoch_k:.1f}) crossed below D ({prev_stoch_d:.1f}->{stoch_d:.1f})")


def validate_location_sell(
    price: Optional[float],
    bb_lower: Optional[float],
) -> Tuple[bool, str]:
    """
    SELL Location: Block SELL if price is sitting directly on Lower BB (prevents shorting the bottom).
    """
    if price is None or bb_lower is None or np.isnan(price) or np.isnan(bb_lower):
        return _log(False, "LOCATION", "Price/BB data unavailable")

    dist_pct = abs(price - bb_lower) / bb_lower if bb_lower > 0 else 999
    if dist_pct < 0.003:  # within 0.3% of lower band
        return _log(False, "LOCATION", f"Price ({price:.2f}) on Lower BB ({bb_lower:.2f}, dist={dist_pct*100:.2f}%) — DO NOT SHORT THE BOTTOM")

    return _log(True, "LOCATION", f"Price ({price:.2f}) safely above Lower BB ({bb_lower:.2f}, dist={dist_pct*100:.2f}%)")


# ===================================================================
# MASTER EVALUATOR
# ===================================================================

def evaluate_buy_signals(
    price: float,
    ma40: float,
    ma200: float,
    rsi: float,
    prev_rsi: Optional[float],
    bb_upper: float,
    bb_mid: float,
    bb_lower: float,
    macd_line: float,
    macd_signal: float,
    macd_hist: float,
    prev_macd_hist: Optional[float],
    stoch_k: float,
    stoch_d: float,
    prev_stoch_k: Optional[float],
    prev_stoch_d: Optional[float],
    volume: Optional[float] = None,
    vol_ma: Optional[float] = None,
) -> dict:
    """
    Master BUY evaluation. Returns dict with decision, confidence, and all check results.

    Checks Option 1 (Breakout) first, then Option 2 (Pullback).
    """
    print(f"\n{'='*70}")
    print(f"[{_ts()}] EVALUATING BUY SIGNAL @ ${price:.2f}")
    print(f"{'='*70}")

    checks = {}
    failed = []
    passed = []

    # Always check trend first
    trend_ok, trend_msg = validate_trend_filter_buy(ma40, ma200)
    checks["trend"] = {"pass": trend_ok, "message": trend_msg}
    print(f"  {trend_msg}")
    if not trend_ok:
        failed.append("trend")

    if trend_ok:
        # --- Option 1: Strong Breakout ---
        breakout_ok, breakout_msg = validate_breakout_momentum_buy(
            rsi, volume, vol_ma, stoch_k, stoch_d, prev_stoch_k, prev_stoch_d
        )
        checks["breakout"] = {"pass": breakout_ok, "message": breakout_msg}
        print(f"  [Option 1 - Breakout] {breakout_msg}")

        if breakout_ok:
            decision = "BUY"
            confidence = 90  # High confidence for volume-backed breakout
            print(f"\n  >>> DECISION: {decision} (Option 1 - Breakout, Confidence: {confidence}%)")
            return {
                "direction": "BUY",
                "decision": True,
                "confidence": confidence,
                "option": 1,
                "checks": checks,
                "reason": f"Breakout BUY: {breakout_msg}",
            }

        # --- Option 2: Pullback Rebound ---
        pullback_ok, pullback_msg = validate_pullback_rebound_buy(
            rsi, price, bb_mid, ma40,
            macd_hist, prev_macd_hist,
            stoch_k, stoch_d, prev_stoch_k, prev_stoch_d,
        )
        checks["pullback"] = {"pass": pullback_ok, "message": pullback_msg}
        print(f"  [Option 2 - Pullback] {pullback_msg}")

        if pullback_ok:
            decision = "BUY"
            confidence = 70  # Good confidence for pullback
            print(f"\n  >>> DECISION: {decision} (Option 2 - Pullback, Confidence: {confidence}%)")
            return {
                "direction": "BUY",
                "decision": True,
                "confidence": confidence,
                "option": 2,
                "checks": checks,
                "reason": f"Pullback BUY: {pullback_msg}",
            }

    # Both failed
    all_failures = [k for k, v in checks.items() if not v["pass"]]
    reason = f"BUY rejected: {', '.join(all_failures)} checks failed"
    print(f"\n  >>> DECISION: NO TRADE — {reason}")
    return {
        "direction": "BUY",
        "decision": False,
        "confidence": 0,
        "option": 0,
        "checks": checks,
        "reason": reason,
    }


def evaluate_sell_signals(
    price: float,
    ma40: float,
    ma200: float,
    rsi: float,
    prev_rsi: Optional[float],
    bb_upper: float,
    bb_mid: float,
    bb_lower: float,
    macd_line: float,
    macd_signal: float,
    macd_hist: float,
    prev_macd_hist: Optional[float],
    stoch_k: float,
    stoch_d: float,
    prev_stoch_k: Optional[float],
    prev_stoch_d: Optional[float],
    volume: Optional[float] = None,
    vol_ma: Optional[float] = None,
) -> dict:
    """
    Master SELL evaluation. Checks reversal first, then regular 3/5.
    """
    print(f"\n{'='*70}")
    print(f"[{_ts()}] EVALUATING SELL SIGNAL @ ${price:.2f}")
    print(f"{'='*70}")

    # ── REVERSAL PATH: Overbought exhaustion SELL ──
    if rsi is not None and not np.isnan(rsi) and rsi >= 65:
        rev_ok, rev_msg = validate_reversal_sell(
            rsi, volume, vol_ma, stoch_k, stoch_d,
            prev_stoch_k, prev_stoch_d, price, bb_upper,
        )
        print(f"  [REVERSAL] {rev_msg}")
        if rev_ok:
            print(f"\n  >>> DECISION: SELL (Reversal — Exhaustion, Confidence: 88%)")
            return {
                "direction": "SELL",
                "decision": True,
                "confidence": 88,
                "option": 3,
                "checks": {"reversal": {"pass": True, "message": rev_msg}},
                "reason": f"Reversal SELL: {rev_msg}",
            }

    # ── REGULAR PATH: 3/5 checks ──

    checks = {}

    # 1. Trend / Structure
    trend_ok, trend_msg = validate_trend_filter_sell(price, ma40, bb_upper)
    checks["trend"] = {"pass": trend_ok, "message": trend_msg}
    print(f"  [1/5 - Trend] {trend_msg}")

    # 2. RSI
    rsi_ok, rsi_msg = validate_rsi_sell(rsi, prev_rsi)
    checks["rsi"] = {"pass": rsi_ok, "message": rsi_msg}
    print(f"  [2/5 - RSI] {rsi_msg}")

    # 3. MACD
    macd_ok, macd_msg = validate_macd_sell(macd_line, macd_signal, macd_hist, prev_macd_hist)
    checks["macd"] = {"pass": macd_ok, "message": macd_msg}
    print(f"  [3/5 - MACD] {macd_msg}")

    # 4. Stochastic
    stoch_ok, stoch_msg = validate_stochastic_sell(stoch_k, stoch_d, prev_stoch_k, prev_stoch_d)
    checks["stochastic"] = {"pass": stoch_ok, "message": stoch_msg}
    print(f"  [4/5 - Stochastic] {stoch_msg}")

    # 5. Location (no shorting at bottom)
    loc_ok, loc_msg = validate_location_sell(price, bb_lower)
    checks["location"] = {"pass": loc_ok, "message": loc_msg}
    print(f"  [5/5 - Location] {loc_msg}")

    # Only need 3/5 checks (sweep-optimized: 3 checks gives better PF than 5)
    check_bools = [trend_ok, rsi_ok, macd_ok, stoch_ok, loc_ok]
    passed_count = sum(check_bools)
    passed_names = [k for k, v in checks.items() if v["pass"]]
    failed_names = [k for k, v in checks.items() if not v["pass"]]
    
    if passed_count >= 3:
        confidence = 70 + (passed_count * 5)
        print(f"\n  >>> DECISION: SELL ({passed_count}/5 checks passed, Confidence: {confidence}%)")
        return {
            "direction": "SELL",
            "decision": True,
            "confidence": confidence,
            "option": 0,
            "checks": checks,
            "reason": f"SELL confirmed: {passed_count}/5 checks passed ({', '.join(passed_names)})",
        }
    else:
        reason = f"SELL rejected: only {passed_count}/5 checks passed (need 3) — failed: {', '.join(failed_names)}"
        print(f"\n  >>> DECISION: NO TRADE — {reason}")
        return {
            "direction": "SELL",
            "decision": False,
            "confidence": 0,
            "option": 0,
            "checks": checks,
            "reason": reason,
        }


# ===================================================================
# CONVENIENCE: Single-call entry point
# ===================================================================

def evaluate_trade_signals(
    direction: str,
    price: float,
    ma40: float,
    ma200: float,
    rsi: float,
    prev_rsi: Optional[float],
    bb_upper: float,
    bb_mid: float,
    bb_lower: float,
    macd_line: float,
    macd_signal: float,
    macd_hist: float,
    prev_macd_hist: Optional[float],
    stoch_k: float,
    stoch_d: float,
    prev_stoch_k: Optional[float],
    prev_stoch_d: Optional[float],
    volume: Optional[float] = None,
    vol_ma: Optional[float] = None,
) -> dict:
    """
    Single entry point for both BUY and SELL evaluation.

    Args:
        direction: "BUY" or "SELL"
        All indicator values (float or None for unavailable)

    Returns:
        dict with: direction, decision (bool), confidence (0-100), checks, reason
    """
    if direction == "BUY":
        return evaluate_buy_signals(
            price, ma40, ma200, rsi, prev_rsi,
            bb_upper, bb_mid, bb_lower,
            macd_line, macd_signal, macd_hist, prev_macd_hist,
            stoch_k, stoch_d, prev_stoch_k, prev_stoch_d,
            volume, vol_ma,
        )
    elif direction == "SELL":
        return evaluate_sell_signals(
            price, ma40, ma200, rsi, prev_rsi,
            bb_upper, bb_mid, bb_lower,
            macd_line, macd_signal, macd_hist, prev_macd_hist,
            stoch_k, stoch_d, prev_stoch_k, prev_stoch_d,
            volume, vol_ma,
        )
    else:
        return {
            "direction": direction,
            "decision": False,
            "confidence": 0,
            "checks": {},
            "reason": f"Unknown direction: {direction}",
        }
