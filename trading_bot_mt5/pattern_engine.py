"""
PATTERN ENGINE — pattern-based trading logic (no backtested filters).

Every rule below is derived from a 3-year (2023-2026) M15 study of how each
indicator relates to forward gold price (3h horizon), measuring hit-rate
deviation from the ~52.6% bull-market baseline. Only CONSISTENT patterns
(same direction in all 4 yearly sub-periods) are encoded.

PATTERNS (hit-rate vs baseline):

BUY patterns (gold was a 3-year bull, so most robust signals are long):
  P1  RSI > 70 for 3+ bars        +4.1%  -> persistent overbought = momentum
      CONTINUATION, not reversal. The single strongest pattern.
  P2  Stochastic K < 20           +1.8%  -> fast oversold = mean-reversion bounce
  P3  Bollinger pos < 0.10        +1.8%  -> near lower band = bounce
  P4  RSI > 70 (single bar)       +2.2%  -> overbought = momentum continuation
  P5  Volume < 0.7x               +1.5%  -> quiet market drifts up
  P6  MACD histogram > 0          +0.6%  -> bullish momentum tilt (weak)
  P7  +DI < -DI & price<SMA50     +1.5%  -> bearish exhaustion = bounce

SELL patterns (weaker in a bull market):
  N1  RSI < 30 for 3+ bars        -1.2%  -> sustained oversold = downtrend continuation
  N2  RSI < 30                    -1.9%  -> oversold = momentum continues down
  N3  Stochastic K > 80           -0.9%  -> overbought (weak, near baseline)
  N4  MACD histogram < 0          -0.6%  -> bearish momentum tilt (weak)

Scoring: BUY = sum of positive pattern weights, SELL = sum of negative.
Thresholds tune how many patterns must align before a trade fires.
"""
import numpy as np
import pandas as pd


def _last(series, k=1):
    """Return the last k non-NaN values of a series as a list."""
    if series is None:
        return []
    s = series.dropna() if hasattr(series, 'dropna') else pd.Series(series).dropna()
    if len(s) == 0:
        return []
    return [float(v) for v in s.iloc[-k:].values]


def evaluate_patterns(i15, price, vol_ratio=None):
    """
    Evaluate all patterns and return a directional score.

    Args:
        i15: dict from compute_all_indicators (has rsi, macd, stochastic, bb,
             volume_ma, di_14, smas, atr)
        price: current close price (float)
        vol_ratio: current tick_volume / volume_ma (float, optional)

    Returns:
        dict with keys: direction ("BUY"/"SELL"/"NONE"), buy_score, sell_score,
        reasons (list of pattern names that fired).
    """
    buy_score = 0.0
    sell_score = 0.0
    reasons = []

    # ---- RSI patterns ----
    rsi = i15.get('rsi')
    if rsi is not None:
        rvals = _last(rsi, 3)
        if len(rvals) >= 3:
            # P1: RSI > 70 for 3+ consecutive bars (momentum continuation)
            if all(v > 70 for v in rvals):
                buy_score += 3.0
                reasons.append("RSI_overbought_3bars")
            # N1: RSI < 30 for 3+ consecutive bars (downtrend continuation)
            elif all(v < 30 for v in rvals):
                sell_score += 2.0
                reasons.append("RSI_oversold_3bars")
        if rvals:
            r = rvals[-1]
            # P4: RSI > 70 single bar
            if r > 70 and buy_score < 3.0:
                buy_score += 1.0
                if "RSI_overbought_3bars" not in reasons:
                    reasons.append("RSI_overbought")
            # N2: RSI < 30 single bar
            if r < 30 and sell_score < 2.0:
                sell_score += 1.0
                if "RSI_oversold_3bars" not in reasons:
                    reasons.append("RSI_oversold")

    # ---- Stochastic patterns ----
    stoch = i15.get('stochastic')
    if stoch is not None and 'stoch_k' in stoch.columns:
        k = _last(stoch['stoch_k'], 1)
        if k:
            kv = k[-1]
            # P2: K < 20 oversold bounce
            if kv < 20:
                buy_score += 2.0
                reasons.append("Stoch_oversold")
            # N3: K > 80 overbought
            elif kv > 80:
                sell_score += 1.0
                reasons.append("Stoch_overbought")

    # ---- Bollinger position ----
    bb = i15.get('bb')
    if bb is not None and 'upper' in bb.columns and 'lower' in bb.columns:
        u = _last(bb['upper'], 1); l = _last(bb['lower'], 1)
        if u and l and (u[-1] - l[-1]) > 0:
            pos = (price - l[-1]) / (u[-1] - l[-1])
            # P3: near lower band -> bounce
            if pos < 0.10:
                buy_score += 2.0
                reasons.append("BB_lower_band")
            elif pos > 0.90:
                sell_score += 1.0
                reasons.append("BB_upper_band")

    # ---- Volume ----
    # P5: low volume (quiet market) drifts up
    if vol_ratio is not None and vol_ratio < 0.7:
        buy_score += 1.0
        reasons.append("Volume_quiet")

    # ---- MACD histogram ----
    macd = i15.get('macd')
    if macd is not None and 'histogram' in macd.columns:
        h = _last(macd['histogram'], 1)
        if h:
            # P6 / N4: histogram sign tilt
            if h[-1] > 0:
                buy_score += 1.0
                reasons.append("MACD_hist_positive")
            else:
                sell_score += 1.0
                reasons.append("MACD_hist_negative")

    # ---- DI + trend (bearish exhaustion bounce) ----
    di = i15.get('di_14')
    smas = i15.get('smas')
    if di is not None and 'pdi' in di.columns and smas is not None and 'SMA_50' in smas.columns:
        pdi = _last(di['pdi'], 1); ndi = _last(di['ndi'], 1)
        sma50 = _last(smas['SMA_50'], 1)
        if pdi and ndi and sma50:
            # P7: bearish momentum but price below SMA -> bounce
            if pdi[-1] < ndi[-1] and price < sma50[-1]:
                buy_score += 1.0
                reasons.append("bearish_exhaustion_bounce")

    # ---- Decide direction ----
    direction = "NONE"
    if buy_score >= 3.0 and buy_score > sell_score:
        direction = "BUY"
    elif sell_score >= 3.0 and sell_score > buy_score:
        direction = "SELL"

    return {
        "direction": direction,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reasons": reasons,
    }
