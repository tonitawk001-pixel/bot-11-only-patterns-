"""
Technical indicator calculation module.

Provides clean, reusable functions for computing common technical
indicators from OHLCV data. All functions accept and return pandas
DataFrames/Series for seamless integration with the data feed.

Indicators implemented:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - EMA (Exponential Moving Average) at configurable periods
    - ATR (Average True Range) for volatility measurement
"""

import pandas as pd
import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger


# ------------------------------------------------------------------
# RSI
# ------------------------------------------------------------------

def compute_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes to
    evaluate overbought (typically > 70) or oversold (< 30) conditions.

    Args:
        close: Series of closing prices.
        period: Lookback period (default: 14).

    Returns:
        pd.Series: RSI values, with the first `period` entries as NaN.
    """
    if len(close) < period + 1:
        logger.warning(f"RSI: Not enough data points ({len(close)} < {period + 1})")
        return pd.Series([np.nan] * len(close), index=close.index)

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi.name = f"RSI_{period}"
    return rsi


# ------------------------------------------------------------------
# MACD
# ------------------------------------------------------------------

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute the Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two exponential moving averages
    and is used to identify changes in trend momentum.

    Args:
        close: Series of closing prices.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        pd.DataFrame with columns:
            - macd: MACD line (fast EMA - slow EMA)
            - signal: Signal line (EMA of MACD line)
            - histogram: MACD line - Signal line
    """
    if len(close) < slow + signal:
        logger.warning(f"MACD: Not enough data points ({len(close)} < {slow + signal})")
        return pd.DataFrame({
            "macd": pd.Series([np.nan] * len(close), index=close.index),
            "signal": pd.Series([np.nan] * len(close), index=close.index),
            "histogram": pd.Series([np.nan] * len(close), index=close.index),
        })

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })

    return result


# ------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------

def compute_ema(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Compute an Exponential Moving Average (EMA).

    EMAs give more weight to recent prices, making them more responsive
    to new information than simple moving averages.

    Args:
        close: Series of closing prices.
        period: EMA period (default: 20).

    Returns:
        pd.Series: EMA values, with the first `period - 1` entries as NaN.
    """
    if len(close) < period:
        logger.warning(f"EMA({period}): Not enough data ({len(close)} < {period})")
        return pd.Series([np.nan] * len(close), index=close.index)

    ema = close.ewm(span=period, adjust=False).mean()
    ema.name = f"EMA_{period}"
    return ema


def compute_multiple_emas(
    close: pd.Series,
    periods: list = None,
) -> pd.DataFrame:
    """
    Compute EMAs for multiple periods in one call.

    Args:
        close: Series of closing prices.
        periods: List of EMA periods (default: [20, 50, 200]).

    Returns:
        pd.DataFrame where each column is an EMA at the specified period.
    """
    if periods is None:
        periods = Config.EMA_PERIODS

    result = pd.DataFrame(index=close.index)
    for period in periods:
        result[f"EMA_{period}"] = compute_ema(close, period)

    return result


def compute_multiple_smas(
    close: pd.Series,
    periods: list = None,
) -> pd.DataFrame:
    """Compute SMAs for multiple periods."""
    if periods is None:
        periods = Config.SMA_PERIODS
    result = pd.DataFrame(index=close.index)
    for period in periods:
        result[f"SMA_{period}"] = compute_sma(close, period)
    return result


# ------------------------------------------------------------------
# ATR
# ------------------------------------------------------------------

def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Average True Range (ATR) for volatility measurement.

    ATR measures market volatility by decomposing the entire range of
    an asset price for a given period. Higher ATR indicates higher volatility.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: ATR period (default: 14).

    Returns:
        pd.Series: ATR values, with the first `period` entries as NaN.
    """
    if len(close) < period + 1:
        logger.warning(f"ATR: Not enough data ({len(close)} < {period + 1})")
        return pd.Series([np.nan] * len(close), index=close.index)

    # True Range components
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR as an EMA of True Range (standard approach)
    atr = true_range.ewm(span=period, adjust=False).mean()
    atr.name = f"ATR_{period}"

    return atr

# ------------------------------------------------------------------
# Bollinger Bands
# ------------------------------------------------------------------

def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    Returns DataFrame with columns: middle, upper, lower.
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower
    }, index=close.index)


# ------------------------------------------------------------------
# SMA (Simple Moving Average)
# ------------------------------------------------------------------

def compute_sma(
    close: pd.Series,
    period: int = 40,
) -> pd.Series:
    """
    Compute a Simple Moving Average (SMA).
    """
    if len(close) < period:
        return pd.Series([np.nan] * len(close), index=close.index)
    sma = close.rolling(window=period).mean()
    sma.name = f"SMA_{period}"
    return sma


# ------------------------------------------------------------------
# Stochastic Oscillator
# ------------------------------------------------------------------

def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> pd.DataFrame:
    """
    Compute the Stochastic Oscillator (%K and %D).

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        k_period: %K period (default: 14).
        d_period: %D period (default: 3).
        slowing: Slowing factor (default: 3).

    Returns:
        pd.DataFrame with columns: stoch_k, stoch_d
    """
    n = len(close)
    if n < k_period + d_period:
        logger.warning(f"Stochastic: Not enough data ({n} < {k_period + d_period})")
        return pd.DataFrame({
            "stoch_k": pd.Series([np.nan] * n, index=close.index),
            "stoch_d": pd.Series([np.nan] * n, index=close.index),
        })

    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    # Raw %K
    raw_k = ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)) * 100

    # Smoothed %K (slowing)
    stoch_k = raw_k.rolling(window=slowing).mean()

    # %D (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    result = pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }, index=close.index)

    return result


# ------------------------------------------------------------------
# Volume Moving Average
# ------------------------------------------------------------------

def compute_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute 20-period Volume Moving Average.
    """
    if len(volume) < period:
        return pd.Series([np.nan] * len(volume), index=volume.index)
    vma = volume.rolling(window=period).mean()
    vma.name = f"Volume_MA_{period}"
    return vma


# ------------------------------------------------------------------

def compute_sr_levels(ohlcv: pd.DataFrame) -> dict:
    """
    Compute basic Support and Resistance levels from recent history.
    Using Daily High/Low and Pivot Points.
    """
    recent = ohlcv.tail(20) # Last 20 candles
    support = recent['low'].min()
    resistance = recent['high'].max()
    
    # Pivot Points (Standard)
    last_close = ohlcv['close'].iloc[-1]
    last_high = ohlcv['high'].iloc[-1]
    last_low = ohlcv['low'].iloc[-1]
    
    pivot = (last_high + last_low + last_close) / 3
    r1 = (2 * pivot) - last_low
    s1 = (2 * pivot) - last_high
    
    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2)
    }


# ------------------------------------------------------------------
# ADX (Average Directional Index)
# ------------------------------------------------------------------

def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Average Directional Index (ADX).
    ADX > 25 = trending market, ADX < 20 = ranging.
    """
    n = len(close)
    if n < period * 2:
        return pd.Series([np.nan] * n, index=close.index)

    tr = pd.concat([(high - low).abs(),
                     (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = low.shift() - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    adx.name = f"ADX_{period}"
    return adx


def compute_directional_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Compute +DI and -DI (Directional Indicators).
    +DI > -DI = bullish momentum, +DI < -DI = bearish momentum.
    """
    n = len(close)
    if n < period * 2:
        return pd.DataFrame({
            "pdi": pd.Series([np.nan] * n, index=close.index),
            "ndi": pd.Series([np.nan] * n, index=close.index),
        })

    tr = pd.concat([(high - low).abs(),
                     (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = low.shift() - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    return pd.DataFrame({"pdi": pdi, "ndi": ndi}, index=close.index)


# ------------------------------------------------------------------
# Batch computation
# ------------------------------------------------------------------

def compute_all_indicators(
    ohlcv: pd.DataFrame,
) -> dict:
    """
    Compute all standard technical indicators from a single OHLCV DataFrame.

    Returns:
        dict: {
            "rsi": pd.Series,
            "macd": pd.DataFrame (macd, signal, histogram),
            "emas": pd.DataFrame (EMA_20, EMA_50, EMA_200),
            "smas": pd.DataFrame (SMA_40, SMA_200),
            "atr": pd.Series,
            "bb": pd.DataFrame (middle, upper, lower),
            "stochastic": pd.DataFrame (stoch_k, stoch_d),
            "volume_ma": pd.Series,
        }
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    logger.debug("Computing all technical indicators...")

    rsi = compute_rsi(close, period=Config.RSI_PERIOD)
    macd = compute_macd(close, fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
    emas = compute_multiple_emas(close, periods=Config.EMA_PERIODS)
    smas = compute_multiple_smas(close, periods=Config.SMA_PERIODS)
    atr = compute_atr(high, low, close, period=Config.ATR_PERIOD)
    bb = compute_bollinger_bands(close)
    stochastic = compute_stochastic(high, low, close,
                                     k_period=Config.STOCH_K_PERIOD,
                                     d_period=Config.STOCH_D_PERIOD,
                                     slowing=Config.STOCH_SLOWING)
    volume_ma = None
    if "tick_volume" in ohlcv.columns:
        volume_ma = compute_volume_ma(ohlcv["tick_volume"], period=20)
    adx_14 = compute_adx(high, low, close, period=14)
    di_14 = compute_directional_index(high, low, close, period=14)

    return {
        "rsi": rsi,
        "macd": macd,
        "bb": bb,
        "emas": emas,
        "smas": smas,
        "atr": atr,
        "stochastic": stochastic,
        "volume_ma": volume_ma,
        "adx_14": adx_14,
        "di_14": di_14,
    }