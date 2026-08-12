# Pattern-Based Bot (bot-11-only-patterns)

This bot trades using **patterns discovered in the 3-year (2023-2026) relationship
between each indicator and gold price** — NOT backtested filters.

## Methodology

- Pulled 85,406 M15 candles of XAUUSD (2023-01-02 → 2026-08-12).
- For each indicator, measured the 3-hour forward return hit-rate vs the
  ~52.6% bull-market baseline, checked for consistency across all 4 yearly
  sub-periods (2023, 2024, 2025, 2026).
- Only patterns with a consistent direction in all 4 years were encoded.

## The Patterns (hit-rate deviation from baseline)

### BUY patterns (gold was a 3-year bull, so the robust signals are long)

| # | Pattern | Dev | Meaning |
|---|---|---|---|
| P1 | RSI > 70 for 3+ bars | +4.1% | Persistent overbought = momentum **continuation** (strongest) |
| P2 | Stochastic %K < 20 | +1.8% | Fast oversold = mean-reversion bounce |
| P3 | Bollinger position < 0.10 | +1.8% | Near lower band = bounce |
| P4 | RSI > 70 (single bar) | +2.2% | Overbought = momentum continuation |
| P5 | Volume < 0.7x | +1.5% | Quiet market drifts up |
| P6 | MACD histogram > 0 | +0.6% | Bullish momentum tilt (weak) |
| P7 | +DI < -DI & price < SMA50 | +1.5% | Bearish exhaustion = bounce |

### SELL patterns (weaker in a bull market)

| # | Pattern | Dev | Meaning |
|---|---|---|---|
| N1 | RSI < 30 for 3+ bars | -1.2% | Sustained oversold = downtrend continuation |
| N2 | RSI < 30 | -1.9% | Oversold = momentum continues down |
| N3 | Stochastic %K > 80 | -0.9% | Overbought (weak, near baseline) |
| N4 | MACD histogram < 0 | -0.6% | Bearish momentum tilt (weak) |

## Key Insight

The single strongest pattern is **RSI > 70 for 3+ bars → price CONTINUES up
(56.7% hit rate, +4.1% above baseline)**. This means "overbought" is a
*momentum* signal, not a reversal signal — the opposite of textbook RSI usage.

## Scoring

Each pattern adds a weighted score. A trade fires when:
- BUY score ≥ 3.0 (and > SELL score) → BUY
- SELL score ≥ 3.0 (and > BUY score) → SELL
- otherwise → NONE

Result over 3 years: ~22% BUY, ~6% SELL, ~72% no-trade (selective, bull-biased).

## Files

- `pattern_engine.py` — the pattern scoring logic (self-documenting)
- `main_patterns.py` — the bot entry point (uses pattern engine for direction)
- `pattern_macd.py`, `pattern_rest.py` — the pattern-discovery analysis
- `gold_m15_3y.csv` — the 3-year dataset used
