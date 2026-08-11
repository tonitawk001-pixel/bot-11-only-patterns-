"""
DOWNLOAD 1 YEAR DATA — XAUUSD M5 and M15
========================================
Downloads 365 days of data from MT5 for backtesting.
"""

import os, sys, warnings
from datetime import datetime, timedelta
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not found. Please install it with 'pip install MetaTrader5'.")
    sys.exit(1)

# CONFIG
SYMBOL = "XAUUSD"
DAYS = 365
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def download():
    print(f"Initializing MT5...")
    if not mt5.initialize():
        print(f"initialize() failed, error code={mt5.last_error()}")
        return

    print(f"Downloading {DAYS} days of {SYMBOL}...")
    
    # M15
    print(f"Downloading M15...")
    rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 35000) # ~25k-30k candles in a year
    if rates_m15 is None:
        print(f"Failed to download M15, error={mt5.last_error()}")
    else:
        df15 = pd.DataFrame(rates_m15)
        df15['time'] = pd.to_datetime(df15['time'], unit='s', utc=True)
        df15.set_index('time', inplace=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df15.to_csv(os.path.join(OUTPUT_DIR, "XAUUSD_1y_M15.csv"))
        print(f"Saved M15: {len(df15)} bars")

    # M5
    print(f"Downloading M5...")
    # 365 days * 24 hours * 12 M5 bars = ~105,120 bars
    # MT5 has limits on how many bars can be requested in one go depending on settings
    # We'll try to get as much as possible, or do it in chunks if needed.
    # For now, let's try 90,000 bars (about 10 months).
    rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 90000)
    if rates_m5 is None:
        print(f"Failed to download M5, error={mt5.last_error()}")
    else:
        df5 = pd.DataFrame(rates_m5)
        df5['time'] = pd.to_datetime(df5['time'], unit='s', utc=True)
        df5.set_index('time', inplace=True)
        df5.to_csv(os.path.join(OUTPUT_DIR, "XAUUSD_1y_M5.csv"))
        print(f"Saved M5: {len(df5)} bars")

    mt5.shutdown()

if __name__ == "__main__":
    download()
