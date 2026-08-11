"""
DOWNLOAD FULL 1-YEAR DATA — XAUUSD.r M1, M5, M15
===============================================
Downloads high-resolution historical data from local MT5 in chunks.
"""

import os, sys
import pandas as pd
try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not found.")
    sys.exit(1)

# CONFIG
SYMBOL = "XAUUSD.r"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def download_tf(symbol, timeframe, total_bars, filename):
    print(f"Downloading {timeframe} ({total_bars} bars total)...")
    
    chunk_size = 50000 
    all_rates = []
    
    for start_pos in range(0, total_bars, chunk_size):
        print(f"  Chunk: start={start_pos}, count={chunk_size}")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, chunk_size)
        if rates is None or len(rates) == 0:
            print(f"  Reached end of history or error: {mt5.last_error()}")
            break
        all_rates.append(pd.DataFrame(rates))
    
    if not all_rates:
        print(f"Failed to download {timeframe}")
        return None
    
    df = pd.concat(all_rates).drop_duplicates(subset=['time']).sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    print(f"Saved {filename}: {len(df)} bars")
    return df

def main():
    if not mt5.initialize():
        print(f"MT5 initialize failed, error={mt5.last_error()}")
        return

    print(f"MT5 Connected. Downloading history for {SYMBOL}...")
    
    # M15 and M5 already done
    # download_tf(SYMBOL, mt5.TIMEFRAME_M15, 35000, "XAUUSD_1y_M15.csv")
    # download_tf(SYMBOL, mt5.TIMEFRAME_M5, 100000, "XAUUSD_1y_M5.csv")
    
    # M1: try 300,000 bars first (approx 8-9 months)
    download_tf(SYMBOL, mt5.TIMEFRAME_M1, 300000, "XAUUSD_1y_M1.csv")

    mt5.shutdown()
    print("\nData download complete.")

if __name__ == "__main__":
    main()
