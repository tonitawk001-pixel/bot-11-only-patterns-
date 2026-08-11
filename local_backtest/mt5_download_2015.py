"""
Try to download 2015 gold data from MT5 terminal and run backtest.
Requires MT5 to be running and logged in.
"""
import os, sys, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_bot_mt5"))

import pandas as pd
import numpy as np

print("="*60)
print("  MT5 2015 DATA DOWNLOAD & BACKTEST")
print("="*60)

# Try to connect to MT5 and download data
try:
    import MetaTrader5 as mt5
    print("\nInitializing MT5 connection...")
    if not mt5.initialize():
        print("MT5 init FAILED. Make sure MT5 terminal is running.")
        print("Error:", mt5.last_error())
        mt5.shutdown()
        sys.exit(1)
    
    print("MT5 connected!")
    
    # Try H1 data for 2015
    print("\nDownloading 2015 H1 data...")
    rates_h1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, datetime(2015,1,1), datetime(2016,1,1))
    
    if rates_h1 is not None and len(rates_h1) > 0:
        df = pd.DataFrame(rates_h1)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        print(f"Got {len(df)} H1 candles!")
        print(f"Range: {df.index[0]} -> {df.index[-1]}")
        
        # Save for reuse
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "XAUUSD_2015_H1.csv")
        df.to_csv(out_path)
        print(f"Saved to: {out_path}")
    else:
        print("No H1 data for 2015 (broker limitation)")
        print("Trying M15...")
        rates_m15 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, datetime(2015,1,1), datetime(2015,2,1))
        if rates_m15 is not None and len(rates_m15) > 0:
            print(f"Got {len(rates_m15)} M15 candles for Jan 2015!")
        else:
            print("No intraday data for 2015. Trying D1...")
            rates_d1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, datetime(2015,1,1), datetime(2016,1,1))
            if rates_d1 is not None and len(rates_d1) > 0:
                df = pd.DataFrame(rates_d1)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)
                print(f"Got {len(df)} D1 candles for 2015!")
                out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "XAUUSD_2015_D1.csv")
                df.to_csv(out_path)
                print(f"Saved D1 data to: {out_path}")
            else:
                print("No 2015 data available from this broker")
    
    mt5.shutdown()
    
except ImportError:
    print("MetaTrader5 not installed. Try: pip install MetaTrader5")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")