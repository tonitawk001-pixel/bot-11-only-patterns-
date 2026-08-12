"""Pull 3 years of XAUUSD M15 data in yearly chunks and save to CSV."""
import os, warnings
warnings.filterwarnings('ignore')
OUT = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5"
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print("MT5 INIT FAILED")
    raise SystemExit

chunks = [
    ("2023", datetime(2023,1,1), datetime(2024,1,1)),
    ("2024", datetime(2024,1,1), datetime(2025,1,1)),
    ("2025", datetime(2025,1,1), datetime(2026,1,1)),
    ("2026", datetime(2026,1,1), datetime(2026,8,13)),
]

frames = []
for name, s, e in chunks:
    r = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, s, e)
    if r is not None and len(r) > 0:
        d = pd.DataFrame(r)
        d['time'] = pd.to_datetime(d['time'], unit='s')
        d = d[['time','open','high','low','close','tick_volume']]
        frames.append(d)
        print(f"{name}: {len(d)} candles")
    else:
        print(f"{name}: NO DATA")

mt5.shutdown()

df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
df.columns = [c.lower() for c in df.columns]
out_path = os.path.join(OUT, "gold_m15_3y.csv")
df.to_csv(out_path, index=False)
print(f"\nTOTAL: {len(df)} candles saved to {out_path}")
print(f"Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
