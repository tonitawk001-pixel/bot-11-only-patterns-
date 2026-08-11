"""Download 7 days of M1 data for ultra-frequent analysis testing."""
import os, yfinance as yf, pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMBOL = "GC=F"
os.makedirs(DATA_DIR, exist_ok=True)

# Download M1 data (7 days max from Yahoo)
print("Downloading M1 data (7 days)...")
df = yf.download(SYMBOL, interval="1m", period="7d", progress=False)
if df.empty:
    print("FAILED")
else:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)
    df.index.name = 'Datetime'
    df.reset_index(inplace=True)
    path = os.path.join(DATA_DIR, "XAUUSD_7d_M1.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} M1 candles to {path}")
    print(f"Range: {df['Datetime'].min()} -> {df['Datetime'].max()}")