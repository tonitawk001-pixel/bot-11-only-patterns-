import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
for f in sorted(files):
    fp = os.path.join(data_dir, f)
    size_kb = os.path.getsize(fp) / 1024
    df = pd.read_csv(fp)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        print(f"{f:<30} {size_kb:>8.0f} KB  {df['Datetime'].min()}:{df['Datetime'].max()}  ({len(df)} rows)")
    else:
        print(f"{f:<30} {size_kb:>8.0f} KB  (no Datetime column)")