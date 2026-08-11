"""Check available data files and date ranges."""
import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
files = [
    "XAUUSD_2y_H1.csv",
    "XAUUSD_60d_M15.csv", 
    "XAUUSD_60d_M5.csv",
    "XAUUSD_1y_H1.csv",
    "XAUUSD_5y_D1.csv",
]

for f in files:
    fp = os.path.join(data_dir, f)
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        if "Datetime" in d.columns:
            d["Datetime"] = pd.to_datetime(d["Datetime"], utc=True)
            print(f"{f}: {d['Datetime'].min().strftime('%Y-%m-%d')} -> {d['Datetime'].max().strftime('%Y-%m-%d')} rows={len(d)} cols={list(d.columns)}")
        else:
            print(f"{f}: UNKNOWN FORMAT - cols={list(d.columns)} first_row={d.iloc[0].to_dict()}")
    else:
        print(f"{f}: NOT FOUND")