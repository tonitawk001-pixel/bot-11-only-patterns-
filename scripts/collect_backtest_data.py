
import os
import sys
sys.path.append(os.path.abspath('trading_bot_mt5'))
from mt5_connection import MT5Connection

def collect():
    mt5 = MT5Connection()
    if not mt5.initialize():
        print('MT5 connection failed.')
        return
    data_dir = os.path.join('local_backtest', 'data')
    os.makedirs(data_dir, exist_ok=True)
    for tf in ['M5', 'M15']:
        df = mt5.get_candles('XAUUSD', tf, 5000)
        if df is not None:
            df.index.name = 'Datetime'
            df.to_csv(os.path.join(data_dir, f'XAUUSD_60d_{tf}.csv'))
            print(f'Saved {tf}')
    mt5.shutdown()
if __name__ == '__main__':
    collect()
