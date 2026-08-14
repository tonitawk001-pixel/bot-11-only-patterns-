path = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5\main_super.py"

with open(path, 'r', encoding='utf-8') as f:
    t = f.read()

changes = []

# 1. ADX fallback: 99 sentinel collides with "block >=50" -> use NaN (means "unavailable, skip ADX filtering")
old1 = '''                try:
                    m15_adx = float(i15.get('adx_14', pd.Series([0])).iloc[-1]) if 'adx_14' in i15 else 99
                except:
                    m15_adx = 99'''
new1 = '''                try:
                    m15_adx = float(i15['adx_14'].iloc[-1]) if 'adx_14' in i15 else float('nan')
                except:
                    m15_adx = float('nan')'''
changes.append((old1, new1, "ADX fallback NaN"))

# 2. Stale comments
changes.append(("# VOLATILITY FILTER: direction-specific. BUY skips dead (<0.1%). SELL requires 0.2-0.5%.",
                 "# VOLATILITY FILTER: direction-specific. BUY skips dead (<0.1%). SELL blocked <0.3%.", "comment ATR"))
changes.append(("# BOLLINGER BAND FILTER: block SELL when price in middle of bands (0.3-0.7)",
                 "# BOLLINGER BAND FILTER: block SELL when price in middle of bands (0.25-0.7)", "comment BB"))
changes.append(("# DIRECTIONAL MOMENTUM: trade WITH momentum, never against (+DI/-DI)",
                 "# DIRECTIONAL MOMENTUM (SELL-only): block SELL when +DI > -DI (bullish momentum)", "comment DI"))
changes.append(("# SELL VOLUME FILTER: sell requires volume 1.2x-2.5x (conviction, not panic). BUY unfiltered.",
                 "# SELL VOLUME FILTER: block SELL when volume <0.8x (too quiet). BUY unfiltered.", "comment Volume"))

for old, new, name in changes:
    cnt = t.count(old)
    if cnt == 0:
        print(f"NOT FOUND: {name}")
    else:
        t = t.replace(old, new)
        print(f"APPLIED: {name} ({cnt})")

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(t)

print("DONE")
