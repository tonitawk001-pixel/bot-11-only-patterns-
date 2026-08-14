path = r"C:\visual studio code\ai bot\bot-6--master\bot-6--master\trading_bot_mt5\main_super.py"

with open(path, 'r', encoding='utf-8') as f:
    t = f.read()

changes = [
    ("sl_distance * 2.0)  # 1:2 RR", "sl_distance * 1.0)  # 1:1 RR"),
    ("sl_distance * 2.0)", "sl_distance * 1.0)"),
    ("1:2 RISK TO REWARD", "1:1 RISK TO REWARD"),
    ("TP = SL distance * 2.0 (minimum 1:2 RR)", "TP = SL distance * 1.0 (1:1 RR)"),
]

for old, new in changes:
    cnt = t.count(old)
    t = t.replace(old, new)
    print(f"{old!r} -> {new!r}: {cnt} occurrence(s)")

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(t)
print("DONE")
